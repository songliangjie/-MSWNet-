import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules import DWT, SE_Block, ConvLayer, Conv, ConvLayer_Dis, Multi_feature_Fusion_Module, Multi_feature_transformer_Fusion_Module
from tcm import ConvTransBlock, ResidualBlockWithStride
from bifpn import BiFPNLayer
from wtconv2d import WTConv2d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Attention(nn.Module):
#     def __init__(self, dim):
#         super(Attention, self).__init__()
#         self.conv = nn.Conv2d(dim, dim, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         attention = self.sigmoid(self.conv(x))
#         return x * attention
#
# class Block(nn.Module):
#     def __init__(self, conv, dim):
#         super(Block, self).__init__()
#         self.conv1 = conv(dim, dim, 3, padding=1, bias=True)
#         self.act1 = nn.GELU()
#         self.conv2 = conv(dim, dim, 1, bias=True)
#         self.act2 = nn.GELU()
#         self.conv3 = conv(dim, dim, 3, padding=1, bias=True)
#         self.attention = Attention(dim)
#
#     def forward(self, x):
#         res1 = self.act1(self.conv1(x))
#         res2 = self.act2(self.conv2(x))
#         res = res1 + res2
#         res = x + res
#         res = self.attention(res)
#         res = self.conv3(res)
#         res = x + res
#         return res

def catcat(inputs1, inputs2):
    return torch.cat((inputs1, inputs2), 2)

class Cat(nn.Module):
    def __init__(self):
        super(Cat, self).__init__()

    def forward(self, x1, x2):
        return catcat(x1, x2)

class Block(nn.Module):
    def __init__(self, conv, dim):
        super(Block, self).__init__()
        # 使用无填充卷积，然后上采样恢复尺寸
        self.conv1 = nn.Conv2d(dim, dim, 3, bias=True)  # 无填充，尺寸会减小
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 1, bias=True)  # 1x1卷积，尺寸不变
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(dim, dim, 3, bias=True)  # 无填充，尺寸会减小
        
        self.attention = Attention(dim)
        
    def forward(self, x):
        # 保存原始尺寸
        original_size = x.shape[2:]
        
        # 第一个分支：3x3卷积
        res1 = self.conv1(x)
        res1 = self.act1(res1)
        # 上采样恢复原始尺寸
        res1 = F.interpolate(res1, size=original_size, mode='bilinear', align_corners=False)
        
        # 第二个分支：1x1卷积（尺寸不变）
        res2 = self.act2(self.conv2(x))
        
        # 融合两个分支
        res = res1 + res2
        res = x + res
        
        # 注意力机制
        res = self.attention(res)
        
        # 最终3x3卷积
        res = self.conv3(res)
        # 上采样恢复原始尺寸
        res = F.interpolate(res, size=original_size, mode='bilinear', align_corners=False)
        
        res = x + res
        return res

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention,self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((None, 1))
        self.max = nn.AdaptiveMaxPool2d((1, None))
        # 使用1x1卷积避免尺寸问题
        self.conv1x1 = nn.Conv2d(dim, dim//2, kernel_size=1, bias=True)
        self.conv1x1_mid = nn.Conv2d(dim//2, dim, kernel_size=1, bias=True)  # 中间层，将dim//2转为dim
        self.conv1x1_final = nn.Conv2d(dim, dim, kernel_size=1, bias=True)  # 最终层
        
        self.GELU = nn.GELU()
        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)
        
    def forward(self, x):
        batch_size, channel, height, width = x.size()
        original_size = (height, width)
        
        x_h = self.avg(x)
        x_w = self.max(x)
        x_h = torch.squeeze(x_h, 3)
        x_w = torch.squeeze(x_w, 2)
        x_h1 = x_h.unsqueeze(3)
        x_w1 = x_w.unsqueeze(2)
        x_h_w = catcat(x_h, x_w)
        x_h_w = x_h_w.unsqueeze(3)
        x_h_w = self.conv1x1(x_h_w)  # 1x1卷积，尺寸不变，通道数变为dim//2
        x_h_w = self.GELU(x_h_w)
        x_h_w = torch.squeeze(x_h_w, 3)
        x1, x2 = torch.split(x_h_w, [height, width], 2)
        x1 = x1.unsqueeze(3)
        x2 = x2.unsqueeze(2)
        
        # 使用1x1卷积替代3x3卷积，避免尺寸问题，并调整通道数
        x1 = self.conv1x1_mid(x1)  # 将dim//2转为dim
        x2 = self.conv1x1_mid(x2)  # 将dim//2转为dim
        
        mix1 = self.mix1(x_h1, x1)
        mix2 = self.mix2(x_w1, x2)
        
        # 最终使用1x1卷积
        x1 = self.conv1x1_final(mix1)
        x2 = self.conv1x1_final(mix2)
        
        matrix = torch.matmul(x1, x2)
        matrix = torch.sigmoid(matrix)
        final = torch.mul(x, matrix)
        final = x + final
        return final

class se_block(nn.Module):
    def __init__(self, inplanes, reduction=16):
        super(se_block, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inplanes, inplanes//reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // reduction, inplanes, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        input = x
        x = self.se(x)
        return input*x

class residual_block(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3):
        super(residual_block, self).__init__()
        # 使用无填充卷积，然后上采样恢复尺寸
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=True),  # 无填充，尺寸会减小
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, bias=True),  # 无填充，尺寸会减小
            se_block(out_channels, reduction=16)
        )

    def forward(self, x):
        input = x
        original_size = x.shape[2:]
        
        # 第一个卷积
        x = self.residual[0](x)
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        # 激活函数
        x = self.residual[1](x)
        
        # 第二个卷积
        x = self.residual[2](x)
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        # SE注意力
        x = self.residual[3](x)
        
        return input + x


class Generator_DWT(nn.Module):
    def __init__(self):
        super(Generator_DWT, self).__init__()
        # self.bifpn = BiFPNLayer()
        self.ConvBlock_1 = nn.Sequential(ConvLayer(1, 16, 3, 1, True), ConvLayer(16, 16, 3, 1, True))
        # #
        # self.ConvBlock_2 = nn.Sequential(
        #     ConvTransBlock(8, 8, 8, 8, 0, 'W'),
        #     ConvTransBlock(8, 8, 8, 8, 0, 'SW'), # 16通道分成8+8
        #     ResidualBlockWithStride(16, 32, stride=2)
        # )
        # self.ConvBlock_3 = nn.Sequential(
        #     ConvTransBlock(16, 16, 16, 8, 0, 'W'),
        #     ConvTransBlock(16, 16, 16, 8, 0, 'SW'), # 32通道分成16+16
        #     ResidualBlockWithStride(32, 64, stride=2)
        # )
        # self.ConvBlock_4 = nn.Sequential(
        #     ConvTransBlock(32, 32, 32, 8, 0, 'W'),  # 第一个 W-MSA
        #     ConvTransBlock(32, 32, 32, 8, 0, 'SW'),  # 第二个 SW-MSA
        #     ResidualBlockWithStride(64, 128, stride=2)
        # )
        # self.ConvBlock_5 = nn.Sequential(
        #     ConvTransBlock(64, 64, 64, 8, 0, 'W'),  # 第一个 W-MSA
        #     ConvTransBlock(64, 64, 64, 8, 0, 'SW'),  # 第二个 SW-MSA
        #     ResidualBlockWithStride(128, 256, stride=2)
        # )
        self.ConvBlock_2 = nn.Sequential(
            ConvTransBlock(8, 8, 8, 8, 0, 'W', block_type='simple'),
            ConvTransBlock(8, 8, 8, 8, 0, 'SW', block_type='simple'),
            ResidualBlockWithStride(16, 32, stride=2)
        )
        self.ConvBlock_3 = nn.Sequential(
            ConvTransBlock(16, 16, 16, 8, 0, 'W', block_type='simple'),
            ConvTransBlock(16, 16, 16, 8, 0, 'SW', block_type='simple'),
            ResidualBlockWithStride(32, 64, stride=2)
        )
        self.ConvBlock_4 = nn.Sequential(
            ConvTransBlock(32, 32, 32, 8, 0, 'W', block_type='simple'),
            ConvTransBlock(32, 32, 32, 8, 0, 'SW', block_type='simple'),
            ResidualBlockWithStride(64, 128, stride=2)
        )
        self.ConvBlock_5 = nn.Sequential(
            ConvTransBlock(64, 64, 64, 8, 0, 'W', block_type='complex'),
            ConvTransBlock(64, 64, 64, 8, 0, 'SW', block_type='complex'),
            ResidualBlockWithStride(128, 256, stride=2)
        )
        # self.ConvBlock_1 = nn.Sequential(ConvLayer(1, 16, 3, 1, True), ConvLayer(16, 16, 3, 1, True))
        # self.ConvBlock_2 = nn.Sequential(ConvLayer(16, 32, 3, 1, True), ConvLayer(32, 32, 3, 2, True))
        # self.ConvBlock_3 = nn.Sequential(ConvLayer(32, 64, 3, 1, True), ConvLayer(64, 64, 3, 2, True))
        # self.ConvBlock_4 = nn.Sequential(ConvLayer(64, 128, 3, 1, True), ConvLayer(128, 128, 3, 2, True))
        # self.ConvBlock_5 = nn.Sequential(ConvLayer(128, 256, 3, 1, True), ConvLayer(256, 256, 3, 2, True))

        # Wavelet
        self.DWT_1 = DWT(16)
        self.DWT_2 = DWT(32)
        self.DWT_3 = DWT(64)
        self.DWT_4 = DWT(128)

        # MFM : Multi_feature Fusion Module
        self.MFM_1 = Multi_feature_Fusion_Module(128)
        self.MFM_2 = Multi_feature_Fusion_Module(64)
        self.MFM_3 = Multi_feature_Fusion_Module(32)
        self.MFM_4 = Multi_feature_transformer_Fusion_Module(16)

    def forward(self, Input_Ir, Input_Vis):
        # Encoder_Ir
        Ir_1 = self.ConvBlock_1(Input_Ir)
        FeatsHigh_Ir_1, LL1 = self.DWT_1(Ir_1)
        #print('ConvBlock_1 out shape:', Ir_1.shape)


        Ir_2 = self.ConvBlock_2(Ir_1)
        FeatsHigh_Ir_2, LL2 = self.DWT_2(Ir_2)
        #print('ConvBlock_2 out shape:', Ir_2.shape)


        Ir_3 = self.ConvBlock_3(Ir_2 + LL1)
        FeatsHigh_Ir_3, LL3 = self.DWT_3(Ir_3)

        #print('ConvBlock_3 out shape:', Ir_3.shape)

        Ir_4 = self.ConvBlock_4(Ir_3 + LL2)
        FeatsHigh_Ir_4, LL4 = self.DWT_4(Ir_4)
        #print('ConvBlock_4 out shape:', Ir_4.shape)

        Ir_5 = self.ConvBlock_5(Ir_4 + LL3)
        Ir_5 = Ir_5 + LL4

        # Encoder_Vis
        Vis_1 = self.ConvBlock_1(Input_Vis)
        FeatsHigh_Vis_1, LL1 = self.DWT_1(Vis_1)

        Vis_2 = self.ConvBlock_2(Vis_1)
        FeatsHigh_Vis_2, LL2 = self.DWT_2(Vis_2)

        Vis_3 = self.ConvBlock_3(Vis_2 + LL1)
        FeatsHigh_Vis_3, LL3 = self.DWT_3(Vis_3)

        Vis_4 = self.ConvBlock_4(Vis_3 + LL2)
        FeatsHigh_Vis_4, LL4 = self.DWT_4(Vis_4)

        Vis_5 = self.ConvBlock_5(Vis_4 + LL3)
        Vis_5 = Vis_5 + LL4

        # FusionLayer - 使用加权融合
        # 计算注意力权重
        attention_ir = torch.sigmoid(Ir_5)
        attention_vis = torch.sigmoid(Vis_5)
        # 归一化权重
        total_attention = attention_ir + attention_vis + 1e-6
        attention_ir = attention_ir / total_attention
        attention_vis = attention_vis / total_attention
        # 加权融合
        FusedImage = attention_ir * Ir_5 + attention_vis * Vis_5

        # Decoder
        Recon_1 = self.MFM_1(FusedImage, Ir_5, FeatsHigh_Ir_4, Vis_5, FeatsHigh_Vis_4)
        # print(f'Recon_1 shape: {Recon_1.shape}')
        Recon_2 = self.MFM_2(Recon_1, Ir_4, FeatsHigh_Ir_3, Vis_4, FeatsHigh_Vis_3)
        #print(f'Recon_2 shape: {Recon_2.shape}')
        Recon_3 = self.MFM_3(Recon_2, Ir_3, FeatsHigh_Ir_2, Vis_3, FeatsHigh_Vis_2)
        #print(f'Recon_3 shape: {Recon_3.shape}')
        Recon_4 = self.MFM_4(Recon_3, Ir_2, FeatsHigh_Ir_1, Vis_2, FeatsHigh_Vis_1)
        #print(f'Recon_4 shape: {Recon_4.shape}')

        output = Recon_4
        bifpn = BiFPNLayer()
        # output = self.bifpn(Recon_1, Recon_2, Recon_3, Recon_4)
        #print(output.shape)
        return output


class D_IR(nn.Module):
    def __init__(self):
        super(D_IR, self).__init__()

        self.Conv_1 = nn.Sequential(ConvLayer_Dis(4, 4, 3, 2, 1), ConvLayer_Dis(4, 8, 3, 1, 1))
        self.Conv_2 = nn.Sequential(ConvLayer_Dis(8, 8, 3, 2, 1), ConvLayer_Dis(8, 16, 3, 1, 1))
        self.Conv_3 = nn.Sequential(ConvLayer_Dis(16, 16, 3, 2, 1), ConvLayer_Dis(16, 32, 3, 1, 1))
        self.Conv_4 = nn.Sequential(ConvLayer_Dis(32, 32, 3, 2, 1), ConvLayer_Dis(32, 64, 3, 1, 1))

        self.SE_Block_1 = SE_Block(8, is_dis=True)
        self.SE_Block_2 = SE_Block(16, is_dis=True)
        self.SE_Block_3 = SE_Block(32, is_dis=True)
        self.SE_Block_4 = SE_Block(64, is_dis=True)

        self.ConvFC = nn.Sequential(ConvLayer_Dis(64, 64, 1, 1, 1), ConvLayer_Dis(64, 64, 1, 1, 1))

    def forward(self, x):
        x1 = self.Conv_1(x)
        w1 = self.SE_Block_1(x1)
        x1 = w1 * x1

        x2 = self.Conv_2(x1)
        w2 = self.SE_Block_2(x2)
        x2 = w2 * x2

        x3 = self.Conv_3(x2)
        w3 = self.SE_Block_3(x3)
        x3 = w3 * x3

        x4 = self.Conv_4(x3)
        w4 = self.SE_Block_4(x4)
        out = w4 * x4

        out = self.ConvFC(out)

        return out


class D_VI(nn.Module):
    def __init__(self):
        super(D_VI, self).__init__()

        self.Conv_1 = nn.Sequential(ConvLayer_Dis(4, 4, 3, 2, 1), ConvLayer_Dis(4, 8, 3, 1, 1))
        self.Conv_2 = nn.Sequential(ConvLayer_Dis(8, 8, 3, 2, 1), ConvLayer_Dis(8, 16, 3, 1, 1))
        self.Conv_3 = nn.Sequential(ConvLayer_Dis(16, 16, 3, 2, 1), ConvLayer_Dis(16, 32, 3, 1, 1))
        self.Conv_4 = nn.Sequential(ConvLayer_Dis(32, 32, 3, 2, 1), ConvLayer_Dis(32, 64, 3, 1, 1))

        self.SE_Block_1 = SE_Block(8)
        self.SE_Block_2 = SE_Block(16)
        self.SE_Block_3 = SE_Block(32)
        self.SE_Block_4 = SE_Block(64)

        self.ConvFC = nn.Sequential(ConvLayer_Dis(64, 64, 1, 1, 1), ConvLayer_Dis(64, 64, 1, 1, 1))

    def forward(self, x):
        x1 = self.Conv_1(x)
        w1 = self.SE_Block_1(x1)
        x1 = w1 * x1

        x2 = self.Conv_2(x1)
        w2 = self.SE_Block_2(x2)
        x2 = w2 * x2

        x3 = self.Conv_3(x2)
        w3 = self.SE_Block_3(x3)
        x3 = w3 * x3

        x4 = self.Conv_4(x3)
        w4 = self.SE_Block_4(x4)
        out = w4 * x4

        out = self.ConvFC(out)

        return out



# class D_Fusion(nn.Module):
#     def __init__(self):
#         super(D_Fusion, self).__init__()
#
#         self.Conv_1 = nn.Sequential(ConvLayer_Dis(4, 4, 3, 2, 1), ConvLayer_Dis(4, 8, 3, 1, 1))
#         self.Conv_2 = nn.Sequential(ConvLayer_Dis(8, 8, 3, 2, 1), ConvLayer_Dis(8, 16, 3, 1, 1))
#         self.Conv_3 = nn.Sequential(ConvLayer_Dis(16, 16, 3, 2, 1), ConvLayer_Dis(16, 32, 3, 1, 1))
#         self.Conv_4 = nn.Sequential(ConvLayer_Dis(32, 32, 3, 2, 1), ConvLayer_Dis(32, 64, 3, 1, 1))
#
#         self.SE_Block_1 = SE_Block(8)
#         self.SE_Block_2 = SE_Block(16)
#         self.SE_Block_3 = SE_Block(32)
#         self.SE_Block_4 = SE_Block(64)
#
#         self.ConvFC = nn.Sequential(ConvLayer_Dis(64, 64, 1, 1, 1), ConvLayer_Dis(64, 64, 1, 1, 1))
#
#     def forward(self, x):
#         x1 = self.Conv_1(x)
#         w1 = self.SE_Block_1(x1)
#         x1 = w1 * x1
#
#         x2 = self.Conv_2(x1)
#         w2 = self.SE_Block_2(x2)
#         x2 = w2 * x2
#
#         x3 = self.Conv_3(x2)
#         w3 = self.SE_Block_3(x3)
#         x3 = w3 * x3
#
#         x4 = self.Conv_4(x3)
#         w4 = self.SE_Block_4(x4)
#         out = w4 * x4
#
#         out = self.ConvFC(out)
#
#         return out

# class D_Fusion(nn.Module):
#     def __init__(self):
#         super(D_Fusion, self).__init__()
#
#         # 特征提取器（降低Dropout概率）
#         self.feature_extractor = nn.ModuleList([
#             nn.Sequential(
#                 ConvLayer_Dis(4, 8, 3, 2, 1),
#                 nn.Dropout2d(p=0.1),  # 从0.3改为0.1
#                 ConvLayer_Dis(8, 8, 3, 1, 1),
#                 nn.Dropout2d(p=0.1)  # 从0.3改为0.1
#             ),
#             nn.Sequential(
#                 ConvLayer_Dis(8, 16, 3, 2, 1),
#                 nn.Dropout2d(p=0.1),  # 从0.3改为0.1
#                 ConvLayer_Dis(16, 16, 3, 1, 1),
#                 nn.Dropout2d(p=0.1)  # 从0.3改为0.1
#             ),
#             nn.Sequential(
#                 ConvLayer_Dis(16, 32, 3, 2, 1),
#                 nn.Dropout2d(p=0.1),  # 从0.3改为0.1
#                 ConvLayer_Dis(32, 32, 3, 1, 1),
#                 nn.Dropout2d(p=0.1)  # 从0.3改为0.1
#             ),
#             nn.Sequential(
#                 ConvLayer_Dis(32, 64, 3, 2, 1),
#                 nn.Dropout2d(p=0.1),  # 从0.3改为0.1
#                 ConvLayer_Dis(64, 64, 3, 1, 1),
#                 nn.Dropout2d(p=0.1)  # 从0.3改为0.1
#             )
#         ])
#
#         # 特征融合层（添加BatchNorm）
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(120, 1, kernel_size=1),
#             nn.BatchNorm2d(1)
#         )
#
#         # 判别器输出层（添加BatchNorm和Sigmoid）
#         self.disc_conv = nn.Sequential(
#             nn.Conv2d(64, 1, kernel_size=1),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()  # 确保输出在[0,1]范围内
#         )
#
#         # SE注意力模块
#         self.SE_Block_1 = SE_Block(8)
#         self.SE_Block_2 = SE_Block(16)
#         self.SE_Block_3 = SE_Block(32)
#         self.SE_Block_4 = SE_Block(64)
#
#     def extract_features(self, x):
#         features = []
#         for i, layer in enumerate(self.feature_extractor):
#             x = layer(x)
#             if i == 0:
#                 x = self.SE_Block_1(x) * x
#             elif i == 1:
#                 x = self.SE_Block_2(x) * x
#             elif i == 2:
#                 x = self.SE_Block_3(x) * x
#             elif i == 3:
#                 x = self.SE_Block_4(x) * x
#             features.append(x)
#         return features
#
#     def forward(self, fake_fusion, teacher_fusion=None, mc_samples=10):
#         if teacher_fusion is None:
#             # 判别器模式：只返回最后一层的特征
#             features = self.extract_features(fake_fusion)
#             disc_out = self.disc_conv(features[-1])
#             # 上采样到与输入相同的大小
#             disc_out = F.interpolate(disc_out, size=fake_fusion.shape[2:], mode='bilinear', align_corners=False)
#             return disc_out
#         else:
#             # 不确定性图模式：使用 MC Dropout
#             predictions = []
#             for _ in range(mc_samples):
#                 # 提取特征
#                 fake_features = self.extract_features(fake_fusion)
#                 teacher_features = self.extract_features(teacher_fusion)
#
#                 # 计算特征差异
#                 diff_features = []
#                 for f_fake, f_teacher in zip(fake_features, teacher_features):
#                     diff = torch.abs(f_fake - f_teacher)  # L1距离
#                     # 将每个差异特征上采样到与输入相同的大小
#                     diff = F.interpolate(diff, size=fake_fusion.shape[2:], mode='bilinear', align_corners=False)
#                     diff_features.append(diff)
#
#                 # 融合差异特征
#                 diff_sum = torch.cat(diff_features, dim=1)  # 在通道维度上拼接
#                 diff_sum = self.fusion_conv(diff_sum)  # 使用1x1卷积融合所有特征
#
#                 # 生成不确定性图
#                 uncertainty_map = torch.sigmoid(diff_sum)  # 使用sigmoid确保输出在[0,1]范围内
#                 predictions.append(uncertainty_map)
#
#             # 计算均值和方差
#             stacked_preds = torch.stack(predictions, dim=0)
#             mean_pred = stacked_preds.mean(dim=0)
#             var_pred = stacked_preds.var(dim=0)
#
#             return mean_pred, var_pred

# class D_Fusion(nn.Module):
#     def __init__(self):
#         super(D_Fusion, self).__init__()
#
#         # 特征提取器（降低Dropout概率）
#         self.feature_extractor = nn.ModuleList([
#             nn.Sequential(
#                 ConvLayer_Dis(4, 8, 3, 2, 1),
#                 nn.Dropout2d(p=0.1),  # 从0.3改为0.1
#                 ConvLayer_Dis(8, 8, 3, 1, 1),
#                 nn.Dropout2d(p=0.1)  # 从0.3改为0.1
#             ),
#             nn.Sequential(
#                 ConvLayer_Dis(8, 16, 3, 2, 1),
#                 nn.Dropout2d(p=0.1),  # 从0.3改为0.1
#                 ConvLayer_Dis(16, 16, 3, 1, 1),
#                 nn.Dropout2d(p=0.1)  # 从0.3改为0.1
#             ),
#             nn.Sequential(
#                 ConvLayer_Dis(16, 32, 3, 2, 1),
#                 nn.Dropout2d(p=0.1),  # 从0.3改为0.1
#                 ConvLayer_Dis(32, 32, 3, 1, 1),
#                 nn.Dropout2d(p=0.1)  # 从0.3改为0.1
#             ),
#             nn.Sequential(
#                 ConvLayer_Dis(32, 64, 3, 2, 1),
#                 nn.Dropout2d(p=0.1),  # 从0.3改为0.1
#                 ConvLayer_Dis(64, 64, 3, 1, 1),
#                 nn.Dropout2d(p=0.1)  # 从0.3改为0.1
#             )
#         ])
#
#         # 特征融合层（添加BatchNorm）
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(120, 1, kernel_size=1),
#             nn.BatchNorm2d(1)
#         )
#
#         # 判别器输出层（添加BatchNorm和Sigmoid）
#         self.disc_conv = nn.Sequential(
#             nn.Conv2d(64, 1, kernel_size=1),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()  # 确保输出在[0,1]范围内
#         )
#
#         # SE注意力模块
#         self.SE_Block_1 = SE_Block(8)
#         self.SE_Block_2 = SE_Block(16)
#         self.SE_Block_3 = SE_Block(32)
#         self.SE_Block_4 = SE_Block(64)
#
#     def extract_features(self, x):
#         features = []
#         for i, layer in enumerate(self.feature_extractor):
#             x = layer(x)
#             if i == 0:
#                 x = self.SE_Block_1(x) * x
#             elif i == 1:
#                 x = self.SE_Block_2(x) * x
#             elif i == 2:
#                 x = self.SE_Block_3(x) * x
#             elif i == 3:
#                 x = self.SE_Block_4(x) * x
#             features.append(x)
#         return features
#
#     def forward(self, fake_fusion, teacher_fusion=None, mc_samples=10):
#         if teacher_fusion is None:
#             # 判别器模式：只返回最后一层的特征
#             features = self.extract_features(fake_fusion)
#             disc_out = self.disc_conv(features[-1])
#             # 上采样到与输入相同的大小
#             disc_out = F.interpolate(disc_out, size=fake_fusion.shape[2:], mode='bilinear', align_corners=False)
#             return disc_out
#         else:
#             # 不确定性图模式：使用 MC Dropout
#             predictions = []
#             for _ in range(mc_samples):
#                 # 提取特征
#                 fake_features = self.extract_features(fake_fusion)
#                 teacher_features = self.extract_features(teacher_fusion)
#
#                 # 计算特征差异
#                 diff_features = []
#                 for f_fake, f_teacher in zip(fake_features, teacher_features):
#                     diff = torch.abs(f_fake - f_teacher)  # L1距离
#                     # 将每个差异特征上采样到与输入相同的大小
#                     diff = F.interpolate(diff, size=fake_fusion.shape[2:], mode='bilinear', align_corners=False)
#                     diff_features.append(diff)
#
#                 # 融合差异特征
#                 diff_sum = torch.cat(diff_features, dim=1)  # 在通道维度上拼接
#                 diff_sum = self.fusion_conv(diff_sum)  # 使用1x1卷积融合所有特征
#
#                 # 生成不确定性图
#                 uncertainty_map = torch.sigmoid(diff_sum)  # 使用sigmoid确保输出在[0,1]范围内
#                 predictions.append(uncertainty_map)
#
#             # 计算均值和方差
#             stacked_preds = torch.stack(predictions, dim=0)
#             mean_pred = stacked_preds.mean(dim=0)
#             var_pred = stacked_preds.var(dim=0)
#
#             return mean_pred, var_pred
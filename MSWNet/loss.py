from torch import nn
import torch
import torch.nn.functional as F
from math import exp
from args import args
from Modules import DWT_2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 新增：RGB到灰度的转换函数
def rgb_to_gray(x):
    """
    将RGB图像转换为灰度图
    Args:
        x: 输入图像，形状为 [B, C, H, W]，C可以是1或3
    Returns:
        灰度图像，形状为 [B, 1, H, W]
    """
    if x.size(1) == 3:
        # RGB转灰度：0.299*R + 0.587*G + 0.114*B
        return 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
    elif x.size(1) == 1:
        # 已经是灰度图，直接返回
        return x
    else:
        raise ValueError(f"Unsupported channel number: {x.size(1)}")


# ===================== 内容损失总和 =====================
class g_content_loss(nn.Module):
    """
    内容损失总和：由SSIM、Intensity、Grad三项损失加权组成。
    作用：保证融合图像在结构、亮度、细节等多方面都尽量兼顾红外和可见光源图像的优势。
    """
    def __init__(self):
        super(g_content_loss, self).__init__()
        self.SSIM_loss = L_SSIM()
        self.Grad_loss = L_Grad()
        self.Intensity_loss = L_Intensity()

    def forward(self, img_ir, img_vi, img_fusion):
        # 结构相似性损失
        SSIM_loss = self.SSIM_loss(img_ir, img_vi, img_fusion)
        # 梯度损失
        Grad_loss = self.Grad_loss(img_ir, img_vi, img_fusion)
        # 强度损失
        Intensity_loss = self.Intensity_loss(img_ir, img_vi, img_fusion)
        # 总损失加权和
        total_loss = (
            args.weight_SSIM * SSIM_loss
            + args.weight_Grad * Grad_loss
            + args.weight_Intensity * Intensity_loss
        )
        return total_loss, SSIM_loss, Intensity_loss, Grad_loss


# ===================== SSIM 损失（结构相似性损失） =====================
class L_SSIM(nn.Module):
    """
    作用：衡量融合图像与源图像在结构上的相似性，提升主观视觉质量。
    实现：
    - 对红外、可见光、融合图像分别做小波分解，得到LL、LH、HL、HH四个子带。
    - 对每个子带分别计算SSIM。
    - 最终损失是红外和可见光两个模态的SSIM加权和。
    """
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv = Sobelxy()
        self.DWT = DWT_2D(wavename='haar')

    def forward(self, image_A, image_B, image_fused):
        r = args.r

        # 新增：转换为灰度图进行损失计算
        image_A_gray = rgb_to_gray(image_A)
        image_B_gray = rgb_to_gray(image_B)
        image_fused_gray = rgb_to_gray(image_fused)


        #红外和可见光两个模态的SSIM加权和
        ssim_ir = ssim(image_A_gray, image_fused_gray)
        ssim_vis = ssim(image_B_gray, image_fused_gray)
        Loss_SSIM = ((1-ssim_ir) + (1-ssim_vis))/2
        return Loss_SSIM


# ===================== Intensity 损失（强度损失） =====================
# class L_Intensity(nn.Module):
#     """
#     作用：保持融合图像的亮度（强度）信息，防止信息丢失。
#     实现：
#     - 对三幅图像做小波分解。
#     - 对每个子带，融合图像与红外/可见光的最大值做L1损失。
#     - 各子带损失加权求和。
#     """
#     def __init__(self):
#         super(L_Intensity, self).__init__()
#         # self.DWT = DWT_2D(wavename='haar')
#
#     def forward(self, image_A, image_B, image_fused):
#         r = args.r
#
#         # 新增：转换为灰度图进行损失计算
#         image_A_gray = rgb_to_gray(image_A)
#         image_B_gray = rgb_to_gray(image_B)
#         image_fused_gray = rgb_to_gray(image_fused)
#
#
#         intensity_joint = torch.max(image_A_gray, image_B_gray)
#         Loss_intensity = F.l1_loss(intensity_joint, image_fused_gray)
#
#         return Loss_intensity
import torch
import torch.nn.functional as F
import torch.nn as nn

class L_Intensity(nn.Module):
    """
    自适应强度损失：
    - 先对 IR / VIS 灰度图做 ‘亮度+纹理’ 判断，生成自适应强度参考图 intensity_joint
    - 再用 L1 损失约束融合图的灰度
    """
    def __init__(self,
                 perc=70,          # 动态阈值分位
                 sigma_th=8,       # 局部无纹理阈值
                 ksize=7,          # 滑窗大小
                 T_dark=30):       # VIS 保护阈值
        super().__init__()
        self.perc = perc
        self.sigma_th = sigma_th
        self.ksize = ksize
        self.T_dark = T_dark
        self.blur = nn.AvgPool2d(kernel_size=ksize, stride=1, padding=ksize//2)  # 取局部均值

    def local_std(self, x):
        """批量版局部 σ 计算（可反向传播）"""
        mean  = self.blur(x)
        mean2 = self.blur(x * x)
        var   = torch.clamp(mean2 - mean * mean, min=0.)
        return torch.sqrt(var + 1e-6)

    def forward(self, image_A, image_B, image_fused):
        """
        image_A : IR   (B,3,H,W) or (B,1,H,W)
        image_B : VIS  (...)
        image_fused :   (...)
        """
        # ------ 灰度转换（保持梯度） ------
        def rgb_to_gray(img):
            if img.shape[1] == 3:
                r, g, b = img[:,0:1], img[:,1:2], img[:,2:3]
                gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                return gray
            return img               # 已经是单通道
        ir_gray  = rgb_to_gray(image_A)
        vis_gray = rgb_to_gray(image_B)
        fused_gray = rgb_to_gray(image_fused)

        # ------ (a) 动态阈值 T per‑batch ------
        # 形状 (B,1,1,1) 方便 broadcast
        q = self.perc / 100.0
        T = torch.quantile(vis_gray.view(vis_gray.shape[0], -1), q, dim=1, keepdim=True)
        T = T.view(-1, 1, 1, 1)

        # ------ (b) 局部 σ ------
        sigma_vis = self.local_std(vis_gray)

        # ------ (c‑1) 过曝掩码 ------
        overexposed = (vis_gray > T) & (sigma_vis < self.sigma_th)

        # ------ (c‑2) VIS 保护 ------
        vis_protect = (ir_gray <= self.T_dark) & (vis_gray > self.T_dark)

        # ------ 三路组合 ------
        use_vis  = vis_protect.float()
        use_ir   = (overexposed & ~vis_protect).float()
        use_max  = 1.0 - use_vis - use_ir          # 剩余像素

        intensity_joint = (
              use_vis * vis_gray
            + use_ir  * ir_gray
            + use_max * torch.maximum(vis_gray, ir_gray)
        )                                           ### NEW intensity_joint

        # ------ L1 强度损失 ------
        Loss_intensity = F.l1_loss(intensity_joint, fused_gray)

        return Loss_intensity

# =================== 颜色损失 =======================
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms.functional as TF
#
# class L_Color(nn.Module):
#     """
#     颜色一致性损失：
#       - 将 RGB 转为 YCbCr
#       - 仅对色度通道 Cb/Cr 施加 L1 或 L2
#     """
#     def __init__(self, loss_type="L1"):
#         super().__init__()
#         assert loss_type in ("L1", "L2")
#         self.loss_fn = F.l1_loss if loss_type == "L1" else F.mse_loss
#
#     @staticmethod
#     def rgb_to_ycbcr(img):            # img: (B,3,H,W), 0~1
#         # torchvision 转换返回 (B,3,H,W) YCbCr，值域 0~1
#         return TF.rgb_to_ycbcr(img)
#
#     def forward(self, fused_rgb, vis_rgb):
#         """
#         fused_rgb : (B,3,H,W) 融合图像
#         vis_rgb   : (B,3,H,W) 可见光原图
#         """
#         fused_ycbcr = self.rgb_to_ycbcr(fused_rgb)
#         vis_ycbcr   = self.rgb_to_ycbcr(vis_rgb)
#
#         # 取色度通道 (Cb,Cr)
#         fused_chroma = fused_ycbcr[:, 1:, ...]    # (B,2,H,W)
#         vis_chroma   = vis_ycbcr[:, 1:, ...]      # (B,2,H,W)
#
#         loss_color = self.loss_fn(fused_chroma, vis_chroma)
#         return loss_color

# =================== Grad 损失（梯度损失） =======================
# class L_Grad(nn.Module):
#     """
#     作用：保持图像的边缘和纹理信息，防止融合图像变得模糊。
#     实现：
#     - 对三幅图像做小波分解。
#     - 用Sobel算子提取每个子带的梯度。
#     - 对红外和可见光的梯度取最大值，与融合图像的梯度做L1损失。
#     - 各子带损失加权求和。
#     """
#     def __init__(self):
#         super(L_Grad, self).__init__()
#         self.sobelconv = Sobelxy()
#         self.DWT = DWT_2D(wavename='haar')
#
#     def forward(self, image_A, image_B, image_fused):
#         r = args.r
#
#         # 新增：转换为灰度图进行损失计算
#         image_A_gray = rgb_to_gray(image_A)
#         image_B_gray = rgb_to_gray(image_B)
#         image_fused_gray = rgb_to_gray(image_fused)
#
#
#         y_grad = self.sobelconv(image_B_gray)
#         ir_grad = self.sobelconv(image_A_gray)
#         fused_grad = self.sobelconv(image_fused_gray)
#         x_grad_joint = torch.max(y_grad, ir_grad)
#         Loss_gradient = F.l1_loss(x_grad_joint, fused_grad)
#
#         return Loss_gradient
# class GaussianBlur(nn.Module):
#     def __init__(self, kernel_size=25, sigma=3.0):
#         super(GaussianBlur, self).__init__()
#         kernel = self.get_gaussian_kernel2d(kernel_size, sigma)
#         self.register_buffer('weight', kernel)
#
#     def get_gaussian_kernel2d(self, kernel_size, sigma):
#         ax = torch.arange(kernel_size) - kernel_size // 2
#         kernel_1d = torch.exp(-0.5 * (ax / sigma)**2)
#         kernel_1d /= kernel_1d.sum()
#         kernel_2d = torch.outer(kernel_1d, kernel_1d)
#         return kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
#
#     def forward(self, x):
#         C = x.shape[1]
#         weight = self.weight.expand(C, 1, -1, -1)
#         return F.conv2d(x, weight, padding=weight.shape[-1] // 2, groups=C)
#
#
# class L_Grad(nn.Module):
#     """
#     Function: Maintain the edge and texture information of the image and prevent the fused image from becoming blurry.
#     Implement:
#     - Texture enhancement of input images via DoG.
#     - Use Sobel operator to extract gradient.
#     - Take the max of IR and VIS gradient, compare with fused gradient via L1 loss.
#     """
#     def __init__(self):
#         super(L_Grad, self).__init__()
#         self.sobelconv = Sobelxy()
#         self.blur1 = GaussianBlur(kernel_size=5, sigma=0.4)
#         self.blur2 = GaussianBlur(kernel_size=5, sigma=1.1)
#         self.lamb = 3  # λ for texture enhancement
#
#     def texture_enhance(self, x):
#         """E(x) = x + λ * (Gσ1(x) - Gσ2(x))"""
#         return x + self.lamb * (self.blur1(x) - self.blur2(x))
#
#     def forward(self, image_A, image_B, image_fused):
#         # Convert to grayscale
#         image_A_gray = rgb_to_gray(image_A)
#         image_B_gray = rgb_to_gray(image_B)
#         image_fused_gray = rgb_to_gray(image_fused)
#
#         # Texture enhancement
#         image_A_enh = self.texture_enhance(image_A_gray)
#         image_B_enh = self.texture_enhance(image_B_gray)
#
#         # Sobel gradients
#         ir_grad = self.sobelconv(image_A_enh)
#         y_grad = self.sobelconv(image_B_enh)
#         fused_grad = self.sobelconv(image_fused_gray)
#
#         # Joint max gradient
#         x_grad_joint = torch.max(y_grad, ir_grad)
#         Loss_gradient = F.l1_loss(x_grad_joint, fused_grad)
#
#         return Loss_gradient
#log

#  log
class GaussianBlur(nn.Module):
    def __init__(self, sigma=3.0, truncate=4.0):
        super(GaussianBlur, self).__init__()
        # 动态计算kernel_size
        radius = int(truncate * sigma + 0.5)
        kernel_size = 2 * radius + 1
        kernel = self.get_gaussian_kernel2d(kernel_size, sigma)
        self.register_buffer('weight', kernel)

    def get_gaussian_kernel2d(self, kernel_size, sigma):
        ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (ax / sigma)**2)
        kernel_1d /= kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        return kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]

    def forward(self, x):
        C = x.shape[1]
        weight = self.weight.expand(C, 1, -1, -1)
        return F.conv2d(x, weight, padding=weight.shape[-1] // 2, groups=C)


class L_Grad(nn.Module):
    """
    Function: Maintain the edge and texture information of the image and prevent the fused image from becoming blurry.
    Implement:
    - Texture enhancement of input images via DoG.
    - Use Sobel operator to extract gradient.
    - Take the max of IR and VIS gradient, compare with fused gradient via L1 loss.
    """
    def __init__(self, lamb=2.0):  # 使用项目中的lamb=7
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()
        # 使用项目中的DOG参数：sigma1=0.3, sigma2=3
        self.blur1 = GaussianBlur(sigma=0.4)  # 对应项目中的g1
        self.blur2 = GaussianBlur(sigma=1.1)  # 对应项目中的g2
        self.lamb = lamb

    def texture_enhance(self, x):
        """E(x) = x + λ * (Gσ1(x) - Gσ2(x))"""
        return x + self.lamb * (self.blur1(x) - self.blur2(x))

    def forward(self, image_A, image_B, image_fused):
        # Convert to grayscale
        image_A_gray = rgb_to_gray(image_A)
        image_B_gray = rgb_to_gray(image_B)
        image_fused_gray = rgb_to_gray(image_fused)

        # Texture enhancement
        image_A_enh = self.texture_enhance(image_A_gray)
        image_B_enh = self.texture_enhance(image_B_gray)

        # Sobel gradients
        ir_grad = self.sobelconv(image_A_enh)
        y_grad = self.sobelconv(image_B_enh)
        fused_grad = self.sobelconv(image_fused_gray)

        # Joint max gradient
        x_grad_joint = torch.max(y_grad, ir_grad)
        Loss_gradient = F.l1_loss(x_grad_joint, fused_grad)

        return Loss_gradient

    #     def __init__(self):
    #         super(L_Grad, self).__init__()
    #         self.sobelconv = Sobelxy()
    #
    #
    #     def forward(self, image_A, image_B, image_fused):
    #         r = args.r
    #
    #         # 新增：转换为灰度图进行损失计算
    #         image_A_gray = rgb_to_gray(image_A)
    #         image_B_gray = rgb_to_gray(image_B)
    #         image_fused_gray = rgb_to_gray(image_fused)
    #
    #
    #         y_grad = self.sobelconv(image_B_gray)
    #         ir_grad = self.sobelconv(image_A_gray)
    #         fused_grad = self.sobelconv(image_fused_gray)
    #         x_grad_joint = torch.max(y_grad, ir_grad)
    #         Loss_gradient = F.l1_loss(x_grad_joint, fused_grad)
    #
    #         return Loss_gradient



def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class grad(nn.Module):
    def __init__(self, channels=1):
        super(grad, self).__init__()
        laplacian_kernel = torch.tensor([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]).float()

        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        self.laplacian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                          kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter.weight.data = laplacian_kernel
        self.laplacian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.laplacian_filter(x) ** 2


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(
        0)  # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()  # window shape: [1,1, 11, 11]
    return window


# 计算 ssim 损失函数
def mssim(img1, img2, window_size=11):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).

    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2

    (_, channel, height, width) = img1.size()

    # 滤波器窗口
    window = create_window(window_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map
    return ret


def mse(img1, img2, window_size=9):
    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2

    (_, channel, height, width) = img1.size()

    img1_f = F.unfold(img1, (window_size, window_size), padding=padd)
    img2_f = F.unfold(img2, (window_size, window_size), padding=padd)

    res = (img1_f - img2_f) ** 2

    res = torch.sum(res, dim=1, keepdim=True) / (window_size ** 2)

    res = F.fold(res, output_size=(256, 256), kernel_size=(1, 1))
    return res


# 方差计算
def std(img, window_size=9):
    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq

    return sigma1


def sum(img, window_size=9):
    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    win1 = torch.ones_like(window)
    res = F.conv2d(img, win1, padding=padd, groups=channel)
    return res


def final_ssim(img_ir, img_vis, img_fuse):
    ssim_ir = mssim(img_ir, img_fuse)
    ssim_vi = mssim(img_vis, img_fuse)

    # std_ir = std(img_ir)
    # std_vi = std(img_vis)
    std_ir = std(img_ir)
    std_vi = std(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    # m = torch.mean(img_ir)
    # w_ir = torch.where(img_ir > m, one, zero)

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one)

    ssim = map1 * ssim_ir + map2 * ssim_vi
    # ssim = ssim * w_ir
    return ssim.mean()


def final_mse(img_ir, img_vis, img_fuse):
    mse_ir = mse(img_ir, img_fuse)
    mse_vi = mse(img_vis, img_fuse)

    std_ir = std(img_ir)
    std_vi = std(img_vis)
    # std_ir = sum(img_ir)
    # std_vi = sum(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    m = torch.mean(img_ir)
    w_vi = torch.where(img_ir <= m, one, zero)

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one)

    res = map1 * mse_ir + map2 * mse_vi
    res = res * w_vi
    return res.mean()


class L_Laplacian(nn.Module):
    def __init__(self):
        super(L_Laplacian, self).__init__()
        # 定义拉普拉斯卷积核
        self.laplacian_kernel = torch.tensor([
            [1, 1, 1],
            [1, -8, 1],  # 中心为-8,周围为1
            [1, 1, 1]
        ]).float()
        
        # 扩展卷积核维度以适应图像通道
        self.laplacian_kernel = self.laplacian_kernel.view(1, 1, 3, 3)
        
        # 创建卷积层
        self.laplacian_conv = nn.Conv2d(
            in_channels=1,  # 输入为灰度图
            out_channels=1, 
            kernel_size=3,
            padding=1,  # 添加padding以保持输出尺寸
            bias=False
        )
        
        # 设置卷积核权重并冻结
        self.laplacian_conv.weight.data = self.laplacian_kernel
        self.laplacian_conv.weight.requires_grad = False

    def rgb_to_gray(self, x):
        # 确保输入是3通道RGB图像
        if x.size(1) == 3:
            # 使用标准权重: 0.299R + 0.587G + 0.114B
            return 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        else:
            # 如果已经是单通道，直接返回
            return x

    def forward(self, image_A, image_B, image_fused):
        # 1. 转换为灰度图
        x_ir_gray = self.rgb_to_gray(image_A)
        x_vis_gray = self.rgb_to_gray(image_B)
        x_fusion_gray = self.rgb_to_gray(image_fused)
        
        # 2. 提取拉普拉斯特征
        x_ir_laplacian = self.laplacian_conv(x_ir_gray)
        x_vis_laplacian = self.laplacian_conv(x_vis_gray)
        x_fusion_laplacian = self.laplacian_conv(x_fusion_gray)
        
        # 3. 计算目标特征(取最大值)
        x_laplacian_joint = torch.max(x_ir_laplacian, x_vis_laplacian)
        
        # 4. 计算L1损失
        loss = F.l1_loss(x_fusion_laplacian, x_laplacian_joint)
        
        return loss

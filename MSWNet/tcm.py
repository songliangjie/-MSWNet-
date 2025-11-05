from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
# from layers import CheckerboardMaskedConv2d
# from models import *
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from einops import rearrange 
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, DropPath
import numpy as np
import math

from deformable_self_attention import DeformableNeighborhoodAttention

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)

def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)
def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')

def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

class Space2Depth(nn.Module):
    """
    ref: https://github.com/huzi96/Coarse2Fine-PyTorch/blob/master/networks.py
    """

    def __init__(self, r=2):
        super().__init__()
        self.r = r

    def forward(self, x):
        r = self.r
        b, c, h, w = x.size()
        out_c = c * (r**2)
        out_h = h // r
        out_w = w // r
        x_view = x.view(b, c, out_h, r, out_w, r)
        x_prime = x_view.permute(0, 3, 5, 1, 2, 4).contiguous().view(b, out_c, out_h, out_w)
        return x_prime

class Depth2Space(nn.Module):
    def __init__(self, r=2):
        super().__init__()
        self.r = r

    def forward(self, x):
        r = self.r
        b, c, h, w = x.size()
        out_c = c // (r**2)
        out_h = h * r
        out_w = w * r
        x_view = x.view(b, r, r, out_c, h, w)
        x_prime = x_view.permute(0, 3, 4, 1, 5, 2).contiguous().view(b, out_c, out_h, out_w)
        return x_prime

def Demultiplexer(x):
    """
    See Supplementary Material: Figure 2.
    This operation can also implemented by slicing.
    """
    x_prime = Space2Depth(r=2)(x)

    _, C, _, _ = x_prime.shape
    anchor_index = tuple(range(C // 4, C * 3 // 4))
    non_anchor_index = tuple(range(0, C // 4)) + tuple(range(C * 3 // 4, C))

    anchor = x_prime[:, anchor_index, :, :]
    non_anchor = x_prime[:, non_anchor_index, :, :]

    return anchor, non_anchor

def Multiplexer(anchor, non_anchor):
    """
    The inverse opperation of Demultiplexer.
    This operation can also implemented by slicing.
    """
    _, C, _, _ = non_anchor.shape
    x_prime = torch.cat((non_anchor[:, : C//2, :, :], anchor, non_anchor[:, C//2:, :, :]), dim=1)
    return Depth2Space(r=2)(x_prime)

class DeformableWindowAttention(nn.Module):
    """Deformable Attention inside Swin window, shape: [B*num_windows, window_size*window_size, C]"""
    def __init__(self, input_dim, head_dim, window_size, num_heads=1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.num_heads = num_heads if num_heads is not None else 1
        self.dna = DeformableNeighborhoodAttention(
            dim=input_dim,
            num_heads=self.num_heads,
            kernel_size=window_size,  # 窗口内做attention
            dilation=1,
            rel_pos_bias=True,
            # 其他参数可根据需要传递
        )

    def forward(self, x):
        # x: [B*num_windows, window_size*window_size, C]
        Bn, N, C = x.shape
        ws = self.window_size
        assert N == ws * ws, f"window_size不对: {ws}*{ws}!={N}"
        x = x.transpose(1, 2).reshape(Bn, C, ws, ws)  # [Bn, C, ws, ws]
        out = self.dna(x)  # [Bn, C, ws, ws]
        # 动态reshape，确保元素数量一致
        Bn_, C_, ws1, ws2 = out.shape
        assert Bn_ == Bn and C_ == C and ws1 * ws2 == N, f"out.shape={out.shape}, Bn={Bn}, C={C}, N={N}"
        out = out.reshape(Bn, C, N).transpose(1, 2)  # [Bn, N, C]
        return out

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None, num_heads=1):
        """ SwinTransformer Block (Deformable Attention version) """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.window_size = window_size
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )
        # 替换为DeformableWindowAttention
        self.msa = DeformableWindowAttention(input_dim, head_dim, window_size, num_heads=num_heads)

    def window_partition(self, x):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        return windows

    def window_reverse(self, windows, H, W):
        # windows: [num_windows*B, window_size*window_size, C]
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x):
        # x: [B, H, W, C]
        shortcut = x
        x = self.ln1(x)
        B, H, W, C = x.shape
        # pad input if needed
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h), mode='reflect')
            H_pad, W_pad = H + pad_h, W + pad_w
        else:
            H_pad, W_pad = H, W
        # cyclic shift for SW-MSA
        if self.type == 'SW':
            shift_size = self.window_size // 2
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        # partition windows
        x_windows = self.window_partition(x)  # [num_windows*B, window_size*window_size, C]
        # window attention
        attn_windows = self.msa(x_windows)  # [num_windows*B, window_size*window_size, C]
        # merge windows
        x = self.window_reverse(attn_windows, H_pad, W_pad)  # [B, H_pad, W_pad, C]
        # reverse cyclic shift
        if self.type == 'SW':
            shift_size = self.window_size // 2
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
        # remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        x = shortcut + self.drop_path(x)
        # FFN
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class SimpleBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(SimpleBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )
    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x



#
# class DenseBlock(nn.Module):
#     """Simple residual block with two 3x3 convolutions.
#
#     Args:
#         in_ch (int): number of input channels
#         out_ch (int): number of output channels
#     """
#
#     def __init__(self, in_ch: int, out_ch: int):
#         super().__init__()
#         self.conv1 = conv3x3(in_ch, out_ch)
#         self.leaky_relu = nn.LeakyReLU(inplace=True)
#         self.conv2 = conv3x3(out_ch*2, out_ch)
#         self.leaky_relu = nn.LeakyReLU(inplace=True)
#         self.conv3 = conv3x3(in_ch*3, out_ch)
#         self.leaky_relu = nn.LeakyReLU(inplace=True)
#
#
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#
#         out = self.conv1(x)
#         out = self.leaky_relu(out)
#         out1 = self.conv2(torch.cat([x, out], 1))
#         out1 = self.leaky_relu(out1)
#         out2 = self.conv3(torch.cat([x, out, out1], 1))
#         out2 = self.leaky_relu(out2)
#
#         out = out + out1 + out2 + identity
#         return out

# class ConvTransBlock(nn.Module):
#     def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', num_heads=1):
#         """ SwinTransformer and Conv Block
#         """
#         super(ConvTransBlock, self).__init__()
#         self.conv_dim = conv_dim
#         self.trans_dim = trans_dim
#         self.head_dim = head_dim
#         self.window_size = window_size
#         self.drop_path = drop_path
#         self.type = type
#         assert self.type in ['W', 'SW']
#         self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type, num_heads=num_heads)
#         self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
#         self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
#
#         # self.conv_block = DenseBlock(self.conv_dim, self.conv_dim)
#         self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)
#
#     def forward(self, x):
#         conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
#         conv_x = self.conv_block(conv_x) + conv_x
#         trans_x = Rearrange('b c h w -> b h w c')(trans_x)
#         trans_x = self.trans_block(trans_x)
#         trans_x = Rearrange('b h w c -> b c h w')(trans_x)
#         res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
#         x = x + res
#         return x
# class ConvTransBlock(nn.Module):
#     def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W'):
#         """ SwinTransformer and Conv Block
#         """
#         super(ConvTransBlock, self).__init__()
#         self.conv_dim = conv_dim
#         self.trans_dim = trans_dim
#         self.head_dim = head_dim
#         self.window_size = window_size
#         self.drop_path = drop_path
#         self.type = type
#         assert self.type in ['W', 'SW']
#         self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type)
#         self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
#         self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
#
#         self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)
#
    # def forward(self, x):
    #     conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
    #     conv_x = self.conv_block(conv_x) + conv_x
    #     trans_x = Rearrange('b c h w -> b h w c')(trans_x)
    #     trans_x = self.trans_block(trans_x)
    #     trans_x = Rearrange('b h w c -> b c h w')(trans_x)
    #     res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
    #     x = x + res
    #     return x

class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', block_type='complex'):
        """ SwinTransformer and Conv Block
        block_type: 'complex' (default) uses Block, 'simple' uses SimpleBlock
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']
        if block_type == 'simple':
            self.trans_block = SimpleBlock(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type)
        else:
            self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type)
        self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)

    def forward(self, x):
            conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
            conv_x = self.conv_block(conv_x) + conv_x
            trans_x = Rearrange('b c h w -> b h w c')(trans_x)
            trans_x = self.trans_block(trans_x)
            trans_x = Rearrange('b h w c -> b c h w')(trans_x)
            res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
            x = x + res
            return x

class SWAtten(AttentionBlock):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, inter_dim=192) -> None:
        if inter_dim is not None:
            super().__init__(N=inter_dim)
            self.non_local_block = SwinBlock(inter_dim, inter_dim, head_dim, window_size, drop_path)
        else:
            super().__init__(N=input_dim)
            self.non_local_block = SwinBlock(input_dim, input_dim, head_dim, window_size, drop_path)
        if inter_dim is not None:
            self.in_conv = conv1x1(input_dim, inter_dim)
            self.out_conv = conv1x1(inter_dim, output_dim)

    def forward(self, x):
        x = self.in_conv(x)
        identity = x
        z = self.non_local_block(x)
        a = self.conv_a(x)
        b = self.conv_b(z)
        out = a * torch.sigmoid(b)
        out += identity
        out = self.out_conv(out)
        return out

class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path) -> None:
        super().__init__()
        self.block_1 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='W')
        self.block_2 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='SW')
        self.window_size = window_size

    def forward(self, x):
        resize = False
        if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            # x = F.pad(x, (padding_col, padding_col+1, padding_row, padding_row+1), mode='reflect')
            x = F.pad(x, (padding_col, padding_col + 1, padding_row, padding_row + 1))
        trans_x = Rearrange('b c h w -> b h w c')(x)
        trans_x = self.block_1(trans_x)
        trans_x =  self.block_2(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        if resize:
            x = F.pad(x, (-padding_col, -padding_col-1, -padding_row, -padding_row-1))
        return trans_x



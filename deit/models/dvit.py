import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from functools import partial
import numpy as np
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.utils.checkpoint as cp

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from .builder import BACKBONE_REGISTRY
from .swin_transformer import WindowAttention, SwinTransformerBlock, PatchMerging


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, *, key=None, value=None, return_attn=False):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C//self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C//self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C//self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        log_attn = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C//self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        if return_attn:
            return out, log_attn
        else:
            return out


class SelfAttnBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 out_dim=None, with_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.with_mlp = with_mlp
        if self.with_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                           act_layer=act_layer, drop=drop, out_features=out_dim)
            if out_dim is not None and dim != out_dim:
                self.reduction = nn.Sequential(norm_layer(dim),
                                               nn.Linear(dim, out_dim, bias=False))
            else:
                self.reduction = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.with_mlp:
            x = self.reduction(x) + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossAttnBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, out_dim=None, with_mlp=True):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_k = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.with_mlp = with_mlp
        if self.with_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                           act_layer=act_layer, drop=drop, out_features=out_dim)
            if out_dim is not None and dim != out_dim:
                self.reduction = nn.Sequential(norm_layer(dim),
                                               nn.Linear(dim, out_dim, bias=False))
            else:
                self.reduction = nn.Identity()
        # self.with_cp = True

    def forward(self, query, key, *, return_attn=False):
        x = query
        if return_attn:
            out, attn = self.attn(self.norm_q(query), key=self.norm_k(key), return_attn=return_attn)
        else:
            out = self.attn(self.norm_q(query), key=self.norm_k(key), return_attn=return_attn)
        x = x + self.drop_path(out)
        if self.with_mlp:
            x = self.reduction(x) + self.drop_path(self.mlp(self.norm2(x)))
        if return_attn:
            return x, attn
        else:
            return x

    # def forward(self, query, key, *, return_attn=False):
    #
    #     def _inner_forward(query, key):
    #         x = query
    #         if return_attn:
    #             out, attn = self.attn(self.norm_q(query), key=self.norm_k(key),
    #                                   return_attn=return_attn)
    #         else:
    #             out = self.attn(self.norm_q(query), key=self.norm_k(key),
    #                             return_attn=return_attn)
    #         x = x + self.drop_path(out)
    #         x = self.reduction(x) + self.drop_path(self.mlp(self.norm2(x)))
    #         if return_attn:
    #             return x, attn
    #         else:
    #             return x
    #
    #     if self.with_cp and query.requires_grad:
    #         out = cp.checkpoint(_inner_forward, query, key)
    #     else:
    #         out = _inner_forward(query, key)
    #
    #     return out

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    windows = rearrange(x, 'b (h wh) (w ww) c -> (b h w) wh ww c', wh=window_size, ww=window_size, b=B, h=H//window_size, w=W//window_size, c=C)
    # assert torch.allclose(new_x, windows)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    C = windows.shape[-1]
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    x = rearrange(windows, '(b h w) wh ww c -> b (h wh) (w ww) c',
                      wh=window_size, ww=window_size, b=B, h=H // window_size,
                      w=W // window_size, c=C)
    # assert torch.allclose(new_x, x)
    return x

def pairwise_jsd(input, target=None):
    if target is None:
        target = input
    input = input.unsqueeze(2)
    target = target.unsqueeze(1)
    pair_result = torch.sum(input.softmax(dim=-1)*(input - target) + target.softmax(dim=-1)*(target - input), dim=-1)
    pair_result = F.normalize(pair_result, p=1, dim=-1)
    return pair_result


class WindowAggregation(WindowAttention):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__(dim=dim, window_size=window_size, num_heads=num_heads,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         attn_drop=attn_drop, proj_drop=proj_drop)
        delattr(self, 'qkv')
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x, attn, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, N, C = x.shape
        v = rearrange(self.v_proj(x), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C//self.num_heads)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinAttnBlock(SwinTransformerBlock):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, with_mlp=True):
        super().__init__(dim=dim,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=shift_size, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                         attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
        self.input_resolution = input_resolution
        self.with_mlp = with_mlp
        if not self.with_mlp:
            delattr(self, 'mlp')
            delattr(self, 'norm2')
        self.attn = WindowAggregation(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x, attn, mask_matrix):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # [B * head, H, W, C]
        attn = rearrange(attn, 'b head (h w) c -> (b head) h w c', h=H, w=W)
        attn = F.pad(attn, (0, 0, pad_l, pad_r, pad_t, pad_b))
        # cyclic shift
        if self.shift_size > 0:
            shifted_attn = torch.roll(attn, shifts=(
            -self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_attn = attn
        # [B * head * nW, ws, ws, C]
        attn_windows = window_partition(shifted_attn,
                                        self.window_size)  # nW*B, window_size, window_size, C
        attn_windows = pairwise_jsd(
            rearrange(attn_windows, 'b wh ww c -> b (wh ww) c', wh=self.window_size, ww=self.window_size))
        attn_windows = rearrange(attn_windows, '(b head nw) w c -> (b nw) head w c',
                                 b=B, head=self.num_heads,
                                 nw=(Hp // self.window_size) * (Wp // self.window_size))

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, attn=attn_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        if self.with_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class AttnStage(nn.Module):

    def __init__(self, input_resolution, dim, num_cluster, num_blocks, num_heads, drop_path,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, out_dim=None,
                 downsample=None, attn_pool_type='conv', merge_type='conv', window_size=3):
        super().__init__()
        self.cluster_token = nn.Parameter(torch.zeros(1, num_cluster, dim))
        cluster_attn_blocks = []
        input_attn_blocks = []
        attn_pools = []
        merge_blocks = []
        assert attn_pool_type in ['conv', 'max']
        assert merge_type in ['conv', 'swin']
        self.window_size = window_size
        self.shift_size = window_size // 2
        for blk_idx in range(num_blocks):
            cluster_attn_blocks.append(
                CrossAttnBlock(
                    dim=dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path[blk_idx],
                    norm_layer=norm_layer,
                    with_mlp=False))
            input_attn_blocks.append(
                CrossAttnBlock(
                    dim=dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path[blk_idx],
                    norm_layer=norm_layer,
                    out_dim=out_dim if blk_idx == num_blocks - 1 else None))
            if attn_pool_type == 'conv':
                attn_pools.append(
                    nn.Conv2d(in_channels=dim, out_channels=dim, groups=dim,
                              kernel_size=3, stride=2, padding=1)
                )
            else:
                attn_pools.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            merge_dim = dim if blk_idx < num_blocks - 1 or out_dim is None else out_dim
            if merge_type == 'conv':
                merge_blocks.append(
                    nn.Conv2d(in_channels=merge_dim, out_channels=merge_dim, groups=merge_dim,
                              kernel_size=window_size, stride=1, padding=window_size // 2))
            else:
                merge_blocks.append(
                    SwinAttnBlock(dim=merge_dim,
                                  input_resolution=input_resolution,
                                  num_heads=num_heads, window_size=window_size,
                                  shift_size=0 if (blk_idx % 2 == 0) else window_size // 2,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=drop_path[blk_idx],
                                  norm_layer=norm_layer,
                                  with_mlp=False))
        self.cluster_attn_blocks = nn.ModuleList(cluster_attn_blocks)
        self.input_attn_blocks = nn.ModuleList(input_attn_blocks)
        self.attn_pools = nn.ModuleList(attn_pools)
        # self.attn_pools = list(range(len(attn_pools)))
        self.merge_blocks = nn.ModuleList(merge_blocks)
        trunc_normal_(self.cluster_token, std=.02)
        self.downsample = downsample

    def forward(self, x):
        # print(f'input: {x.shape}')

        output_token = x
        cluster_token = self.cluster_token.expand(x.size(0), -1, -1)

        B, _, H, W = output_token.shape
        for cluster_attn, input_attn, attn_pool, merge in zip(
                self.cluster_attn_blocks,
                self.input_attn_blocks,
                self.attn_pools,
                self.merge_blocks):
            assert isinstance(cluster_attn, CrossAttnBlock)
            assert isinstance(input_attn, CrossAttnBlock)
            cluster_token = cluster_attn(
                cluster_token,
                torch.cat((cluster_token,
                           rearrange(attn_pool(output_token), 'b c h w -> b (h w) c')), dim=1))
            # cluster_token = cluster_attn( cluster_token, cluster_token,)
            output_token = rearrange(output_token, 'b c h w -> b (h w) c', h=H, w=W)
            output_token, log_attn = input_attn(output_token, cluster_token, return_attn=True)
            if isinstance(merge, SwinAttnBlock):
                Hp = int(np.ceil(H / self.window_size)) * self.window_size
                Wp = int(np.ceil(W / self.window_size)) * self.window_size
                img_mask = torch.zeros((1, Hp, Wp, 1),
                                       device=x.device)  # 1 Hp Wp 1
                h_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                w_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1

                mask_windows = window_partition(img_mask,
                                                self.window_size)  # nW, window_size, window_size, 1
                mask_windows = mask_windows.view(-1,
                                                 self.window_size * self.window_size)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
                    2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                                  float(-100.0)).masked_fill(
                    attn_mask == 0, float(0.0))
                output_token = merge(output_token, log_attn, attn_mask)
            output_token = rearrange(output_token, 'b (h w) c -> b c h w', h=H, w=W)
            if isinstance(merge, nn.Conv2d):
                output_token = merge(output_token)

        # print(f'after cluster: {output_token.shape}')
        if self.downsample is not None:
            output_token = self.downsample(output_token)
        # print(f'downsample: {output_token.shape}')

        return output_token

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, kernel_size=7, stride=4, padding=2, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.img_size = img_size
        self.patches_resolution = (
        int((img_size[1] + 2 * padding[1] - kernel_size[1]) / stride[1] + 1),
        int((img_size[0] + 2 * padding[0] - kernel_size[0]) / stride[0] + 1))

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)

    @property
    def num_patches(self):
        return self.patches_resolution[1] * self.patches_resolution[0]

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x


@BACKBONE_REGISTRY.register()
class TokenVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, in_chans=3,
                 num_classes=1000, base_dim=96, downsample_type='max', merge_type='conv',
                 window_size=3,
                 stage_blocks=(1, 2, 11, 2),
                 cluster_tokens=(64, 32, 16, 8),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.stage_dims = tuple(base_dim * 2 ** i for i in range(len(stage_blocks)))
        self.depth = sum(stage_blocks)
        assert downsample_type in ['max', 'conv', 'avg', 'merge']

        self.patch_embed = PatchEmbed(
            img_size=img_size, in_chans=in_chans,
            embed_dim=base_dim)
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, in_chans=in_chans, embed_dim=base_dim,
        #     kernel_size=4, stride=4, padding=0
        # )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, base_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        stages = []
        for stage_idx, num_blocks in enumerate(stage_blocks):
            H = patches_resolution[1]//2**stage_idx
            W = patches_resolution[0]//2**stage_idx
            stage_dim = self.stage_dims[stage_idx]
            if stage_idx < len(stage_blocks) - 1:
                out_dim = self.stage_dims[stage_idx+1]
                if downsample_type == 'max':
                    downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                elif downsample_type == 'avg':
                    downsample = nn.AvgPool2d(kernel_size=2, stride=2)
                elif downsample_type == 'merge':
                    downsample = nn.Sequential(Rearrange('b c h w -> b (h w) c', h=H, w=W),
                                               PatchMerging(input_resolution=(H, W), dim=stage_dim, norm_layer=norm_layer),
                                               Rearrange('b (h w) c -> b c h w', h=H//2, w=W//2))
                else:
                    downsample = nn.Conv2d(in_channels=stage_dim,
                                           out_channels=out_dim,
                                           groups=stage_dim,
                                           kernel_size=(2, 2), stride=(2, 2))
            else:
                out_dim=None
                downsample=None
            stage_block = AttnStage(input_resolution=(H, W),
                                    dim=stage_dim, num_blocks=num_blocks, num_heads=stage_dim // 96,
                                    num_cluster=cluster_tokens[stage_idx], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, drop_path=dpr[sum(stage_blocks[:stage_idx]):sum(stage_blocks[:stage_idx+1])],
                                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                    norm_layer=norm_layer, out_dim=out_dim if downsample_type not in ['merge', 'conv'] else None,
                                    downsample=downsample,
                                    merge_type=merge_type,
                                    window_size=window_size)
            stages.append(stage_block)
        self.stages = nn.ModuleList(stages)

        self.norm = norm_layer(self.stage_dims[-1])

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(self.stage_dims[-1],
                              num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        self.default_cfg = _cfg()

        if pretrained:
            print(f'Load pretrained weight from {pretrained}')
            checkpoint = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(checkpoint["model"])
        self.default_cfg = _cfg()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        skip_set = {'pos_embed', 'cls_token'}
        for name, param in self.named_parameters():
            if 'cluster_token' in name:
                skip_set.add(name)
        print(f'no weight decay: {skip_set}')
        return skip_set

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim,
                              num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c', h=H, w=W)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        for stage_idx, stage_block in enumerate(self.stages):
            x = stage_block(x)

        x = self.norm(rearrange(x, 'b c h w -> b (h w) c'))
        return F.adaptive_avg_pool1d(x.transpose(1, 2), 1).squeeze(2)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from functools import partial
from einops import rearrange, reduce, repeat

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from .builder import BACKBONE_REGISTRY


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

class Identity(nn.Identity):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input, *args, **kwargs):
        return input

class ConvDown(nn.Module):
    def __init__(self, in_channels, stride):
        super(ConvDown, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                              groups=in_channels, kernel_size=stride + 1,
                              stride=stride, padding=stride // 2)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)

    def forward(self, x, width):
        assert x.ndim in [3, 4]
        if x.ndim == 4:
            cls_token = x[:, :, 0].unsqueeze(2)
            num_heads = x.size(1)
            x = rearrange(x[:, :, 1:], 'b n (h w) c -> b (n c) h w', w=width)
            x = self.norm(self.conv(x))
            x = rearrange(x, 'b (n c) h w -> b n (h w) c', n=num_heads)
            x = torch.cat([cls_token, x], dim=2)
        else:
            cls_token = x[:, 0].unsqueeze(1)
            x = rearrange(x[:, 1:], 'b (h w) c -> b c h w', w=width)
            x = self.norm(self.conv(x))
            x = rearrange(x, 'b c h w -> b (h w) c', w=width)
            x = torch.cat([cls_token, x], dim=1)

        return x

class MaxDown(nn.Module):
    def __init__(self, in_channels, stride):
        super(MaxDown, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.max_pool = nn.MaxPool2d(kernel_size=stride+1, stride=stride, padding=stride//2)

    def forward(self, x, width):
        assert x.ndim in [3, 4]
        if x.ndim == 4:
            cls_token = x[:, :, 0].unsqueeze(2)
            num_heads = x.size(1)
            x = rearrange(x[:, :, 1:], 'b n (h w) c -> b (n c) h w', w=width)
            x = self.max_pool(x)
            x = rearrange(x, 'b (n c) h w -> b n (h w) c', n=num_heads, w=width//self.stride)
            x = torch.cat([cls_token, x], dim=2)
        else:
            cls_token = x[:, 0].unsqueeze(1)
            x = rearrange(x[:, 1:], 'b (h w) c -> b c h w', w=width)
            x = self.max_pool(x)
            x = rearrange(x, 'b c h w -> b (h w) c', w=width//self.stride)
            x = torch.cat([cls_token, x], dim=1)

        return x


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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, width=None, q_pool=None, k_pool=None, v_pool=None):
        q_pool = q_pool if q_pool is not None else Identity()
        k_pool = k_pool if k_pool is not None else Identity()
        v_pool = v_pool if v_pool is not None else Identity()
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [B, head, N, C//head]
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q_pool(q, width)
        if hasattr(q_pool, 'stride'):
            N = 1 + (N-1) // q_pool.stride ** 2
        k = k_pool(k, width)
        v = v_pool(v, width)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, out_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, out_features=out_dim)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(norm_layer(dim),
                                           nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = Identity()

    def forward(self, x, width=None, q_pool=None, k_pool=None, v_pool=None):
        q_pool = q_pool if q_pool is not None else Identity()
        x = q_pool(x, width) + self.drop_path(
            self.attn(self.norm1(x), width, q_pool=q_pool, k_pool=k_pool,
                      v_pool=v_pool))
        x = self.reduction(x) + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, kernel_size=7, stride=4, padding=2, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        num_patches = int((img_size[1] + 2 * padding[1] - kernel_size[1]) / stride[1] + 1) * \
                      int((img_size[0] + 2 * padding[0] - kernel_size[0]) / stride[0] + 1)
        self.img_size = img_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        width = x.size(-1)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x, width


@BACKBONE_REGISTRY.register()
class RecurrentMultiscaleVisionTransformer(nn.Module):

    def __init__(self, img_size=224, in_chans=3,
                 num_classes=1000, base_dim=96, base_stride=8, pool_type='max', stages=(1, 2, 11, 2),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), stem_stride = 4):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.stages = stages
        self.stage_strides = tuple(base_stride // 2 ** i for i in range(len(stages)))
        self.depth = sum(stages)
        assert pool_type in ['max', 'conv']
        if pool_type == 'max':
            pool_op = MaxDown
        else:
            pool_op = ConvDown

        assert stem_stride in [4, 8]
        if stem_stride == 4:
            self.patch_embed = PatchEmbed(
                img_size=img_size, in_chans=in_chans,
                embed_dim=base_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, in_chans=in_chans,
                embed_dim=base_dim, stride=8, kernel_size=8, padding=0)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, base_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, base_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.block = Block(dim=base_dim, num_heads=base_dim // 96,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                      qk_scale=qk_scale, drop=drop_rate,
                      attn_drop=attn_drop_rate,
                      drop_path=0,
                      norm_layer=norm_layer)

        self.pools = nn.ModuleDict()
        for stage_idx, num_blocks in enumerate(stages):
            for blk_idx in range(num_blocks):
                stage_stride = self.stage_strides[stage_idx]
                if stage_idx > 0 and blk_idx == 0:
                    q_pool = pool_op(base_dim, 2)
                else:
                    q_pool = None
                k_pool = pool_op(base_dim, stage_stride) if stage_stride > 1 else None
                v_pool = pool_op(base_dim, stage_stride) if stage_stride > 1 else None
                self.pools[f'{stage_idx}-{blk_idx}-q_pool'] = q_pool
                self.pools[f'{stage_idx}-{blk_idx}-k_pool'] = k_pool
                self.pools[f'{stage_idx}-{blk_idx}-v_pool'] = v_pool

        self.norm = norm_layer(base_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(base_dim,
                              num_classes) if num_classes > 0 else Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
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
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim,
                              num_classes) if num_classes > 0 else Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, W = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1,
                                           -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for stage_idx, recurrence in enumerate(self.stages):
            # print(f'stage_idx: {stage_idx}')
            for blk_idx in range(recurrence):
                # print(f'block_idx: {blk_idx}')
                # print(f'W: {W}')
                if blk_idx == 1 and stage_idx >= 1:
                    W //= 2
                    # print(f'new W: {W}')
                # print(f'input shape: {x.shape}')
                x = self.block(x, W,
                               q_pool=self.pools[f'{stage_idx}-{blk_idx}-q_pool'],
                               k_pool=self.pools[f'{stage_idx}-{blk_idx}-k_pool'],
                               v_pool=self.pools[f'{stage_idx}-{blk_idx}-v_pool'])
                # print(f'output shape: {x.shape}')

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


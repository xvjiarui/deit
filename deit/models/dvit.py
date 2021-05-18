import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from functools import partial
import numpy as np
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
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        if return_attn:
            return out, attn.mean(dim=1)
        else:
            return out


class SelfAttnBlock(nn.Module):

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
            drop_path) if drop_path > 0. else nn.Identity()
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
        x = self.reduction(x) + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossAttnBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, out_dim=None):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_k = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, out_features=out_dim)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(norm_layer(dim),
                                           nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = nn.Identity()

    def forward(self, query, key, *, return_attn=False):
        x = query
        if return_attn:
            out, attn = self.attn(self.norm_q(query), key=self.norm_k(key), return_attn=return_attn)
        else:
            out = self.attn(self.norm_q(query), key=self.norm_k(key), return_attn=return_attn)
        x = x + self.drop_path(out)
        x = self.reduction(x) + self.drop_path(self.mlp(self.norm2(x)))
        if return_attn:
            return x, attn
        else:
            return x


class AttnStage(nn.Module):

    def __init__(self, dim, num_cluster, num_blocks, num_heads, drop_path,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, out_dim=None,
                 downsample=None):
        super().__init__()
        self.cluster_token = nn.Parameter(torch.zeros(1, num_cluster, dim))
        cluster_attn_blocks = []
        input_attn_blocks = []
        for blk_idx in range(num_blocks):
            cluster_attn_blocks.append(
                SelfAttnBlock(
                    dim=dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path[blk_idx],
                    norm_layer=norm_layer))
            input_attn_blocks.append(
                CrossAttnBlock(
                    dim=dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path[blk_idx],
                    norm_layer=norm_layer,
                    out_dim=out_dim if blk_idx == num_blocks - 1 else None))
        self.cluster_attn_blocks = nn.ModuleList(cluster_attn_blocks)
        self.input_attn_blocks = nn.ModuleList(input_attn_blocks)
        trunc_normal_(self.cluster_token, std=.02)
        self.downsample = downsample

    def forward(self, x, x_shape):
        B, _, H, W = x_shape
        input_token = x
        cluster_token = self.cluster_token.expand(x.size(0), -1, -1)

        for cluster_attn, input_attn in zip(self.cluster_attn_blocks,
                                            self.input_attn_blocks):
            assert isinstance(cluster_attn, SelfAttnBlock)
            assert isinstance(input_attn, CrossAttnBlock)
            cluster_token = cluster_attn(cluster_token)
            input_token = input_attn(input_token, cluster_token)

        cls_token = input_token[:, 0].unsqueeze(1)
        output_token = rearrange(input_token[:, 1:], 'b (h w) c -> b c h w', h=H, w=W,
                                 b=B)

        if self.downsample is not None:
            output_token = self.downsample(output_token)

        output_shape = output_token.shape
        output_token = rearrange(output_token, 'b c h w -> b (h w) c')
        output_token = torch.cat((cls_token, output_token), dim=1)

        return output_token, output_shape

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
        return x


@BACKBONE_REGISTRY.register()
class TokenVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, in_chans=3,
                 num_classes=1000, base_dim=96, pool_type='max',
                 stage_blocks=(1, 2, 11, 2),
                 cluster_tokens=(64, 32, 16, 8),
                 strides=(1, 2, 2, 2),
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
        assert pool_type in ['max', 'conv']

        self.patch_embed = PatchEmbed(
            img_size=img_size, in_chans=in_chans,
            embed_dim=base_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, base_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, base_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        cumsum_blocks = np.cumsum(stage_blocks)
        stages = []
        for stage_idx, num_blocks in enumerate(stage_blocks):
            stage_dim = self.stage_dims[stage_idx]
            if stage_idx < len(stage_blocks) - 1:
                out_dim = self.stage_dims[stage_idx+1]
            else:
                out_dim = None
            if stage_idx > 0:
                stage_drop_path = dpr[cumsum_blocks[stage_idx-1]:cumsum_blocks[stage_idx]]
            else:
                stage_drop_path = dpr[:cumsum_blocks[stage_idx]]
            if strides[stage_idx] > 1:
                if pool_type == 'max':
                    downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                else:
                    downsample = nn.Conv2d(in_channels=out_dim if out_dim is not None else stage_dim,
                                           out_channels=out_dim if out_dim is not None else stage_dim,
                                           kernel_size=(2, 2), stride=(2, 2))
            else:
                downsample=None
            stage_block = AttnStage(dim=stage_dim, num_blocks=num_blocks, num_heads=stage_dim // 96,
                                    num_cluster=cluster_tokens[stage_idx], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, drop_path=stage_drop_path,
                                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                    norm_layer=norm_layer, out_dim=out_dim,
                                    downsample=downsample)
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
        trunc_normal_(self.cls_token, std=.02)
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
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim,
                              num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)

        B, C, H, W = x.shape
        x_shape = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c', b=B, h=H, w=W, c=C)
        cls_tokens = self.cls_token.expand(B, -1,
                                           -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for stage_idx, stage_block in enumerate(self.stages):
            x, x_shape = stage_block(x, x_shape)


        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

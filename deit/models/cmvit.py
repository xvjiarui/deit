import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from functools import partial
from einops import rearrange, reduce, repeat
import time
import torch
from pykeops.torch import LazyTensor

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from .builder import BACKBONE_REGISTRY
from addict import Dict
import numpy as np


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

def KMeans(x, K, num_iter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    B, N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:, :K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(B, N, 1, D))  # (B, N, 1, D) samples
    c_j = LazyTensor(c.view(B, 1, K, D))  # (B, 1, K, D) centroids

    # K-means loop:
    # - x  is the (B, N, D) point cloud,
    # - cl is the (B, N,) vector of class labels
    # - c  is the (B, K, D) cloud of cluster centroids
    for i in range(num_iter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (B, N, K) symbolic squared distances
        cl = D_ij.argmin(dim=2).long().view(B, N)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(1, cl[:, :, None].repeat(1, 1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.scatter_add(
            torch.zeros(B, K, device=x.device, dtype=x.dtype), dim=1, index=cl,
            src=torch.ones(B, N, device=x.device, dtype=x.dtype))
        c /= Ncl  # in-place division to compute the average

    new_x = ((x_i - c_j) ** 2).sum(-1).transpose(-2, -1)@x


    if verbose:  # Fancy display -----------------------------------------------
        torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                num_iter, end - start, num_iter, (end - start) / num_iter
            )
        )

    return new_x

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

    def forward(self, x, *, width, **kwargs):
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

    def forward(self, x, *, width, **kwargs):
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

class TokenDown(nn.Module):

    def __init__(self, in_channels, stride):
        super(TokenDown, self).__init__()
        self.stride = stride
        self.in_channels = in_channels

    def forward(self, x, *, width, key, attn=None, as_dict=False):
        assert x.ndim in [3, 4]
        if x.ndim == 4:
            # [B, nh, 1, C]
            cls_token = x[:, :, 0].unsqueeze(2)
            # [B, nh, N_q, C]
            query = x[:, :, 1:]
            # [B, nh, N_q, C]
            B, num_head, N_q, C = query.shape
            # [B, nh, N_k, C]
            N_k = key.size(2)

            # [B, nh, N_q, N_k]
            attn = (query @ key.transpose(-2, -1)) * key.size(-1) ** -0.5
            attn = attn.softmax(dim=-1)

            # [B, nh, N_k-1, C]
            group_out = rearrange(query.transpose(2, 3) @ attn[..., :-1],
                                  'b h c n -> b h n c', h=num_head, b=B, c=C,
                                  n=N_k - 1)
            num_ungroup = N_q//self.stride**2 - N_k + 1
            indices = attn[..., -1].argsort(dim=-1, descending=True)[..., :num_ungroup]
            ungroup_out = torch.gather(query, dim=2, index=indices.unsqueeze(3).expand(-1, -1, -1, C))
            x = torch.cat([cls_token, group_out, ungroup_out], dim=2)
        else:
            assert attn is not None
            # [B, 1, C]
            cls_token = x[:, 0].unsqueeze(1)
            # [B, N_q, C]
            query = x[:, 1:]
            # [B, N_q, C]
            B, N_q, C = query.shape
            # [B, N_k, C]
            N_k = key.size(1)
            # [B, N_q, N_k]
            attn = attn.sum(dim=1)
            # [B, N_k-1, C]
            group_out = rearrange(query.transpose(1, 2) @ attn[..., :-1],
                                  'b c n -> b n c', b=B, c=C,
                                  n=N_k - 1)
            num_ungroup = N_q//self.stride**2 - N_k + 1
            indices = attn[..., -1].argsort(dim=-1, descending=True)[..., :num_ungroup]
            ungroup_out = torch.gather(query, dim=1, index=indices.unsqueeze(2).expand(-1, -1, C))
            x = torch.cat([cls_token, group_out, ungroup_out], dim=1)

        if as_dict:
            return Dict(x=x, attn=attn)
        else:
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
                 attn_drop=0., proj_drop=0., q_pool=None, k_pool=None, v_pool=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.q_pool = q_pool if q_pool is not None else Identity()
        self.k_pool = k_pool if k_pool is not None else Identity()
        self.v_pool = v_pool if v_pool is not None else Identity()

    def forward_without_pool(self, x, key=None, value=None):
        B, N, C = x.shape
        if key is None:
            key = x
        if value is None:
            value = key
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(x), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N)
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B)
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B)

        # [B, nh, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x, width=None, key=None, value=None):
        B, N, C = x.shape
        if key is None:
            key = x
        if value is None:
            value = key
        q = rearrange(self.q_proj(x), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N)
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B)
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B)

        q_result = self.q_pool(q, width=width, key=k, as_dict=True)
        if isinstance(q_result, Dict):
            q = q_result.x
        else:
            q = q_result
        k = self.k_pool(k, width=width)
        v = self.v_pool(v, width=width)
        if hasattr(self.q_pool, 'stride'):
            N = 1 + (N-1) // self.q_pool.stride ** 2
            width //= 2

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return Dict(x=x, width=width, attn=q_result.attn if isinstance(q_result, Dict) else None)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 q_pool=None, k_pool=None, v_pool=None, out_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, q_pool=q_pool,
            k_pool=k_pool, v_pool=v_pool)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, out_features=out_dim)
        self.q_pool = q_pool if q_pool is not None else Identity()
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(norm_layer(dim),
                                           nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = Identity()

    def forward_without_pool(self, x, key_tokens, width=None):
        x = self.q_pool(x, width=width, key=key_tokens) + self.drop_path(
            self.attn.forward_without_pool(self.norm1(x), key=key_tokens))
        x = self.reduction(x) + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x, key_tokens, width=None):
        attn_result = self.attn(self.norm1(x), width=width, key=key_tokens)
        x = self.q_pool(x, width=width, key=key_tokens, attn=attn_result.attn) + self.drop_path(attn_result.x)
        x = self.reduction(x) + self.drop_path(self.mlp(self.norm2(x)))
        return Dict(x=x, width=attn_result.width)

class Stage(nn.Module):

    def __init__(self, dim, num_cluster, num_blocks, num_heads, drop_path, mlp_ratio=4.
                 , qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), out_dim=None, q_pool=None):
        super(Stage, self).__init__()
        self.cluster_token = nn.Parameter(torch.zeros(1, num_cluster, dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_cluster, dim))
        stage_blocks = []
        for blk_idx in range(num_blocks):
            block = Block(dim=dim, num_heads=num_heads,
                          mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                          qk_scale=qk_scale, drop=drop_rate,
                          attn_drop=attn_drop_rate,
                          drop_path=drop_path[blk_idx],
                          norm_layer=norm_layer,
                          q_pool=q_pool if blk_idx == 0 else None,
                          k_pool=None,
                          v_pool=None,
                          out_dim=out_dim if blk_idx == num_blocks - 1 else None)
            stage_blocks.append(block)
        self.blocks = nn.ModuleList(stage_blocks)
        trunc_normal_(self.cluster_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x, width):
        cluster_token = self.cluster_token + self.pos_embed
        cluster_token = cluster_token.expand(x.size(0), -1, -1)
        for blk_idx, blk in enumerate(self.blocks):
            assert isinstance(blk, Block)
            # print(f'blk_idx: {blk_idx}')
            # if blk_idx < len(self.blocks)-1 and isinstance(blk.q_pool, Identity):
            #     cluster_token = blk.forward_without_pool(cluster_token, key_tokens=x)
            blk_result = blk(x, key_tokens=cluster_token, width=width)
            x = blk_result.x
            width = blk_result.width
            # print(f'x.shape: {x.shape}')

        return Dict(x=x, width=width)


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
        return Dict(x=x, width=width)


@BACKBONE_REGISTRY.register()
class ClusterMultiscaleVisionTransformer(nn.Module):

    def __init__(self, img_size=224, in_chans=3,
                 num_classes=1000, base_dim=96, base_stride=8, pool_type='max',
                 stages=(1, 2, 11, 2),
                 cluster_tokens=(256, 128, 32, 8),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.stage_dims = tuple(base_dim * 2 ** i for i in range(len(stages)))
        self.stage_strides = tuple(base_stride // 2 ** i for i in range(len(stages)))
        self.depth = sum(stages)
        assert pool_type in ['max', 'conv', 'token']
        if pool_type == 'max':
            pool_op = MaxDown
        elif pool_type == 'token':
            pool_op = TokenDown
        else:
            pool_op = ConvDown

        self.patch_embed = PatchEmbed(
            img_size=img_size, in_chans=in_chans,
            embed_dim=base_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, base_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, base_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        cumsum_blocks = np.cumsum(stages)
        stage_blocks = []
        for stage_idx, num_blocks in enumerate(stages):
            stage_dim = self.stage_dims[stage_idx]
            if stage_idx < len(stages) - 1:
                out_dim = self.stage_dims[stage_idx+1]
            else:
                out_dim = None
            if stage_idx > 0:
                q_pool = pool_op(stage_dim, 2)
                stage_drop_path = dpr[cumsum_blocks[stage_idx-1]:cumsum_blocks[stage_idx]]
            else:
                q_pool = None
                stage_drop_path = dpr[:cumsum_blocks[stage_idx]]
            stage_block = Stage(dim=stage_dim, num_blocks=num_blocks, num_heads=stage_dim // 96,
                          num_cluster=cluster_tokens[stage_idx],
                          mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                          qk_scale=qk_scale, drop_path=stage_drop_path,
                          drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, norm_layer=norm_layer,
                          out_dim=out_dim, q_pool=q_pool)
            stage_blocks.append(stage_block)
        self.stage_blocks = nn.ModuleList(stage_blocks)

        self.norm = norm_layer(self.stage_dims[-1])

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(self.stage_dims[-1],
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
        embed_result = self.patch_embed(x)
        x = embed_result.x
        width = embed_result.width

        cls_tokens = self.cls_token.expand(B, -1,
                                           -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # for blk in self.blocks:
        #     x = blk(x)
        for stage_idx, stage_block in enumerate(self.stage_blocks):
            # print(f'stage_idx: {stage_idx}')
            stage_result = stage_block(x, width)
            x = stage_result.x
            width = stage_result.width

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


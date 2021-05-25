import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from .builder import BACKBONE_REGISTRY
from .swin_transformer import PatchMerging
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GlobalContext(nn.Module):

    def __init__(self, dim, qk_dim=32, num_heads=8, context_drop=0., proj_drop=0., tau=1.):
        super(GlobalContext, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_dim = qk_dim
        self.tau = tau

        # self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_proj = nn.Linear(dim, qk_dim, bias=True)
        self.k_proj = nn.Linear(dim, qk_dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.context_drop = nn.Dropout(context_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key=None):
        x = query
        if key is None:
            key = x
        batch, q_length, dim = x.shape
        k_length = key.size(1)
        # # [batch, length, dim]
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(key)
        # [batch, num_heads, q_length, qk_dim//num_heads]
        q = q.reshape(batch, q_length, self.num_heads, self.qk_dim//self.num_heads).transpose(1, 2)
        # [batch, num_heads, k_length, qk_dim//num_heads]
        k = k.reshape(batch, k_length, self.num_heads, self.qk_dim//self.num_heads).transpose(1, 2)
        # [batch, num_heads, k_length, dim//num_heads]
        v = v.reshape(batch, k_length, self.num_heads, dim//self.num_heads).transpose(1, 2)
        # [batch, num_heads, k_length, qk_dim//num_heads]
        context_map = F.softmax(k/self.tau, dim=2)
        # [batch, num_heads, qk_dim//num_heads, dim//num_heads]
        context_value = context_map.transpose(-2, -1) @ v
        # [batch, num_heads, q_length, dim//num_heads]
        context_query = q @ context_value
        # [batch, num_heads, q_length, dim//num_heads]
        context_query = context_query.transpose(1, 2).reshape(batch, q_length, dim)
        x = self.proj(context_query)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, qk_dim, num_heads, mlp_ratio=4.,
                 tau=1., drop=0., context_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.context = GlobalContext(dim=dim, qk_dim=qk_dim,
                                     num_heads=num_heads,
                                     context_drop=context_drop, proj_drop=drop,
                                     tau=tau)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.context(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossBlock(nn.Module):

    def __init__(self, dim, qk_dim, num_heads, mlp_ratio=4.,
                 tau=1., drop=0., context_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_k = norm_layer(dim)
        self.context = GlobalContext(dim=dim, qk_dim=qk_dim,
                                     num_heads=num_heads,
                                     context_drop=context_drop, proj_drop=drop,
                                     tau=tau)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, query, key):
        x = query
        x = x + self.drop_path(self.context(self.norm_q(query), self.norm_k(key)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
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

class BasicLayer(nn.Module):

    def __init__(self, input_resolution, dim, qk_dim, num_blocks, num_heads, drop_path,
                 mlp_ratio=4., drop_rate=0.,
                 norm_layer=nn.LayerNorm, downsample=None, attn_pool_type='conv'):
        super().__init__()
        assert attn_pool_type in ['conv', 'max']
        blocks = []
        attn_pools = []
        for blk_idx in range(num_blocks):
            blocks.append(
                CrossBlock(
                    dim=dim,
                    qk_dim=qk_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=drop_path[blk_idx],
                    norm_layer=norm_layer))
            if attn_pool_type == 'conv':
                attn_pools.append(
                    nn.Conv2d(in_channels=dim, out_channels=dim, groups=dim,
                              kernel_size=3, stride=2, padding=1)
                )
            else:
                attn_pools.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.blocks = nn.ModuleList(blocks)
        self.attn_pools = nn.ModuleList(attn_pools)
        self.num_blocks = num_blocks
        self.downsample = downsample

    def forward(self, x):
        # print(f'input: {x.shape}')

        B, _, H, W = x.shape
        flattened = False
        # for block in self.blocks:
        #     x = block(x)
        for i in range(self.num_blocks):
            pooled_x = self.attn_pools[i](x)
            pooled_x = rearrange(pooled_x, 'b c h w -> b (h w) c', h=(H+1)//2, w=(W+1)//2)
            x = rearrange(x, 'b c h w -> b (h w) c', h=H, w=W)
            x = self.blocks[i](x, pooled_x)
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        if isinstance(self.downsample, PatchMerging):
            x = rearrange(x, 'b c h w -> b (h w) c', h=H, w=W)
            flattened = True

        if self.downsample is not None:
            x = self.downsample(x)

        if flattened:
            x = rearrange(x, 'b (h w) c -> b c h w', h=H//2, w=W//2)

        return x


@BACKBONE_REGISTRY.register()
class MultiscaleGlobalContextTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, in_chans=3,
                 num_classes=1000, base_dim=96, downsample_type='max',
                 stage_blocks=(1, 2, 11, 2),
                 mlp_ratio=4.,
                 qk_dim=48,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.stage_dims = tuple(
            base_dim * 2 ** i for i in range(len(stage_blocks)))
        self.depth = sum(stage_blocks)
        assert downsample_type in ['max', 'conv', 'avg', 'merge']

        self.patch_embed = PatchEmbed(
            img_size=img_size, in_chans=in_chans,
            embed_dim=base_dim)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, base_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                self.depth)]  # stochastic depth decay rule
        layers = []
        for stage_idx, num_blocks in enumerate(stage_blocks):
            H = patches_resolution[1] // 2 ** stage_idx
            W = patches_resolution[0] // 2 ** stage_idx
            stage_dim = self.stage_dims[stage_idx]
            if stage_idx < len(stage_blocks) - 1:
                out_dim = self.stage_dims[stage_idx + 1]
                if downsample_type == 'max':
                    downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                elif downsample_type == 'avg':
                    downsample = nn.AvgPool2d(kernel_size=2, stride=2)
                elif downsample_type == 'merge':
                    downsample = PatchMerging(input_resolution=(H, W), dim=stage_dim,
                                     norm_layer=norm_layer)
                else:
                    downsample = nn.Conv2d(in_channels=stage_dim,
                                           out_channels=out_dim,
                                           groups=stage_dim,
                                           kernel_size=(2, 2),
                                           stride=(2, 2))
            else:
                downsample = None
            layer = BasicLayer(input_resolution=(H, W),
                               dim=stage_dim, num_blocks=num_blocks,
                               num_heads=stage_dim // 96,
                               mlp_ratio=mlp_ratio,
                               drop_path=dpr[sum(stage_blocks[:stage_idx]):sum(stage_blocks[:stage_idx + 1])],
                               drop_rate=drop_rate,
                               norm_layer=norm_layer,
                               downsample=downsample,
                               qk_dim=stage_dim)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

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

        for stage_idx, stage_block in enumerate(self.layers):
            x = stage_block(x)

        x = self.norm(rearrange(x, 'b c h w -> b (h w) c'))
        return F.adaptive_avg_pool1d(x.transpose(1, 2), 1).squeeze(2)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# title: DBMNet: Dual-Branch Multi-modal Network with Spatial-Temporal Fusion for Objective Physical Fatigue Assessment
# create: Fangyu Liu, Hao Wang, Ye Li, Fangmin Sun*
# date: July 2024

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict
from torchvision import transforms
import math

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_c=1, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = x.reshape(B, self.num_patches, -1)
        x = self.norm(x)
        return x

# if pos_embed_type == 0
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.encoding = self.generate_encoding(d_model, max_len)

    def generate_encoding(self, d_model, max_len):

        if d_model % 2 != 0:
            d_model += 1

        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)
        return encoding

    def forward(self, x):
        encoding_gpu = self.encoding.to(x.device)
        return x + encoding_gpu[:, :x.size(1), :self.d_model].detach()


class Attention(nn.Module):
    def __init__(self,
                 dim,   # dim of the input token
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_c=1, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None,
                 need_class_token=True, need_mlp_head=True, need_pos_embed=False, pos_embed_type=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.need_class_token = need_class_token
        self.need_mlp_head = need_mlp_head
        self.need_pos_embed = need_pos_embed

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        if self.need_class_token == False:
            self.num_tokens = 0
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed_type = pos_embed_type
        if self.pos_embed_type == 0:
            self.pos_embed = PositionalEncoding(d_model=embed_dim, max_len=1024)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        if self.pos_embed_type != 0:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # example [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # example [B, 196, 768]
        # example [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            if self.need_class_token == True:
                x = torch.cat((cls_token, x), dim=1)  # example [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        if self.need_pos_embed == True:
            if self.pos_embed_type == 0:
                x = self.pos_drop(self.pos_embed(x))
            else:
                x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            if self.need_mlp_head == True:
                return self.pre_logits(x[:, 0])
            else:
                return x
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            if self.need_mlp_head == True:
                x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.Embedding):
        nn.init.trunc_normal_(m.weight, std=.01)


class Spatial_temporal_branches_fusion(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_c=1, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None,
                 need_class_token=True, need_mlp_head=True, need_pos_embed=False, pos_embed_type=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(Spatial_temporal_branches_fusion, self).__init__()
        self.need_class_token = need_class_token
        self.need_mlp_head = need_mlp_head
        self.need_pos_embed = need_pos_embed

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        if self.need_class_token == False:
            self.num_tokens = 0
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed_type = pos_embed_type
        if self.pos_embed_type == 0:
            self.pos_embed = PositionalEncoding(d_model=embed_dim, max_len=1024)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        if self.pos_embed_type != 0:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x, y):
        # Temporal and spatial features are spliced
        out = torch.cat((x, y), dim=2)

        # example [B, C, H, W] -> [B, num_patches, embed_dim]
        out = self.patch_embed(out)  # example [B, 196, 768]

        # example [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(out.shape[0], -1, -1)
        if self.dist_token is None:
            if self.need_class_token == True:
                out = torch.cat((cls_token, out), dim=1)  # example [B, 197, 768]
        else:
            out = torch.cat((cls_token, self.dist_token.expand(out.shape[0], -1, -1), out), dim=1)

        if self.need_pos_embed == True:
            if self.pos_embed_type == 0:
                out = self.pos_drop(self.pos_embed(out))
            else:
                out = self.pos_drop(out + self.pos_embed)
        out = self.blocks(out)
        out = self.norm(out)
        if self.dist_token is None:
            if self.need_mlp_head == True:
                return self.pre_logits(out[:, 0])
            else:
                return out
        else:
            return out[:, 0], out[:, 1]

    def forward(self, x, y):
        out = self.forward_features(x, y)
        if self.head_dist is not None:
            out, out_dist = self.head(out[0]), self.head_dist(out[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return out, out_dist
            else:
                return (out + out_dist) / 2
        else:
            if self.need_mlp_head == True:
                out = self.head(out)
        return out


class Merged_modal_type(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_c=1, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None,
                 need_class_token=True, need_mlp_head=True, need_pos_embed=False, pos_embed_type=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(Merged_modal_type, self).__init__()
        self.need_class_token = need_class_token
        self.need_mlp_head = need_mlp_head
        self.need_pos_embed = need_pos_embed

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        if self.need_class_token == False:
            self.num_tokens = 0
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        self.patch_embed1 = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # Add demographic characteristics
        self.pos_embed_type = pos_embed_type
        if self.pos_embed_type == 0:
            self.pos_embed = PositionalEncoding(d_model=embed_dim, max_len=1024)
        else:
            # By combining multi-modal features, the number of num_patches will be doubled
            self.pos_embed = nn.Parameter(torch.zeros(1, 2 * self.num_patches + self.num_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        if self.pos_embed_type != 0:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.apply(_init_vit_weights)

        self.w_x = torch.nn.Parameter(torch.tensor([0.0]))
        self.w_y = torch.nn.Parameter(torch.tensor([0.0]))
        self.w_z = torch.nn.Parameter(torch.tensor([0.0]))
        self.lstm = nn.LSTM(input_size=750, hidden_size=375, num_layers=2, batch_first=True, bidirectional=True)

    def forward_features(self, x, y, z):
        # example [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # example [B, 196, 768]
        y = self.patch_embed1(y)

        # modal type embedding
        x = x + torch.full(x.shape, self.w_x.item()).to(x.device)
        y = y + torch.full(y.shape, self.w_y.item()).to(y.device)
        z = z + torch.full(z.shape, self.w_z.item()).to(z.device)

        # Splice different modal features together
        out = torch.cat((x, y, z), dim=1)
        # example [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(out.shape[0], -1, -1)
        if self.dist_token is None:
            if self.need_class_token == True:
                out = torch.cat((cls_token, out), dim=1)  # example [B, 197, 768]
        else:
            out = torch.cat((cls_token, self.dist_token.expand(out.shape[0], -1, -1), out), dim=1)

        if self.need_pos_embed == True:
            if self.pos_embed_type == 0:
                out = self.pos_drop(self.pos_embed(out))
            else:
                out = self.pos_drop(out + self.pos_embed)
        out = self.blocks(out)
        out, (_, _) = self.lstm(out)
        out = self.norm(out)
        if self.dist_token is None:
            if self.need_mlp_head == True:
                return self.pre_logits(out[:, 0])
            else:
                return out
        else:
            return out[:, 0], out[:, 1]

    def forward(self, x, y, z):
        out = self.forward_features(x, y, z)
        if self.head_dist is not None:
            out, out_dist = self.head(out[0]), self.head_dist(out[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return out, out_dist
            else:
                return (out + out_dist) / 2
        else:
            if self.need_mlp_head == True:
                out = self.head(out)
        return out


class DBMNet(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self):
        super(DBMNet, self).__init__()

        self.flatten = nn.Flatten()
        self.drop_ratio = 0.0
        self.attn_drop_ratio = 0.0
        self.drop_path_ratio = 0.3

        self.num_heads = 10
        self.depth = 3
        self.depth1 = 1

        # Output Size (B, num_patches=6,embed_dim=125)
        self.model_demographic = VisionTransformer(img_size=(1, 6),
                                                   patch_size=(1, 1),
                                                   embed_dim=125,
                                                   depth=1,
                                                   num_heads=5,
                                                   representation_size=None,
                                                   num_classes=1,
                                                   need_class_token=False,
                                                   need_mlp_head=False,
                                                   need_pos_embed=True,
                                                   pos_embed_type=1,
                                                   drop_ratio=self.drop_ratio,
                                                   attn_drop_ratio=self.attn_drop_ratio,
                                                   drop_path_ratio=self.drop_path_ratio
                                                   )
        # Output Size (B, num_patches=192,embed_dim=375)
        self.sensor_spatial_branch = VisionTransformer(img_size=(24, 6000),
                                                    patch_size=(1, 750),
                                                    embed_dim=750,
                                                    depth=self.depth,
                                                    num_heads=self.num_heads,
                                                    representation_size=None,
                                                    num_classes=1,
                                                    need_class_token=False,
                                                    need_mlp_head=False,
                                                    need_pos_embed=True,
                                                    pos_embed_type=1,
                                                    drop_ratio=self.drop_ratio,
                                                    attn_drop_ratio=self.attn_drop_ratio,
                                                    drop_path_ratio=self.drop_path_ratio
                                                    )
        # Output Size (B, num_patches=192,embed_dim=375)
        self.ecg_spatial_branch = VisionTransformer(img_size=(12, 12000),
                                                 patch_size=(1, 750),
                                                 embed_dim=750,
                                                 depth=self.depth,
                                                 num_heads=self.num_heads,
                                                 representation_size=None,
                                                 num_classes=1,
                                                 need_class_token=False,
                                                 need_mlp_head=False,
                                                 need_pos_embed=True,
                                                 pos_embed_type=1,
                                                 drop_ratio=self.drop_ratio,
                                                 attn_drop_ratio=self.attn_drop_ratio,
                                                 drop_path_ratio=self.drop_path_ratio
                                                 )
        # Sequence independent (channel independent), shared network weights between sequences
        # Output Size (B, num_patches=8,embed_dim=375) * 24条序列
        self.sensor_temporal_branch = VisionTransformer(img_size=(1, 6000),
                                                   patch_size=(1, 750),
                                                   embed_dim=750,
                                                   depth=self.depth,
                                                   num_heads=self.num_heads,
                                                   representation_size=None,
                                                   num_classes=1,
                                                   need_class_token=False,
                                                   need_mlp_head=False,
                                                   need_pos_embed=True,
                                                   pos_embed_type=1,
                                                   drop_ratio=self.drop_ratio,
                                                   attn_drop_ratio=self.attn_drop_ratio,
                                                   drop_path_ratio=self.drop_path_ratio
                                                   )
        # Sequence independent (channel independent), shared network weights between sequences
        # Output Size (B, num_patches=16,embed_dim=375) * 12条序列
        self.ecg_temporal_branch = VisionTransformer(img_size=(1, 12000),
                                                patch_size=(1, 750),
                                                embed_dim=750,
                                                depth=self.depth,
                                                num_heads=self.num_heads,
                                                representation_size=None,
                                                num_classes=1,
                                                need_class_token=False,
                                                need_mlp_head=False,
                                                need_pos_embed=True,
                                                pos_embed_type=1,
                                                drop_ratio=self.drop_ratio,
                                                attn_drop_ratio=self.attn_drop_ratio,
                                                drop_path_ratio=self.drop_path_ratio
                                                )
        # Output Size (B, num_patches=384,embed_dim=375)
        self.sensor_spatial_temporal_merged = Spatial_temporal_branches_fusion(img_size=(384, 750),
                                                          patch_size=(1, 750),
                                                          embed_dim=750,
                                                          depth=self.depth1,
                                                          num_heads=self.num_heads,
                                                          representation_size=None,
                                                          num_classes=1,
                                                          need_class_token=False,
                                                          need_mlp_head=False,
                                                          need_pos_embed=True,
                                                          pos_embed_type=1,
                                                          drop_ratio=self.drop_ratio,
                                                          attn_drop_ratio=self.attn_drop_ratio,
                                                          drop_path_ratio=self.drop_path_ratio
                                                          )
        # Output Size (B, num_patches=384,embed_dim=375)
        self.ecg_spatial_temporal_merged = Spatial_temporal_branches_fusion(img_size=(384, 750),
                                                       patch_size=(1, 750),
                                                       embed_dim=750,
                                                       depth=self.depth1,
                                                       num_heads=self.num_heads,
                                                       representation_size=None,
                                                       num_classes=1,
                                                       need_class_token=False,
                                                       need_mlp_head=False,
                                                       need_pos_embed=True,
                                                       pos_embed_type=1,
                                                       drop_ratio=self.drop_ratio,
                                                       attn_drop_ratio=self.attn_drop_ratio,
                                                       drop_path_ratio=self.drop_path_ratio
                                                       )
        # Output Size (B, num_patches=770,embed_dim=375)
        self.modal_type_merged = Merged_modal_type(img_size=(384, 750),
                                                   patch_size=(1, 750),
                                                   embed_dim=750,
                                                   depth=self.depth1,
                                                   num_heads=self.num_heads,
                                                   representation_size=None,
                                                   num_classes=3,
                                                   need_class_token=True,
                                                   need_mlp_head=True,
                                                   need_pos_embed=True,
                                                   pos_embed_type=1,
                                                   drop_ratio=self.drop_ratio,
                                                   attn_drop_ratio=self.attn_drop_ratio,
                                                   drop_path_ratio=self.drop_path_ratio
                                                   )
        self.sensor_lstm = nn.LSTM(input_size=750, hidden_size=375, num_layers=2, batch_first=True, bidirectional=True)
        self.ecg_lstm = nn.LSTM(input_size=750, hidden_size=375, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x, y, z):
        # Demographic feature extraction (B, 6) -> (B, 1, 1, 6) -> (B, 6, 125) -> (B, 750)
        demographic_feature = self.flatten(self.model_demographic((z.unsqueeze(1)).unsqueeze(1)))
        # Extracting spatial features
        spatial_feature_sensor = self.sensor_spatial_branch(x)
        spatial_feature_ecg = self.ecg_spatial_branch(y)
        # Extraction of temporal features, different temporal features (different channels) of the same modality share model parameters
        # initialize
        temporal_feature_sensor = 0
        temporal_feature_ecg = 0
        for index in range(24):
            extracted_value_x = x[:, :, index, :].unsqueeze(2)
            if index < 12:
                extracted_value_y = y[:, :, index, :].unsqueeze(2)
            if index == 0:
                temporal_feature_sensor = self.sensor_temporal_branch(extracted_value_x)
                temporal_feature_ecg = self.ecg_temporal_branch(extracted_value_y)
            else:
                temporal_feature_sensor = torch.cat((temporal_feature_sensor, self.sensor_temporal_branch(extracted_value_x)), dim=1)
                if index < 12:
                    temporal_feature_ecg = torch.cat((temporal_feature_ecg, self.ecg_temporal_branch(extracted_value_y)), dim=1)
        # feature normalization
        temporal_feature_sensor = nn.functional.normalize(temporal_feature_sensor)
        temporal_feature_ecg = nn.functional.normalize(temporal_feature_ecg)
        spatial_feature_sensor = nn.functional.normalize(spatial_feature_sensor)
        spatial_feature_ecg = nn.functional.normalize(spatial_feature_ecg)
        # Interactive fusion of temporal and spatial features
        sensor_spatial_temporal_merged_feature = self.sensor_spatial_temporal_merged(temporal_feature_sensor.unsqueeze(1), spatial_feature_sensor.unsqueeze(1))
        ecg_spatial_temporal_merged_feature = self.ecg_spatial_temporal_merged(temporal_feature_ecg.unsqueeze(1), spatial_feature_ecg.unsqueeze(1))
        sensor_spatial_temporal_merged_feature, (_, _) = self.sensor_lstm(sensor_spatial_temporal_merged_feature)
        ecg_spatial_temporal_merged_feature, (_, _) = self.ecg_lstm(ecg_spatial_temporal_merged_feature)
        # feature normalization
        demographic_feature = nn.functional.normalize(demographic_feature)
        sensor_spatial_temporal_merged_feature = nn.functional.normalize(sensor_spatial_temporal_merged_feature)
        ecg_spatial_temporal_merged_feature = nn.functional.normalize(ecg_spatial_temporal_merged_feature)
        # Different modal features are interactively fused after modal type embedding
        out = self.modal_type_merged(sensor_spatial_temporal_merged_feature.unsqueeze(1), ecg_spatial_temporal_merged_feature.unsqueeze(1), demographic_feature.unsqueeze(1))
        return out


if __name__ == '__main__':
    # example
    our_DBMNet = DBMNet()

    sensor = torch.randn(4, 1, 24, 6000)
    ecg = torch.randn(4, 1, 12, 12000)
    zz = torch.randn(4, 6)

    output = our_DBMNet(sensor, ecg, zz)

    print(output.shape)
    print(output)

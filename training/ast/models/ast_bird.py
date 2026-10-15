"""
Custom Audio Spectrogram Transformer (AST) for Bird Call Classification
Adapted from the original AST architecture for 30-species bird classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
from typing import Optional, Tuple, List


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.
    ):
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
    """Multi-head self-attention with relative position bias"""
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn


class Block(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if return_attention:
            return x, attn
        return x


class PatchEmbed(nn.Module):
    """2D Spectrogram to Patch Embedding with overlap"""
    def __init__(
        self,
        img_size=(128, 259),  # (freq_bins, time_frames)
        patch_size=16,
        stride=10,
        in_chans=1,
        embed_dim=768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = (patch_size, patch_size)
        self.stride = stride
        
        # Calculate number of patches
        self.grid_size = (
            (img_size[0] - patch_size + stride) // stride,
            (img_size[1] - patch_size + stride) // stride
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # Patch embedding via convolution with stride
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x)  # B, embed_dim, grid_h, grid_w
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        return x


class AST_Bird(nn.Module):
    """Audio Spectrogram Transformer for Bird Call Classification"""
    def __init__(
        self,
        img_size=(128, 259),
        patch_size=16,
        stride=10,
        in_chans=1,
        num_classes=30,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        init_values=None,
        use_mean_pooling=True,
        pretrained=None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.use_mean_pooling = use_mean_pooling
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings (2D sinusoidal)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Time and frequency embeddings (learnable)
        self.time_embed = nn.Parameter(torch.zeros(1, self.patch_embed.grid_size[1], embed_dim))
        self.freq_embed = nn.Parameter(torch.zeros(1, self.patch_embed.grid_size[0], embed_dim))
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Auxiliary outputs
        self.confidence_head = nn.Linear(embed_dim, 1)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        # Initialize patch embedding
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize positional embeddings with 2D sine-cosine
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.patch_embed.grid_size,
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize CLS token
        torch.nn.init.normal_(self.cls_token, std=.02)
        
        # Initialize time and frequency embeddings
        torch.nn.init.normal_(self.time_embed, std=.02)
        torch.nn.init.normal_(self.freq_embed, std=.02)
        
        # Initialize classifier
        torch.nn.init.normal_(self.head.weight, std=0.01)
        if self.head.bias is not None:
            torch.nn.init.constant_(self.head.bias, 0)
            
    def add_time_freq_encoding(self, x):
        """Add time and frequency positional encodings"""
        B, N, C = x.shape
        # Remove CLS token for encoding
        cls_tokens = x[:, :1]
        x = x[:, 1:]
        
        # Reshape to grid
        H, W = self.patch_embed.grid_size
        x = x.reshape(B, H, W, C)
        
        # Add time and frequency embeddings
        x = x + self.time_embed[:, :W].unsqueeze(1)
        x = x + self.freq_embed[:, :H].unsqueeze(2)
        
        # Flatten back
        x = x.reshape(B, H * W, C)
        
        # Re-add CLS token
        x = torch.cat([cls_tokens, x], dim=1)
        return x
        
    def forward_features(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Add time-frequency encoding
        x = self.add_time_freq_encoding(x)
        
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        
        # Global pooling
        if self.use_mean_pooling:
            x = x[:, 1:].mean(dim=1)  # Global average pooling without CLS token
        else:
            x = x[:, 0]  # CLS token
            
        return x
    
    def forward(self, x, return_attention=False):
        # Extract features
        features = self.forward_features(x)
        
        # Classification
        logits = self.head(features)
        confidence = torch.sigmoid(self.confidence_head(features))
        
        if return_attention:
            # Get attention from last layer
            attn_weights = []
            x_tmp = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(x_tmp.shape[0], -1, -1)
            x_tmp = torch.cat((cls_tokens, x_tmp), dim=1)
            x_tmp = x_tmp + self.pos_embed
            x_tmp = self.add_time_freq_encoding(x_tmp)
            x_tmp = self.pos_drop(x_tmp)
            
            for blk in self.blocks[:-1]:
                x_tmp = blk(x_tmp)
            _, attn = self.blocks[-1](x_tmp, return_attention=True)
            
            return logits, confidence, attn
        
        return logits, confidence


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2D sin-cos positional embeddings"""
    grid_h, grid_w = grid_size
    grid = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_h, grid_w])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Generate sin-cos embeddings from grid"""
    
    assert embed_dim % 2 == 0
    H, W = grid.shape[1], grid.shape[2]
    
    # Use half of dimensions for encoding grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sin-cos embeddings"""
    
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def create_ast_bird(num_classes=30, pretrained=None, **kwargs):
    """Create AST model for bird call classification"""
    # Extract parameters that might be in kwargs to avoid duplicates
    default_params = {
        'img_size': (128, 259),
        'patch_size': 16,
        'stride': 10,
        'num_classes': num_classes,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
        'qkv_bias': True,
    }
    
    # Update defaults with any provided kwargs
    default_params.update(kwargs)
    
    model = AST_Bird(**default_params)
    
    if pretrained == 'audioset':
        # Load AudioSet pretrained weights (implementation in pretrained_loader.py)
        from utils.pretrained_loader import load_audioset_weights
        load_audioset_weights(model)
    elif pretrained == 'imagenet':
        # Load ImageNet pretrained weights
        from utils.pretrained_loader import load_imagenet_weights
        load_imagenet_weights(model)
        
    return model

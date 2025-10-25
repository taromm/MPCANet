import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
import math

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class Mlp(nn.Module):
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


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size1, window_size2, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size1 = window_size1
        self.window_size2 = window_size2
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.lambda_ = nn.Parameter(torch.tensor(1.0))

        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, P_thermal=None,mask=None):
        B_, N, C = x.shape
        q = x.reshape(B_, self.num_heads, N, C // self.num_heads)
        kv = self.qkv(y).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        if P_thermal is not None:
            B_, heads, N, _ = attn.shape
            P_thermal = P_thermal.view(B_, -1, N, N)

            if P_thermal.shape[1] != heads:
                P_thermal = P_thermal[:, :heads, :, :]
            elif P_thermal.shape[1] < heads:
                P_thermal = P_thermal.repeat(1, heads // P_thermal.shape[1], 1, 1)

        attn = attn + self.lambda_ * P_thermal

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size1=4, window_size2=6, shift_size=0, up_ratio=1, out_channels=512,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.upSample = nn.UpsamplingBilinear2d(scale_factor=up_ratio)
        self.conv_sem = nn.Sequential(
            conv3x3(1024, out_planes=out_channels, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size1 = window_size1
        self.window_size2 = window_size2
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.lambda_ = nn.Parameter(torch.tensor(1.0))
        self.gamma_sem = nn.Parameter(torch.tensor(1.0))

        if min(self.input_resolution) <= self.window_size2:
            print("window size larger than resolution!!!")
            self.shift_size = 0
            self.window_size2 = min(self.input_resolution)
            self.window_size1 = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size1, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size1=to_2tuple(self.window_size1), window_size2=to_2tuple(self.window_size2), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            print("shift_size > 0!!!!!")
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
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

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, y, semantic=None, P_thermal=None, p_saliency=None, alpha=4, threshold_sal=0.3,use_global=False):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)

        if self.shift_size > 0:

            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if p_saliency is not None:
                shifted_sal = torch.roll(p_saliency,
                                         shifts=(-self.shift_size, -self.shift_size),
                                         dims=(2, 3))
            else:
                shifted_sal = None
        else:
            shifted_x = x
            shifted_y = y
            shifted_sal = p_saliency
        mean_sal = None
        if shifted_sal is not None:
            shifted_sal = F.interpolate(shifted_sal, size=self.input_resolution, mode='bilinear', align_corners=False)
            sal_4d = shifted_sal.permute(0, 2, 3, 1)
            sal_windows = window_partition(sal_4d, self.window_size1)
            sal_windows = sal_windows.view(-1, self.window_size1 * self.window_size1)
            mean_sal = sal_windows.mean(dim=1)

        x_windows = window_partition(shifted_x, self.window_size1)
        x_windows = x_windows.view(-1, self.window_size1 * self.window_size1, C)

        if mean_sal is not None:
            num_windows = mean_sal.shape[0]
            base_size = self.window_size2
            w_small = max(base_size - 2, 4)
            w_large = min(base_size + 2, 8)
            
            window_sizes = torch.where(mean_sal >= threshold_sal, 
                                     torch.tensor(w_small, device=mean_sal.device),
                                     torch.tensor(w_large, device=mean_sal.device))
            
            y_windows_list = []
            for i in range(num_windows):
                window_size = window_sizes[i].item()
                
                windows_per_row = W // self.window_size1
                row = i // windows_per_row
                col = i % windows_per_row
                
                start_h = row * self.window_size1
                end_h = start_h + self.window_size1
                start_w = col * self.window_size1
                end_w = start_w + self.window_size1
                
                window_feature = shifted_y[:, start_h:end_h, start_w:end_w, :]
                
                if window_size != self.window_size1:
                    window_feature = F.interpolate(
                        window_feature.permute(0, 3, 1, 2), 
                        size=(window_size, window_size), 
                        mode='bilinear', 
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                
                window_flat = window_feature.view(-1, window_size * window_size, C)
                y_windows_list.append(window_flat)
            
            y_windows = torch.cat(y_windows_list, dim=0)
        else:
            y_windows = window_partition(shifted_y, self.window_size2)
            y_windows = y_windows.view(-1, self.window_size2 * self.window_size2, C)

        if use_global:
            y_global = shifted_y.mean(dim=(1, 2), keepdim=True)
            num_windows_per_img = y_windows.shape[0] // B
            fused_y_windows = []
            for i in range(y_windows.shape[0]):
                b = i // num_windows_per_img
                patch = y_windows[i]
                global_patch = y_global[b].expand(patch.shape[0], C)
                fused_patch = 0.5 * patch + 0.5 * global_patch
                fused_y_windows.append(fused_patch)
            y_windows = torch.stack(fused_y_windows, dim=0)

        if P_thermal.dim() == 3:
            B, HW, C = P_thermal.shape
            P_thermal = P_thermal.transpose(1, 2).view(B, C, H, W)

        if P_thermal is not None:
            P_thermal = F.interpolate(P_thermal, size=(H, W), mode='bilinear', align_corners=False)
            P_thermal = window_partition(P_thermal.permute(0, 2, 3, 1),self.window_size1)
            P_thermal = P_thermal.view(-1, self.window_size1 * self.window_size1, C)

        if semantic is not None:
            semantic = self.conv_sem(self.upSample(semantic))
            semantic = semantic.flatten(2).transpose(1, 2).view(B, H, W, C)
            if self.shift_size > 0:
                shifted_sem = torch.roll(semantic, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_sem = semantic
            if mean_sal is not None:
                sem_windows_list = []
                for i in range(num_windows):
                    window_size = window_sizes[i].item()
                    
                    windows_per_row = W // self.window_size1
                    row = i // windows_per_row
                    col = i % windows_per_row
                    
                    start_h = row * self.window_size1
                    end_h = start_h + self.window_size1
                    start_w = col * self.window_size1
                    end_w = start_w + self.window_size1
                    
                    sem_window = shifted_sem[:, start_h:end_h, start_w:end_w, :]
                    
                    if window_size != self.window_size1:
                        sem_window = F.interpolate(
                            sem_window.permute(0, 3, 1, 2), 
                            size=(window_size, window_size), 
                            mode='bilinear', 
                            align_corners=False
                        ).permute(0, 2, 3, 1)
                    
                    sem_window_flat = sem_window.view(-1, window_size * window_size, C)
                    sem_windows_list.append(sem_window_flat)
                
                sem_windows = torch.cat(sem_windows_list, dim=0)
            else:
                sem_windows = window_partition(shifted_sem, self.window_size2)
                sem_windows = sem_windows.view(-1, self.window_size2 * self.window_size2, C)
            y_windows = y_windows * (1 + self.gamma_sem * sem_windows)

        attn_windows = self.attn(x_windows, y_windows, P_thermal=P_thermal, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size1, self.window_size1, C)
        shifted_x = window_reverse(attn_windows, self.window_size1, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size1, self.shift_size1), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
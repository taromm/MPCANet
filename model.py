import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional
from torchvision.ops import DeformConv2d
import numpy as np
import copy
from timm.layers import trunc_normal_

from Swin import SwinTransformer
from adaWin import SwinTransformerBlock
from physics import ThermalPhysicsPrior

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SaliencyHead(nn.Module):
    def __init__(self, in_channels, resolution):
        super().__init__()
        self.resolution = resolution
        self.conv_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, fr, ft):
        fused_feat = torch.cat([fr, ft], dim=1)
        saliency_logits = self.conv_head(fused_feat)
        saliency_map = self.sigmoid(saliency_logits)
        return saliency_map

class EdgeHead(nn.Module):
    def __init__(self, in_dim, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_dim, in_dim, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(in_dim)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.edge_pred = nn.Conv2d(in_dim, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        edge_feat = self.relu1(self.bn1(self.conv1(x)))
        edge_feat = self.relu2(self.bn2(self.conv2(edge_feat)))
        
        edge_map = self.edge_pred(edge_feat)
        
        return edge_map

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

class TMSOD(nn.Module):
    def __init__(self):
        super(TMSOD, self).__init__()
        
        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.t_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], in_chans=1)
        
        self.MSA_sem = GMSA_ini(d_model=1024, num_layers=3, resolution=(16, 18))
        self.conv_sem = conv3x3_bn_relu(1024*2, 1024)
        
        self.saliency_head_4 = SaliencyHead(in_channels=1024*2, resolution=(12, 12))
        self.saliency_head_3 = SaliencyHead(in_channels=512*2, resolution=(24, 24))
        self.saliency_head_2 = SaliencyHead(in_channels=256*2, resolution=(48, 48))
        
        self.MSA4_r = SwinTransformerBlock(dim=1024, input_resolution=(12,12), num_heads=2, up_ratio=1, out_channels=1024)
        self.MSA4_t = SwinTransformerBlock(dim=1024, input_resolution=(12, 12), num_heads=2, up_ratio=1,out_channels=1024)
        self.MSA3_r = SwinTransformerBlock(dim=512, input_resolution=(24,24), num_heads=2, up_ratio=2, out_channels=512)
        self.MSA3_t = SwinTransformerBlock(dim=512, input_resolution=(24, 24), num_heads=2, up_ratio=2, out_channels=512)
        self.MSA2_r = SwinTransformerBlock(dim=256, input_resolution=(48,48), num_heads=2, up_ratio=4, out_channels=256)
        self.MSA2_t = SwinTransformerBlock(dim=256, input_resolution=(48, 48), num_heads=2, up_ratio=4, out_channels=256)

        self.align_att4 = get_aligned_feat(inC=1024, outC=1024)
        self.align_att3 = get_aligned_feat(inC=512, outC=512)
        self.align_att2 = get_aligned_feat(inC=256, outC=256)
        self.convAtt4 = conv3x3_bn_relu(1024*2, 1024)
        self.convAtt3 = conv3x3_bn_relu(512*2, 512)
        self.convAtt2 = conv3x3_bn_relu(256*2, 256)

        self.edge_head = EdgeHead(in_dim=256, dilation=1)
        self.edge_preproj_4 = nn.Conv2d(1024 + 1024, 256, kernel_size=1)
        self.edge_preproj_3 = nn.Conv2d(512 + 512, 256, kernel_size=1)
        self.edge_preproj_2 = nn.Conv2d(256 + 256, 256, kernel_size=1)
        self.edge_preproj_1 = nn.Conv2d(128 + 128, 256, kernel_size=1)

        self.shallow_fusion = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.decode4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.decode3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.decode2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.decode1 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.conv64 = conv3x3(64, 1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        
        self.thermal_prior = ThermalPhysicsPrior()
        
        self.thermal_proj_1024 = nn.Conv2d(128, 1024, kernel_size=1)
        self.thermal_gate_proj = nn.Conv2d(128, 1024, kernel_size=1)
        self.thermal_proj_512 = nn.Conv2d(128, 512, kernel_size=1)
        self.thermal_proj_256 = nn.Conv2d(128, 256, kernel_size=1)

        self.apply(init_weights)

        self.pred_saliency = None
        self.debug_thermal = None
        self.P_thermal = self.thermal_prior
        self.attention4 = nn.MultiheadAttention(embed_dim=1024, num_heads=8)
        self.attention3 = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.attention2 = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        
        self.gamma_dec = nn.Parameter(torch.tensor(0.1))

        self.sal_head4 = nn.Conv2d(512, 1, kernel_size=1)
        self.sal_head3 = nn.Conv2d(256, 1, kernel_size=1)
        self.sal_head2 = nn.Conv2d(128, 1, kernel_size=1)
        self.sal_head1 = nn.Conv2d(64, 1, kernel_size=1)

    def _norm_minmax(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)
        
    def forward(self, rgb, t):
        fr = self.rgb_swin(rgb)
        ft = self.t_swin(t)

        P_thermal = self.thermal_prior(t)

        self.debug_thermal = P_thermal

        P_thermal_1024 = self.thermal_proj_1024(P_thermal)
        P_thermal_512 = self.thermal_proj_512(P_thermal)
        P_thermal_256 = self.thermal_proj_256(P_thermal)
        P_thermal_1024 = F.adaptive_avg_pool2d(P_thermal_1024, output_size=(fr[3].shape[2], fr[3].shape[3]))
        P_thermal_512 = F.adaptive_avg_pool2d(P_thermal_512, output_size=(fr[2].shape[2], fr[2].shape[3]))
        P_thermal_256 = F.adaptive_avg_pool2d(P_thermal_256, output_size=(fr[1].shape[2], fr[1].shape[3]))

        p_saliency_4 = self.saliency_head_4(fr[3], ft[3])
        
        p_saliency_3 = self.saliency_head_3(fr[2], ft[2])
        
        p_saliency_2 = self.saliency_head_2(fr[1], ft[1])
        
        self.pred_saliency = p_saliency_4
        
        semantic, p_saliency_init = self.MSA_sem(
            torch.cat((fr[3].flatten(2).transpose(1, 2),
                       ft[3].flatten(2).transpose(1, 2)), dim=1),
            torch.cat((fr[3].flatten(2).transpose(1, 2),
                       ft[3].flatten(2).transpose(1, 2)), dim=1)
        )

        semantic1, semantic2 = torch.split(semantic, fr[3].shape[2] * fr[3].shape[3], dim=1)
        semantic = self.conv_sem(torch.cat((
            semantic1.view(semantic1.shape[0], int(np.sqrt(semantic1.shape[1])), int(np.sqrt(semantic1.shape[1])), -1)
            .permute(0, 3, 1, 2).contiguous(),
            semantic2.view(semantic2.shape[0], int(np.sqrt(semantic2.shape[1])), int(np.sqrt(semantic2.shape[1])), -1)
            .permute(0, 3, 1, 2).contiguous()
        ), dim=1))

        P_sem = F.interpolate(P_thermal, size=semantic.shape[2:], mode='bilinear', align_corners=False)
        P_sem = self.thermal_gate_proj(P_sem)
        semantic = semantic * P_sem

        att_4_r = self.MSA4_r(fr[3].flatten(2).transpose(1, 2),
                              ft[3].flatten(2).transpose(1, 2),
                              semantic=semantic,
                              P_thermal=None,
                              p_saliency=p_saliency_4,
                              alpha=4,
                              threshold_sal=0.3)
        
        att_4_t = self.MSA4_t(ft[3].flatten(2).transpose(1, 2),
                              fr[3].flatten(2).transpose(1, 2),
                              semantic=semantic,
                              P_thermal=P_thermal_1024,
                              p_saliency=p_saliency_4,
                              alpha=4,
                              threshold_sal=0.3)
        
        att_3_r = self.MSA3_r(fr[2].flatten(2).transpose(1, 2),
                              ft[2].flatten(2).transpose(1, 2),
                              semantic=semantic,
                              P_thermal=None,
                              p_saliency=p_saliency_3,
                              alpha=4,
                              threshold_sal=0.3)
        
        att_3_t = self.MSA3_t(ft[2].flatten(2).transpose(1, 2),
                              fr[2].flatten(2).transpose(1, 2),
                              semantic=semantic,
                              P_thermal=P_thermal_512,
                              p_saliency=p_saliency_3,
                              alpha=4,
                              threshold_sal=0.3)
        
        att_2_r = self.MSA2_r(fr[1].flatten(2).transpose(1, 2),
                              ft[1].flatten(2).transpose(1, 2),
                              semantic=semantic,
                              P_thermal=None,
                              p_saliency=p_saliency_2,
                              alpha=4,
                              threshold_sal=0.3)
        
        att_2_t = self.MSA2_t(ft[1].flatten(2).transpose(1, 2),
                              fr[1].flatten(2).transpose(1, 2),
                              semantic=semantic,
                              P_thermal=P_thermal_256,
                              p_saliency=p_saliency_2,
                              alpha=4,
                              threshold_sal=0.3)

        f1 = self.shallow_fusion(torch.cat([fr[0], ft[0]], dim=1))

        r4 = att_4_r.view(att_4_r.shape[0], fr[3].shape[2], fr[3].shape[3], -1).permute(0, 3, 1, 2).contiguous()
        t4 = att_4_t.view(att_4_t.shape[0], fr[3].shape[2], fr[3].shape[3], -1).permute(0, 3, 1, 2).contiguous()

        F_final4, feat_r2t4, feat_t2r4 = self.align_att4(
            fr=r4,
            ft=t4,
            P_thermal=P_thermal_1024,
            saliency_map=p_saliency_4
        )

        self.feat_r2t4 = feat_r2t4
        self.feat_t2r4 = feat_t2r4
        self.r4_orig = r4
        self.t4_orig = t4
        
        r4 = self.convAtt4(torch.cat((r4, F_final4), dim=1))

        r3 = att_3_r.view(att_3_r.shape[0], fr[2].shape[2], fr[2].shape[3], -1).permute(0, 3, 1, 2).contiguous()
        t3 = att_3_t.view(att_3_t.shape[0], fr[2].shape[2], fr[2].shape[3], -1).permute(0, 3, 1, 2).contiguous()

        F_final3, feat_r2t3, feat_t2r3 = self.align_att3(
            fr=r3,
            ft=t3,
            P_thermal=P_thermal_512,
            saliency_map=p_saliency_3
        )

        self.feat_r2t3 = feat_r2t3
        self.feat_t2r3 = feat_t2r3
        self.r3_orig = r3
        self.t3_orig = t3

        r3 = self.convAtt3(torch.cat((r3, F_final3), dim=1))

        r2 = att_2_r.view(att_2_r.shape[0], fr[1].shape[2], fr[1].shape[3], -1).permute(0, 3, 1, 2).contiguous()
        t2 = att_2_t.view(att_2_t.shape[0], fr[1].shape[2], fr[1].shape[3], -1).permute(0, 3, 1, 2).contiguous()

        F_final2, feat_r2t2, feat_t2r2 = self.align_att2(
            fr=r2,
            ft=t2,
            P_thermal=P_thermal_256,
            saliency_map=p_saliency_2
        )

        self.feat_r2t2 = feat_r2t2
        self.feat_t2r2 = feat_t2r2
        self.r2_orig = r2
        self.t2_orig = t2

        r2 = self.convAtt2(torch.cat((r2, F_final2), dim=1))

        U4 = torch.cat([r4, fr[3]], dim=1)
        U4_proj = self.edge_preproj_4(U4)
        E4 = torch.sigmoid(self.edge_head(U4_proj))
        F_tilde_4 = self.decode4(U4_proj) * E4
        p_saliency_norm_4 = self._norm_minmax(p_saliency_4)
        A_sem_4 = 1 + self.gamma_dec * p_saliency_norm_4
        S4 = torch.sigmoid(self.sal_head4(F_tilde_4 * A_sem_4))

        U3 = torch.cat([F.interpolate(F_tilde_4, size=(fr[2].shape[2], fr[2].shape[3]), mode='bilinear', align_corners=False), fr[2]], dim=1)
        U3_proj = self.edge_preproj_3(U3)
        E3 = torch.sigmoid(self.edge_head(U3_proj))
        F_tilde_3 = self.decode3(U3_proj) * E3
        S4_upsampled = F.interpolate(S4, size=(F_tilde_3.shape[2], F_tilde_3.shape[3]), mode='bilinear', align_corners=False)
        A_sem_3 = 1 + self.gamma_dec * self._norm_minmax(S4_upsampled)
        S3 = torch.sigmoid(self.sal_head3(F_tilde_3 * A_sem_3))

        U2 = torch.cat([F.interpolate(F_tilde_3, size=(fr[1].shape[2], fr[1].shape[3]), mode='bilinear', align_corners=False), fr[1]], dim=1)
        U2_proj = self.edge_preproj_2(U2)
        E2 = torch.sigmoid(self.edge_head(U2_proj))
        F_tilde_2 = self.decode2(U2_proj) * E2
        S3_upsampled = F.interpolate(S3, size=(F_tilde_2.shape[2], F_tilde_2.shape[3]), mode='bilinear', align_corners=False)
        A_sem_2 = 1 + self.gamma_dec * self._norm_minmax(S3_upsampled)
        S2 = torch.sigmoid(self.sal_head2(F_tilde_2 * A_sem_2))

        U1 = torch.cat([F.interpolate(F_tilde_2, size=(fr[0].shape[2], fr[0].shape[3]), mode='bilinear', align_corners=False), fr[0]], dim=1)
        U1_proj = self.edge_preproj_1(U1)
        E1 = torch.sigmoid(self.edge_head(U1_proj))
        F_tilde_1 = self.decode1(U1_proj) * E1
        S2_upsampled = F.interpolate(S2, size=(F_tilde_1.shape[2], F_tilde_1.shape[3]), mode='bilinear', align_corners=False)
        A_sem_1 = 1 + self.gamma_dec * self._norm_minmax(S2_upsampled)
        S1 = torch.sigmoid(self.sal_head1(F_tilde_1 * A_sem_1))

        out = self.up4(S1)
        out = self.conv64(out)
        
        self.edge_maps = [E1, E2, E3, E4]
        self.saliency_maps = [S1, S2, S3, S4]
        
        return out, p_saliency_4, P_thermal

    @staticmethod
    def _sigmoid_smooth(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return torch.clamp(x, eps, 1.0 - eps)

    def loss_bce(self, pred_prob: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        pred_prob = self._sigmoid_smooth(pred_prob)
        return F.binary_cross_entropy(pred_prob, gt_mask)

    def loss_dice(self, pred_prob: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pred_prob = self._sigmoid_smooth(pred_prob)
        intersection = (pred_prob * gt_mask).sum(dim=(1, 2, 3))
        union = pred_prob.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + eps) / (union + eps)
        return 1.0 - dice.mean()

    @staticmethod
    def _sobel_kernels(device, dtype):
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
        return kx, ky

    def loss_smooth(self, pred_prob: torch.Tensor, gt_mask: torch.Tensor, alpha: float = 1.0, eps: float = 1e-3) -> torch.Tensor:
        pred_prob = self._sigmoid_smooth(pred_prob)
        device = pred_prob.device
        dtype = pred_prob.dtype
        kx, ky = self._sobel_kernels(device, dtype)
        gx_pred = F.conv2d(pred_prob, kx, padding=1)
        gy_pred = F.conv2d(pred_prob, ky, padding=1)
        gx_gt = F.conv2d(gt_mask, kx, padding=1)
        gy_gt = F.conv2d(gt_mask, ky, padding=1)
        w_x = torch.exp(-alpha * torch.abs(gx_gt))
        w_y = torch.exp(-alpha * torch.abs(gy_gt))
        pen_x = torch.sqrt((w_x * torch.abs(gx_pred)) ** 2 + eps ** 2)
        pen_y = torch.sqrt((w_y * torch.abs(gy_pred)) ** 2 + eps ** 2)
        return (pen_x.mean() + pen_y.mean()) * 0.5

    def moct_loss(
        self,
        pred_logit: torch.Tensor,
        gt_mask: torch.Tensor,
        gt_edge: torch.Tensor,
        lambda_sal: float = 1.0,
        lambda_sm: float = 0.8,
        lambda_edge: float = 0.5,
        lambda_cons: float = 0.1,
    ) -> torch.Tensor:
        
        l_sal = 0.0
        pred_prob = torch.sigmoid(pred_logit)
        l_sal += self.loss_bce(pred_prob, gt_mask) + self.loss_dice(pred_prob, gt_mask)
        
        if hasattr(self, 'saliency_maps') and self.saliency_maps is not None:
            for sal_map in self.saliency_maps:
                sal_resized = F.interpolate(sal_map, size=gt_mask.shape[-2:], 
                                           mode='bilinear', align_corners=False)
                l_sal += self.loss_bce(sal_resized, gt_mask) + self.loss_dice(sal_resized, gt_mask)
        
        l_sm = self.loss_smooth(pred_prob, gt_mask)
        
        l_edge = self.edge_alignment_loss(pred_prob, gt_edge)
        
        l_cons = self.consistency_loss()
        
        return (
            lambda_sal * l_sal
            + lambda_sm * l_sm
            + lambda_edge * l_edge
            + lambda_cons * l_cons
        )

    def load_pre(self, pre_model):
        state_dict = torch.load(pre_model)['model']

        self.rgb_swin.load_state_dict(state_dict, strict=False)
        print(f"RGB SwinTransformer loaded from {pre_model}")

        rgb_weight = state_dict['patch_embed.proj.weight']
        thermal_weight = rgb_weight.mean(dim=1, keepdim=True)
        state_dict['patch_embed.proj.weight'] = thermal_weight
        self.t_swin.load_state_dict(state_dict, strict=False)
        print(f"Thermal SwinTransformer loaded from {pre_model} (converted to 1-channel)")

    def edge_alignment_loss(self, saliency_pred, ground_truth_edge, beta=1.0):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=saliency_pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=saliency_pred.device).view(1, 1, 3, 3)
        
        edge_x = F.conv2d(saliency_pred, sobel_x, padding=1)
        edge_y = F.conv2d(saliency_pred, sobel_y, padding=1)
        pred_edge_from_saliency = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        
        loss_sal_edge = beta * F.l1_loss(pred_edge_from_saliency, ground_truth_edge)
        
        loss_intermediate = 0
        if hasattr(self, 'edge_maps') and self.edge_maps is not None:
            for edge_map in self.edge_maps:
                edge_map_resized = F.interpolate(edge_map, 
                                                size=ground_truth_edge.shape[-2:],
                                                mode='bilinear', 
                                                align_corners=False)
                loss_intermediate += F.l1_loss(edge_map_resized, ground_truth_edge)
        
        return loss_intermediate + loss_sal_edge
    
    def consistency_loss(self, lamda=0.2):
        def _dist_loss(featA, featB):
            featA_norm = F.normalize(featA, p=2, dim=1)
            featB_norm = F.normalize(featB, p=2, dim=1)
            return F.l1_loss(featA_norm, featB_norm)

        loss_layer4 = 0
        if hasattr(self, "feat_r2t4") and hasattr(self, "t4_orig"):
            loss_r2t4 = _dist_loss(self.feat_r2t4, self.t4_orig)
            loss_layer4 += loss_r2t4
        if hasattr(self, "feat_t2r4") and hasattr(self, "r4_orig"):
            loss_t2r4 = _dist_loss(self.feat_t2r4, self.r4_orig)
            loss_layer4 += loss_t2r4

        loss_layer3 = 0
        if hasattr(self, "feat_r2t3") and hasattr(self, "t3_orig"):
            loss_r2t3 = _dist_loss(self.feat_r2t3, self.t3_orig)
            loss_layer3 += loss_r2t3
        if hasattr(self, "feat_t2r3") and hasattr(self, "r3_orig"):
            loss_t2r3 = _dist_loss(self.feat_t2r3, self.r3_orig)
            loss_layer3 += loss_t2r3

        loss_layer2 = 0
        if hasattr(self, "feat_r2t2") and hasattr(self, "t2_orig"):
            loss_r2t2 = _dist_loss(self.feat_r2t2, self.t2_orig)
            loss_layer2 += loss_r2t2
        if hasattr(self, "feat_t2r2") and hasattr(self, "r2_orig"):
            loss_t2r2 = _dist_loss(self.feat_t2r2, self.r2_orig)
            loss_layer2 += loss_t2r2

        loss_consist = loss_layer4 + loss_layer3 + loss_layer2

        return lamda * loss_consist

class GMSA_ini(nn.Module):
    def __init__(self, d_model=256, num_layers=2, resolution=None, decoder_layer=None):
        super(GMSA_ini, self).__init__()
        if decoder_layer is None:
            if resolution is None:
                raise ValueError("Resolution parameter must be provided")
            decoder_layer = GMSA_layer_ini(d_model=d_model, nhead=8, resolution=resolution)
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, fr, ft):
        output = fr
        p_saliency_final = None
        for layer in self.layers:
            output, p_saliency = layer(output, ft)
            p_saliency_final = p_saliency
        return output, p_saliency_final

class GMSA_layer_ini(nn.Module):
    def __init__(self, d_model, nhead, resolution,dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(GMSA_layer_ini, self).__init__()
        self.resolution = resolution
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        self.saliency_head = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 2, d_model // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 4, 1, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, fr, ft, pos=None, query_pos=None):
        fr2 = self.multihead_attn(
            query=self.with_pos_embed(fr, query_pos).transpose(0, 1),
            key=self.with_pos_embed(ft, pos).transpose(0, 1),
            value=ft.transpose(0, 1)
        )[0].transpose(0, 1)

        fr = fr + self.dropout2(fr2)
        fr = self.norm2(fr)

        fr2 = self.linear2(self.dropout(self.activation(self.linear1(fr))))
        fr = fr + self.dropout3(fr2)
        fr = self.norm3(fr)

        B, L, C = fr.shape
        H, W = self.resolution
        assert H * W == L, f"Resolution mismatch: {H}x{W} should equal {L}"

        fr_img = fr.transpose(1, 2).view(B, C, H, W)
        p_sal_map = self.saliency_head(fr_img)
        p_saliency = self.sigmoid(p_sal_map)

        fr_out = fr_img.view(B, C, H * W).transpose(1, 2)

        return fr_out, p_saliency
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class DeformableValueAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        self.out_proj = nn.Linear(dim, dim)
        
        self.lambda_p = nn.Parameter(torch.tensor(1.0))
        self.gamma_val = nn.Parameter(torch.tensor(0.1))
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q_feat, kv_feat, offsets, saliency_map, P_thermal=None):
        B, C, H, W = q_feat.shape
        
        q_feat_flat = q_feat.flatten(2).transpose(1, 2)
        kv_feat_flat = kv_feat.flatten(2).transpose(1, 2)
        
        Q = self.q_proj(q_feat_flat).reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv_feat_flat).reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv_feat_flat).reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        
        if P_thermal is not None:
            P_flat = P_thermal.flatten(2).transpose(1, 2).unsqueeze(1)
            attn = attn + self.lambda_p * P_flat.mean(-1, keepdim=True)
        
        attn_weights = self.softmax(attn)
        
        saliency_flat = saliency_map.flatten(2).transpose(1, 2)
        value_modulation = 1.0 + self.gamma_val * saliency_flat.unsqueeze(1)
        V_modulated = V * value_modulation
        
        V_modulated_spatial = V_modulated.transpose(1, 2).reshape(B, H*W, C).transpose(1, 2).reshape(B, C, H, W)
        V_deformed = self._deformable_sampling(V_modulated_spatial, offsets)
        
        V_deformed_flat = V_deformed.flatten(2).transpose(1, 2).reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        
        out = (attn_weights @ V_deformed_flat).transpose(1, 2).reshape(B, H*W, C)
        out = self.out_proj(out)
        
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out
    
    def _deformable_sampling(self, features, offsets):
        B, C, H, W = features.shape
        
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=features.device),
            torch.linspace(-1, 1, W, device=features.device),
            indexing='ij'
        )
        base_grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        
        offset_x = offsets[:, 0:1, :, :].permute(0, 2, 3, 1) / (W / 2)
        offset_y = offsets[:, 1:2, :, :].permute(0, 2, 3, 1) / (H / 2)
        offset_grid = torch.cat([offset_x, offset_y], dim=-1)
        
        sampling_grid = base_grid + offset_grid
        
        sampled_features = F.grid_sample(
            features, sampling_grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=False
        )
        
        return sampled_features

class get_aligned_feat(nn.Module):
    def __init__(self, inC, outC):
        super(get_aligned_feat, self).__init__()

        self.deformConv1 = defomableConv(inC=inC*2, outC=outC)
        self.deformConv2 = defomableConv(inC=inC, outC=outC)
        self.deformConv3 = defomableConv(inC=inC, outC=outC)

        self.deform_attn_t2r = DeformableValueAttention(dim=inC, num_heads=8)
        self.deform_attn_r2t = DeformableValueAttention(dim=inC, num_heads=8)
        
        self.offset_net = nn.Sequential(
            nn.Conv2d(inC * 2 + 1, inC, 3, padding=1),
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, inC // 2, 3, padding=1),
            nn.BatchNorm2d(inC // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC // 2, 2 * 9, 3, padding=1)
        )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(inC * 2, inC, 3, padding=1),
            nn.BatchNorm2d(inC),
            nn.ReLU(inplace=True)
        )

        self.gamma_sem = nn.Parameter(torch.tensor(0.1))

    def forward(self, fr, ft, P_thermal=None, saliency_map=None):
        
        if saliency_map is not None:
            gate = 1.0 + self.gamma_sem * saliency_map
            ft_gated = ft * gate
            fr_gated = fr * gate
        else:
            ft_gated = ft
            fr_gated = fr
        
        if saliency_map is not None:
            offset_input = torch.cat([fr_gated, ft_gated, saliency_map], dim=1)
        else:
            B, C, H, W = fr_gated.shape
            zero_sal = torch.zeros(B, 1, H, W, device=fr_gated.device)
            offset_input = torch.cat([fr_gated, ft_gated, zero_sal], dim=1)
        
        offsets = self.offset_net(offset_input)
        
        Z_t = self.deform_attn_t2r(
            q_feat=ft_gated,
            kv_feat=fr_gated,
            offsets=offsets,
            saliency_map=saliency_map if saliency_map is not None else torch.ones_like(ft[:, :1]),
            P_thermal=P_thermal
        )
        
        zero_offsets = torch.zeros_like(offsets)
        
        Z_r = self.deform_attn_r2t(
            q_feat=fr_gated,
            kv_feat=ft_gated,
            offsets=zero_offsets,
            saliency_map=saliency_map if saliency_map is not None else torch.ones_like(fr[:, :1]),
            P_thermal=None
        )
        
        F_concat = torch.cat([Z_r, Z_t], dim=1)
        
        F_final = self.fusion_conv(F_concat)

        return F_final, Z_t, Z_r

class defomableConv(nn.Module):
    def __init__(self, inC, outC, kH = 3, kW = 3, num_deformable_groups = 4):
        super(defomableConv, self).__init__()
        self.num_deformable_groups = num_deformable_groups
        self.inC = inC
        self.outC = outC
        self.kH = kH
        self.kW = kW
        self.offset = nn.Conv2d(inC, num_deformable_groups * 2 * kH * kW, kernel_size=(kH, kW), stride=(1, 1), padding=(1, 1), bias=False)
        self.deform = DeformConv2d(inC, outC, (kH, kW), stride=1, padding=1, groups=num_deformable_groups)

    def forward(self, x):
        offset = self.offset(x)
        out = self.deform(x, offset)
        return out

class defomableConv_offset(nn.Module):
    def __init__(self, inC, outC, kH = 3, kW = 3, num_deformable_groups = 2):
        super(defomableConv_offset, self).__init__()
        self.num_deformable_groups = num_deformable_groups
        self.inC = inC
        self.outC = outC
        self.kH = kH
        self.kW = kW
        self.offset = nn.Conv2d(2 * inC + 1, num_deformable_groups * 2 * kH * kW, kernel_size=(kH, kW), stride=(1, 1), padding=(1, 1), bias=False)
        self.deform = DeformConv2d(inC, outC, (kH, kW), stride=1, padding=1, groups=num_deformable_groups)

    def forward(self, feat3, fr, ft, saliency_map, x, thermal_mask=None, beta=None):
        offset_input = torch.cat([fr, ft, saliency_map], dim=1)
        offset = self.offset(offset_input)

        if (thermal_mask is not None) and (beta is not None):
            thermal_mask_expanded = thermal_mask.expand(-1, offset.shape[1], -1, -1)
            offset = offset * (1.0 + beta * thermal_mask_expanded)

        out = self.deform(x, offset)
        return out

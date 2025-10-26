import os
import csv
import torch
import random
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from model import TMSOD

def extract_boundary(gt_mask):

    device = gt_mask.device
    dtype = gt_mask.dtype
    
    # Sobel kernels for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=dtype, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=dtype, device=device).view(1, 1, 3, 3)
    
    # Apply Sobel operators
    edge_x = F.conv2d(gt_mask, sobel_x, padding=1)
    edge_y = F.conv2d(gt_mask, sobel_y, padding=1)
    
    # Compute edge magnitude
    boundary = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
    
    # Normalize to [0, 1]
    boundary = (boundary - boundary.min()) / (boundary.max() - boundary.min() + 1e-6)
    
    return boundary

# Note: The following utility modules can be manually constructed by users:
# - RGBTTMSODDataset: Custom dataset class for RGB-T multi-modal salient object detection
# - SmoothnessLoss, CombinedLoss: Custom loss functions for training
# - EvalThread: Evaluation thread for multi-threaded validation
# - ScaleMonitor: Module for monitoring scale-related metrics during training
# - clip_gradient, adjust_lr, AverageMeter: Training utility functions
# Users can implement these components based on their specific requirements.

train_root = ''
gt_root = ''
thermal_root = ''
save_path = ''
log_path = os.path.join(save_path, 'training_log_1.csv')

trainsize = 384
batchsize = 8
base_lr = 1e-5
weight_decay = 1e-4
grad_clip = 1.0
num_epochs = 200
decay_epoch = 30

os.makedirs(save_path, exist_ok=True)

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} GPU(s)")
    if gpu_count >= 2:
        device = torch.device('cuda:0')
        use_multi_gpu = True
        print("Multi-GPU training mode enabled")
    else:
        device = torch.device('cuda')
        use_multi_gpu = False
else:
    device = torch.device('cpu')
    use_multi_gpu = False

print(f"Using device: {device}, Multi-GPU mode: {use_multi_gpu}")

train_dataset = RGBTTMSODDataset(train_root, gt_root, thermal_root, trainsize, split='train')
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=True)

val_dataset = RGBTTMSODDataset(train_root, gt_root, thermal_root, trainsize, split='val')
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

model = TMSOD().to(device)

if use_multi_gpu:
    print("DataParallel mode enabled, using multi-GPU training")
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
else:
    print("Using single GPU training")

swin_pretrained = "swin_base_patch4_window12_384_22k.pth"

if os.path.exists(swin_pretrained):
    print(f"Loading Swin pretrained weights from {swin_pretrained} ...")
    if use_multi_gpu:
        model.module.load_pre(swin_pretrained)
    else:
        model.load_pre(swin_pretrained)
else:
    print("Warning: Swin pretrained weights not found!")

if use_multi_gpu:
    scale_monitor = ScaleMonitor(model.module, module_path='P_thermal', log_dir=save_path)
else:
    scale_monitor = ScaleMonitor(model, module_path='P_thermal', log_dir=save_path)

if use_multi_gpu:
    model.module.MSA4_r.window_size2 = 4
    model.module.MSA4_t.window_size2 = 4
else:
    model.MSA4_r.window_size2 = 4
    model.MSA4_t.window_size2 = 4

optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

scaler = GradScaler('cuda')

# MOCT loss components tracking
epoch_loss_meter = AverageMeter('TrainLoss')
train_sal_meter = AverageMeter('L_sal')
train_sm_meter = AverageMeter('L_sm')
train_edge_meter = AverageMeter('L_edge')
train_consistency_meter = AverageMeter('L_cons')

print("Start Training...")
best_mae = 1.0

if not os.path.exists(log_path):
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'MAE', 'F-measure', 'S-measure', 'E-measure'])

for epoch in range(num_epochs):
    model.train()
    epoch_loss_meter.reset()
    train_sal_meter.reset()
    train_sm_meter.reset()
    train_edge_meter.reset()
    train_consistency_meter.reset()

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", ncols=120)

    for i, (rgb, gt, thermal) in enumerate(train_loader_tqdm):
        rgb, gt, thermal = rgb.to(device), gt.to(device), thermal.to(device)
        
        gt_edge = extract_boundary(gt)

        optimizer.zero_grad()

        with autocast('cuda'):
            pred, p_saliency, P_thermal = model(rgb, thermal)
            
            if use_multi_gpu:
                total_loss = model.module.moct_loss(
                    pred_logit=pred,
                    gt_mask=gt,
                    gt_edge=gt_edge,
                    lambda_sal=1.0,
                    lambda_sm=0.8,
                    lambda_edge=0.5,
                    lambda_cons=0.1
                )
            else:
                total_loss = model.moct_loss(
                    pred_logit=pred,
                    gt_mask=gt,
                    gt_edge=gt_edge,
                    lambda_sal=1.0,
                    lambda_sm=0.8,
                    lambda_edge=0.5,
                    lambda_cons=0.1
                )

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        clip_gradient(optimizer, grad_clip)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss_meter.update(total_loss.item(), rgb.size(0))

        train_loader_tqdm.set_postfix({
            "BatchLoss": f"{total_loss.item():.4f}",
            "AvgLoss": f"{epoch_loss_meter.avg:.4f}"
        })

    scheduler.step()

    model.eval()
    val_loss_meter = AverageMeter('ValLoss')
    mae_meter = AverageMeter('MAE')
    f_measure_meter = AverageMeter('F-measure')
    s_measure_meter = AverageMeter('S-measure')
    e_measure_meter = AverageMeter('E-measure')

    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", ncols=120)
        for i, (rgb, gt, thermal) in enumerate(val_loader_tqdm):
            rgb, gt, thermal = rgb.to(device), gt.to(device), thermal.to(device)
            
            # Extract boundary for validation loss computation
            gt_edge = extract_boundary(gt)
            
            pred, _, _ = model(rgb, thermal)
            
            # Use complete MOCT loss for validation as well
            if use_multi_gpu:
                val_loss = model.module.moct_loss(
                    pred_logit=pred,
                    gt_mask=gt,
                    gt_edge=gt_edge,
                    lambda_sal=1.0,
                    lambda_sm=0.8,
                    lambda_edge=0.5,
                    lambda_cons=0.1
                )
            else:
                val_loss = model.moct_loss(
                    pred_logit=pred,
                    gt_mask=gt,
                    gt_edge=gt_edge,
                    lambda_sal=1.0,
                    lambda_sm=0.8,
                    lambda_edge=0.5,
                    lambda_cons=0.1
                )
            val_loss_meter.update(val_loss.item(), rgb.size(0))

            pred_prob = torch.sigmoid(pred)
            pred_binary = (pred_prob > 0.5).float()
            mae = torch.abs(pred_prob - gt).mean()
            f_measure = (2 * (pred_binary * gt).sum() + 1e-5) / (pred_binary.sum() + gt.sum() + 1e-5)
            s_measure = 1 - (torch.abs(pred_binary - gt).sum() + 1e-5) / (torch.max(pred_binary, gt).sum() + 1e-5)
            e_measure = (((2 * (pred_prob - pred_prob.mean()) * (gt - gt.mean())) / ((pred_prob - pred_prob.mean())**2 + (gt - gt.mean())**2 + 1e-5) + 1) / 2).mean()

            mae_meter.update(mae.item(), rgb.size(0))
            f_measure_meter.update(f_measure.item(), rgb.size(0))
            s_measure_meter.update(s_measure.item(), rgb.size(0))
            e_measure_meter.update(e_measure.item(), rgb.size(0))
            val_loader_tqdm.set_postfix({
                "ValLoss": f"{val_loss_meter.avg:.4f}",
                "MAE": f"{mae_meter.avg:.4f}"
            })

    if use_multi_gpu:
        torch.save(model.module.state_dict(), os.path.join(save_path, f'best_model_{epoch}.pth'))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f'best_model_{epoch}.pth'))

    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, epoch_loss_meter.avg, val_loss_meter.avg, mae_meter.avg,
                         f_measure_meter.avg, s_measure_meter.avg, e_measure_meter.avg])

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"- Train Loss (MOCT): {epoch_loss_meter.avg:.4f}, "
          f"Val Loss (MOCT): {val_loss_meter.avg:.4f}, "
          f"MAE: {mae_meter.avg:.4f}, "
          f"F-measure: {f_measure_meter.avg:.4f}, "
          f"S-measure: {s_measure_meter.avg:.4f}, "
          f"E-measure: {e_measure_meter.avg:.4f}")

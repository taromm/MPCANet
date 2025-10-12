import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import GaussianBlur
import matplotlib.pyplot as plt
import cv2
import os
from typing import Tuple

class ThermalPhysicsPrior(nn.Module):
    def __init__(self, diffusion_steps=5, alpha=0.1):
        super(ThermalPhysicsPrior, self).__init__()
        self.diffusion_steps = diffusion_steps
        self.alpha = alpha
        self.gaussian_blur = GaussianBlur(kernel_size=5, sigma=2.0)
        self.channel_alignment = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.scale = nn.Parameter(torch.tensor(10.0))

    def compute_temperature_gradient(self, thermal_image):
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=thermal_image.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=thermal_image.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(thermal_image, sobel_x, padding=1)
        grad_y = F.conv2d(thermal_image, sobel_y, padding=1)

        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        return gradient_magnitude

    def compute_heat_diffusion(self, thermal_image):
        heatmap = thermal_image.clone()
        for _ in range(self.diffusion_steps):
            heatmap = self.gaussian_blur(heatmap)
        return heatmap

    def compute_multiscale_log_response(self, thermal_image, scales: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)):
        T = thermal_image.float()
        
        log_responses = []
        
        for sigma in scales:
            log_kernel = self._create_log_kernel(sigma)
            log_kernel = log_kernel.to(T.device)
            
            log_response = F.conv2d(T, log_kernel, padding=log_kernel.size(-1)//2)
            
            log_responses.append(torch.abs(log_response))
        
        stack = torch.stack(log_responses, dim=0)
        multiscale_log = torch.max(stack, dim=0)[0]
        
        multiscale_log = self._normalize01(multiscale_log)
        
        return multiscale_log

    def _create_log_kernel(self, sigma: float, kernel_size: int = None):
        if kernel_size is None:
            kernel_size = max(3, int(2 * round(3 * sigma) + 1))
            if kernel_size % 2 == 0:
                kernel_size += 1
        
        x, y = torch.meshgrid(
            torch.arange(-(kernel_size//2), kernel_size//2 + 1, dtype=torch.float32),
            torch.arange(-(kernel_size//2), kernel_size//2 + 1, dtype=torch.float32),
            indexing='ij'
        )
        
        r_squared = x**2 + y**2
        
        sigma_squared = sigma**2
        gaussian = torch.exp(-r_squared / (2 * sigma_squared))
        log_kernel = -(1 / (torch.pi * sigma_squared**2)) * (1 - r_squared / (2 * sigma_squared)) * gaussian
        
        log_kernel = log_kernel - torch.mean(log_kernel)
        
        log_kernel = log_kernel.unsqueeze(0).unsqueeze(0)
        
        return log_kernel

    def compute_pseudo_time_inertia_map(self, thermal_image, steps: int = 7, sigma0: float = 0.6, growth: float = 1.35, weight_decay: float = 0.85):
        T = thermal_image.float()
        
        accum = torch.zeros_like(T, dtype=torch.float32)
        
        for k in range(steps):
            sigma = sigma0 * (growth ** k)
            
            kernel_size = max(3, int(2 * round(sigma) + 1)) | 1
            gaussian_blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            Tk = gaussian_blur(T)
            
            w = weight_decay ** k
            
            accum += w * torch.abs(T - Tk)
        
        inertia = self._normalize01(accum)
        
        return inertia

    def compute_alpha_energy(self, thermal_image, scales: Tuple[int, ...] = (3, 5, 9, 15), orientations: int = 8):
        T = thermal_image.float()
        
        responses = self._compute_gabor_responses(T, scales, orientations)
        
        stack = torch.stack(responses, dim=0)
        alpha_energy = torch.mean(stack, dim=0)
        
        alpha_energy = self._normalize01(alpha_energy)
        
        return alpha_energy

    def compute_alpha_diff(self, thermal_image, scales: Tuple[int, ...] = (3, 5, 9, 15), orientations: int = 8):
        T = thermal_image.float()
        
        responses = self._compute_gabor_responses(T, scales, orientations)
        
        stack = torch.stack(responses, dim=0)
        alpha_diff = torch.std(stack, dim=0)
        
        alpha_diff = self._normalize01(alpha_diff)
        
        return alpha_diff

    def _compute_gabor_responses(self, thermal_image, scales: Tuple[int, ...], orientations: int):
        responses = []
        
        for s in scales:
            ksize = max(9, int(round(6 * s)) | 1)
            lambd = max(4.0, 2.5 * s)
            gamma = 0.5
            
            for oi in range(orientations):
                theta = (torch.pi * oi) / orientations
                
                kernel = self._create_gabor_kernel(ksize, s, theta, lambd, gamma)
                
                resp = F.conv2d(thermal_image, kernel, padding=ksize//2)
                responses.append(torch.abs(resp))
        
        return responses

    def _create_gabor_kernel(self, ksize: int, sigma: float, theta: float, lambd: float, gamma: float):
        x, y = torch.meshgrid(
            torch.arange(-(ksize//2), ksize//2 + 1, dtype=torch.float32),
            torch.arange(-(ksize//2), ksize//2 + 1, dtype=torch.float32),
            indexing='ij'
        )
        
        theta_tensor = torch.tensor(theta, dtype=torch.float32)
        
        x_theta = x * torch.cos(theta_tensor) + y * torch.sin(theta_tensor)
        y_theta = -x * torch.sin(theta_tensor) + y * torch.cos(theta_tensor)
        
        gb = torch.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * \
             torch.cos(2 * torch.pi * x_theta / lambd)
        
        gb = gb / torch.sum(torch.abs(gb))
        
        gb = gb.unsqueeze(0).unsqueeze(0)
        
        return gb

    def _normalize01(self, x, eps: float = 1e-8):
        mn = torch.min(x)
        mx = torch.max(x)
        
        if mx - mn < eps:
            return torch.zeros_like(x, dtype=torch.float32)
        
        return (x - mn) / (mx - mn + eps)

    def forward(self, thermal_image):
        temp_grad = self.compute_temperature_gradient(thermal_image)
        temp_diffusion = self.compute_heat_diffusion(thermal_image)

        prior_features = torch.cat([temp_grad, temp_diffusion], dim=1)

        prior_features = self.channel_alignment(prior_features) * self.scale

        return prior_features
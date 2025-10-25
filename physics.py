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
        # Calculate total channels: 1 (edge) + 1 (diffusion) + 1 (inertia) + N (emissivity)
        # Emissivity prior has 4 scales * 8 orientations = 32 channels
        emissivity_channels = 4 * 8  # scales * orientations
        total_prior_channels = 1 + 1 + 1 + emissivity_channels  # 35 channels
        
        self.channel_alignment = nn.Sequential(
            nn.Conv2d(total_prior_channels, 64, kernel_size=3, padding=1),
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

    def compute_thermal_inertia_prior(self, thermal_image, steps: int = 7, sigma: float = 1.0):
        """
        Thermal Inertia Prior (P_i) as per Eq. (31-33):
        P_i = ∑_{s=0}^{S-1} |I_t^{(s+1)} - I_t^{(s)}|
        where I_t^{(s+1)} = G_σ * I_t^{(s)}
        """
        T = thermal_image.float()
        
        # Initialize the sequence: I_t^{(0)} = I_t
        I_s = T.clone()
        accum = torch.zeros_like(T, dtype=torch.float32)
        
        # Create Gaussian kernel for diffusion
        kernel_size = max(3, int(2 * round(sigma) + 1)) | 1
        gaussian_blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        
        # Iterative sequence as per formula
        for s in range(steps):
            # I_t^{(s+1)} = G_σ * I_t^{(s)}
            I_s_next = gaussian_blur(I_s)
            
            # |I_t^{(s+1)} - I_t^{(s)}|
            diff = torch.abs(I_s_next - I_s)
            accum += diff
            
            # Update I_s for next iteration
            I_s = I_s_next
        
        inertia = self._normalize01(accum)
        return inertia

    def compute_emissivity_prior(self, thermal_image, scales: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0), orientations: int = 8):
        """
        Emissivity Prior (P_a) as per Eq. (37-39):
        P_a = Concat[F^{-1}(H_b · F(I_t))]_{b=1}^B
        where H_b is log-Gabor filter bank
        """
        T = thermal_image.float()
        B, C, H, W = T.shape
        
        # Convert to frequency domain: F(I_t)
        F_thermal = torch.fft.fft2(T, dim=(-2, -1))
        F_thermal = torch.fft.fftshift(F_thermal, dim=(-2, -1))
        
        responses = []
        
        # Create log-Gabor filter bank
        for scale in scales:
            for oi in range(orientations):
                theta = (torch.pi * oi) / orientations
                
                # Create log-Gabor filter in frequency domain
                H_b = self._create_log_gabor_frequency_filter(H, W, scale, theta, device=T.device)
                
                # Apply filter: H_b · F(I_t)
                filtered = H_b * F_thermal
                
                # Convert back to spatial domain: F^{-1}(H_b · F(I_t))
                filtered = torch.fft.ifftshift(filtered, dim=(-2, -1))
                response = torch.fft.ifft2(filtered, dim=(-2, -1))
                
                # Take magnitude and add to responses
                responses.append(torch.abs(response))
        
        # Concatenate all responses as per formula
        emissivity_prior = torch.cat(responses, dim=1)
        
        return emissivity_prior

    def _create_log_gabor_frequency_filter(self, H: int, W: int, scale: float, theta: float, device: torch.device):
        """
        Create log-Gabor filter in frequency domain
        """
        # Create frequency grids
        u = torch.arange(-W//2, W//2, dtype=torch.float32, device=device)
        v = torch.arange(-H//2, H//2, dtype=torch.float32, device=device)
        U, V = torch.meshgrid(u, v, indexing='ij')
        
        # Convert to polar coordinates
        R = torch.sqrt(U**2 + V**2)
        Phi = torch.atan2(V, U)
        
        # Log-Gabor parameters
        sigma_r = 0.4  # Radial bandwidth
        sigma_phi = 0.3  # Angular bandwidth
        
        # Radial component (log-Gabor)
        radial = torch.exp(-(torch.log(R / scale + 1e-8))**2 / (2 * sigma_r**2))
        
        # Angular component
        angular = torch.exp(-((Phi - theta)**2) / (2 * sigma_phi**2))
        
        # Combine radial and angular components
        filter_freq = radial * angular
        
        # Normalize
        filter_freq = filter_freq / (torch.max(filter_freq) + 1e-8)
        
        # Add channel dimension
        filter_freq = filter_freq.unsqueeze(0).unsqueeze(0)
        
        return filter_freq

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
        # Thermal Edge Prior (P_e): Sobel gradient + multi-scale LoG
        temp_grad = self.compute_temperature_gradient(thermal_image)
        multiscale_log = self.compute_multiscale_log_response(thermal_image)
        
        # Combine gradient and LoG responses as per Eq. (19-21)
        # P_e = Norm(√((∂_x I_t)² + (∂_y I_t)²)) + ∑_σ w_σ · Norm(|∇² G_σ * I_t|)
        temp_grad_norm = self._normalize01(temp_grad)
        multiscale_log_norm = self._normalize01(multiscale_log)
        thermal_edge_prior = temp_grad_norm + multiscale_log_norm
        
        # Thermal Diffusion Prior (P_d): Iterative Gaussian convolution
        temp_diffusion = self.compute_heat_diffusion(thermal_image)
        
        # Thermal Inertia Prior (P_i): Pseudo-temporal analysis
        temp_inertia = self.compute_thermal_inertia_prior(thermal_image)
        
        # Emissivity Prior (P_a): Log-Gabor filter bank in frequency domain
        temp_emissivity = self.compute_emissivity_prior(thermal_image)

        # Prior Fusion: Concat all 4 physical priors as per Eq. (43-45)
        # P^l = φ(Concat[P_e, P_d, P_i, P_a])
        prior_features = torch.cat([thermal_edge_prior, temp_diffusion, temp_inertia, temp_emissivity], dim=1)

        # Channel alignment with 1x1 convolution
        prior_features = self.channel_alignment(prior_features) * self.scale

        return prior_features
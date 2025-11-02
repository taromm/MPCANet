import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from typing import Tuple

class ThermalPhysicsPrior(nn.Module):
    def __init__(self, diffusion_steps=5):
        super(ThermalPhysicsPrior, self).__init__()
        self.diffusion_steps = diffusion_steps
        self.gaussian_blur = GaussianBlur(kernel_size=5, sigma=2.0)
        
        self.log_scales = (1.0, 2.0, 4.0, 8.0)
        self.w_sigma = nn.Parameter(torch.ones(len(self.log_scales)))
        
        emissivity_channels = 4 * 8
        total_prior_channels = 1 + 1 + 1 + emissivity_channels
        
        self.channel_alignment = nn.Conv2d(total_prior_channels, 128, kernel_size=1)

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
        heatmap = self._normalize01(heatmap)
        return heatmap

    def compute_multiscale_log_response(self, thermal_image, scales: Tuple[float, ...] = None, return_per_scale: bool = False):
        T = thermal_image.float()
        
        if scales is None:
            scales = self.log_scales
        
        log_responses = []
        
        for sigma in scales:
            log_kernel = self._create_log_kernel(sigma)
            log_kernel = log_kernel.to(T.device)
            
            log_response = F.conv2d(T, log_kernel, padding=log_kernel.size(-1)//2)
            
            log_response_normalized = self._normalize01(torch.abs(log_response))
            log_responses.append(log_response_normalized)
        
        if return_per_scale:
            return log_responses
        else:
            stack = torch.stack(log_responses, dim=0)
            multiscale_log = torch.max(stack, dim=0)[0]
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
        T = thermal_image.float()
        
        I_s = T.clone()
        accum = torch.zeros_like(T, dtype=torch.float32)
        
        kernel_size = max(3, int(2 * round(sigma) + 1)) | 1
        gaussian_blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        
        for s in range(steps):
            I_s_next = gaussian_blur(I_s)
            diff = torch.abs(I_s_next - I_s)
            accum += diff
            I_s = I_s_next
        
        return accum

    def compute_emissivity_prior(self, thermal_image, scales: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0), orientations: int = 8):
        T = thermal_image.float()
        B, C, H, W = T.shape
        
        F_thermal = torch.fft.fft2(T, dim=(-2, -1))
        F_thermal = torch.fft.fftshift(F_thermal, dim=(-2, -1))
        
        responses = []
        
        for scale in scales:
            for oi in range(orientations):
                theta = (torch.pi * oi) / orientations
                
                H_b = self._create_log_gabor_frequency_filter(H, W, scale, theta, device=T.device)
                
                filtered = H_b * F_thermal
                
                filtered = torch.fft.ifftshift(filtered, dim=(-2, -1))
                response = torch.fft.ifft2(filtered, dim=(-2, -1))
                
                responses.append(response)
        
        emissivity_prior = torch.cat(responses, dim=1)
        
        return emissivity_prior

    def _create_log_gabor_frequency_filter(self, H: int, W: int, scale: float, theta: float, device: torch.device):
        u = torch.arange(-W//2, W//2, dtype=torch.float32, device=device)
        v = torch.arange(-H//2, H//2, dtype=torch.float32, device=device)
        U, V = torch.meshgrid(u, v, indexing='ij')
        
        R = torch.sqrt(U**2 + V**2)
        Phi = torch.atan2(V, U)
        
        sigma_r = 0.4
        sigma_phi = 0.3
        
        radial = torch.exp(-(torch.log(R / scale + 1e-8))**2 / (2 * sigma_r**2))
        
        angular = torch.exp(-((Phi - theta)**2) / (2 * sigma_phi**2))
        
        filter_freq = radial * angular
        
        filter_freq = filter_freq / (torch.max(filter_freq) + 1e-8)
        
        filter_freq = filter_freq.unsqueeze(0).unsqueeze(0)
        
        return filter_freq

    def _normalize01(self, x, eps: float = 1e-8):
        mn = torch.min(x)
        mx = torch.max(x)
        
        if mx - mn < eps:
            return torch.zeros_like(x, dtype=torch.float32)
        
        return (x - mn) / (mx - mn + eps)

    def forward(self, thermal_image):
        temp_grad = self.compute_temperature_gradient(thermal_image)
        temp_grad_norm = self._normalize01(temp_grad)
        
        log_responses_per_scale = self.compute_multiscale_log_response(
            thermal_image, 
            scales=self.log_scales, 
            return_per_scale=True
        )
        
        weighted_log_sum = None
        for i, log_response in enumerate(log_responses_per_scale):
            weighted_response = self.w_sigma[i] * log_response
            if weighted_log_sum is None:
                weighted_log_sum = weighted_response
            else:
                weighted_log_sum = weighted_log_sum + weighted_response
        
        thermal_edge_prior = temp_grad_norm + weighted_log_sum
        
        temp_diffusion = self.compute_heat_diffusion(thermal_image)
        
        temp_inertia = self.compute_thermal_inertia_prior(thermal_image)
        
        temp_emissivity = self.compute_emissivity_prior(thermal_image)

        prior_features = torch.cat([thermal_edge_prior, temp_diffusion, temp_inertia, temp_emissivity], dim=1)

        prior_features = self.channel_alignment(prior_features)

        return prior_features
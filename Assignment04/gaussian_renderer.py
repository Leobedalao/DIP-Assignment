import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import cv2


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width
        
        # Pre-compute pixel coordinates grid
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # Shape: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))


    def compute_projection(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]
        
        # 1. Transform points to camera space
        cam_points = means3D @ R.T + t.unsqueeze(0) # (N, 3)
        
        # 2. Get raw depths before projection for proper sorting and clipping
        depths = cam_points[:, 2]  # (N, )
        safe_z = torch.where(depths > 1e-4, depths, torch.ones_like(depths))
        
        # 3. Project to screen space using camera intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        means2D = torch.stack([
            fx * cam_points[:, 0] / safe_z + cx,
            fy * cam_points[:, 1] / safe_z + cy,
        ], dim=-1)  # (N, 2)
        
        # 4. Transform covariance to camera space and then to 2D
        # Compute Jacobian of perspective projection
        J_proj = torch.zeros((N, 2, 3), device=means3D.device)
        x, y, z = cam_points[:, 0], cam_points[:, 1], safe_z
        J_proj[:, 0, 0] = fx / z
        J_proj[:, 0, 2] = -fx * x / (z * z)
        J_proj[:, 1, 1] = fy / z
        J_proj[:, 1, 2] = -fy * y / (z * z)
        
        # Transform covariance to camera space
        covs_cam = R.unsqueeze(0) @ covs3d @ R.T.unsqueeze(0)  # (N, 3, 3)
        
        # Project to 2D
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2, 1)))  # (N, 2, 2)
        
        return means2D, covs2D, depths

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,    # (N, 2)
        covs2D: torch.Tensor,     # (N, 2, 2)
        pixels: torch.Tensor      # (H, W, 2)
    ) -> torch.Tensor:           # (N, H, W)
        N = means2D.shape[0]
        H, W = pixels.shape[:2]
        
        # Compute offset from mean (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)
        
        # Add a small screen-space footprint so projected Gaussians remain
        # positive definite and differentiable after perspective projection.
        eps = 1e-4
        min_footprint = 0.3
        covs2D = covs2D + min_footprint * torch.eye(2, device=covs2D.device).unsqueeze(0)
        
        # Compute the inverse analytically for 2x2 symmetric matrices. This is
        # more stable for autograd than a batched generic matrix inverse here.
        a = covs2D[:, 0, 0]
        b = 0.5 * (covs2D[:, 0, 1] + covs2D[:, 1, 0])
        c = covs2D[:, 1, 1]
        det = (a * c - b * b).clamp(min=eps)

        dx0 = dx[..., 0]
        dx1 = dx[..., 1]
        mahalanobis = (c.view(N, 1, 1) * dx0 * dx0
                       - 2.0 * b.view(N, 1, 1) * dx0 * dx1
                       + a.view(N, 1, 1) * dx1 * dx1) / det.view(N, 1, 1)
        power = (-0.5 * mahalanobis).clamp(min=-80.0, max=0.0)
        norm = 1.0 / (2.0 * torch.pi * torch.sqrt(det))
        gaussian = norm.view(N, 1, 1) * torch.exp(power)
    
        return gaussian

    def forward(
            self,
            means3D: torch.Tensor,          # (N, 3)
            covs3d: torch.Tensor,           # (N, 3, 3)
            colors: torch.Tensor,           # (N, 3)
            opacities: torch.Tensor,        # (N, 1)
            K: torch.Tensor,                # (3, 3)
            R: torch.Tensor,                # (3, 3)
            t: torch.Tensor                 # (3, 1)
    ) -> torch.Tensor:
        N = means3D.shape[0]
        
        # 1. Project to 2D, means2D: (N, 2), covs2D: (N, 2, 2), depths: (N,)
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        
        # 2. Depth mask
        valid_mask = (depths > 1.) & (depths < 50.0)  # (N,)
        
        # 3. Sort by depth
        indices = torch.argsort(depths, dim=0, descending=False)  # (N, )
        means2D = means2D[indices]      # (N, 2)
        covs2D = covs2D[indices]       # (N, 2, 2)
        colors = colors[ indices]       # (N, 3)
        opacities = opacities[indices] # (N, 1)
        valid_mask = valid_mask[indices] # (N,)
        
        # 4. Compute gaussian values
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)  # (N, H, W)
        
        # 5. Apply valid mask
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)  # (N, H, W)
        
        # 6. Alpha composition setup
        alphas = opacities.view(N, 1, 1) * gaussian_values  # (N, H, W)
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N, 3, H, W)
        colors = colors.permute(0, 2, 3, 1)  # (N, H, W, 3)
        
        # 7. Compute weights
        alphas = alphas.clamp(0.0, 0.999)
        transmittance = torch.cumprod(
            torch.cat([
                torch.ones((1, self.H, self.W), device=alphas.device, dtype=alphas.dtype),
                1.0 - alphas + 1e-10,
            ], dim=0),
            dim=0
        )[:-1]
        weights = alphas * transmittance  # (N, H, W)
        
        # 8. Final rendering
        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H, W, 3)
        
        return rendered

"""
Metrics computation for evaluating reconstruction quality.

Implements PSNR, SSIM, and LPIPS metrics.

TODO:
    - Implement compute_psnr(img1, img2) -> float
    - Implement compute_ssim(img1, img2) -> float
    - Implement compute_lpips(img1, img2, device) -> float
    - Handle normalization (metrics expect specific ranges)
    - Ensure batch processing is correct
    - Add unit tests with known ground truth values
"""

from typing import Dict, Optional
import torch
import numpy as np


def compute_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 2.0
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        img1: First image [C, H, W] or [B, C, H, W]
        img2: Second image (same shape)
        data_range: Range of values (2.0 for [-1, 1])

    Returns:
        PSNR in dB

    TODO: Implement PSNR computation
    """
    raise NotImplementedError("compute_psnr not yet implemented")


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 2.0
) -> float:
    """
    Compute Structural Similarity Index.

    Args:
        img1: First image [C, H, W] or [B, C, H, W]
        img2: Second image (same shape)
        data_range: Range of values (2.0 for [-1, 1])

    Returns:
        SSIM value in [0, 1]

    TODO: Implement SSIM using skimage
    """
    raise NotImplementedError("compute_ssim not yet implemented")


def compute_lpips(
    img1: torch.Tensor,
    img2: torch.Tensor,
    device: Optional[torch.device] = None
) -> float:
    """
    Compute LPIPS perceptual distance.

    Args:
        img1: First image [C, H, W] or [B, C, H, W] in [-1, 1]
        img2: Second image (same shape) in [-1, 1]
        device: Device for computation

    Returns:
        LPIPS distance (lower is better)

    TODO: Implement LPIPS metric (separate from loss)
    """
    raise NotImplementedError("compute_lpips not yet implemented")


def compute_all_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Compute all metrics (PSNR, SSIM, LPIPS) at once.

    Args:
        generated: Generated image [1, 3, H, W]
        target: Target image [1, 3, H, W]
        device: Device for computation

    Returns:
        Dictionary with metric names and values

    TODO: Implement batch metric computation
    """
    raise NotImplementedError("compute_all_metrics not yet implemented")


"""
Metrics computation for evaluating reconstruction quality.

Implements PSNR, SSIM, and LPIPS metrics for image comparison.
All metrics expect inputs in [-1, 1] range.
"""

from typing import Dict, Optional
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import warnings

# Try to import lpips for perceptual metrics
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

# Global LPIPS model (lazy initialization)
_LPIPS_MODEL = None


def _get_lpips_model(device: torch.device):
    """Get or initialize global LPIPS model."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        if not LPIPS_AVAILABLE:
            raise ImportError(
                "lpips package not found. Install with: pip install lpips"
            )
        _LPIPS_MODEL = lpips.LPIPS(net='alex', verbose=False).to(device)
        _LPIPS_MODEL.eval()
        for param in _LPIPS_MODEL.parameters():
            param.requires_grad = False
    # Move to correct device if needed
    if next(_LPIPS_MODEL.parameters()).device != device:
        _LPIPS_MODEL = _LPIPS_MODEL.to(device)
    return _LPIPS_MODEL


def compute_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 2.0
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(data_range^2 / MSE)
    Higher is better (infinite for identical images).

    Args:
        img1: First image [C, H, W] or [B, C, H, W] in [-1, 1]
        img2: Second image (same shape) in [-1, 1]
        data_range: Range of values (2.0 for [-1, 1], 1.0 for [0, 1])

    Returns:
        PSNR in dB (higher is better)

    Example:
        >>> img1 = torch.randn(1, 3, 256, 256).clamp(-1, 1)
        >>> img2 = torch.randn(1, 3, 256, 256).clamp(-1, 1)
        >>> psnr = compute_psnr(img1, img2)
    """
    # Validate shapes
    if img1.shape != img2.shape:
        raise ValueError(
            f"Shape mismatch: img1 {img1.shape} vs img2 {img2.shape}"
        )

    # Handle both 3D [C, H, W] and 4D [B, C, H, W]
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    elif img1.ndim != 4:
        raise ValueError(
            f"Expected 3D or 4D tensor, got shape {img1.shape}"
        )

    # Compute MSE
    mse = torch.mean((img1 - img2) ** 2).item()

    # Handle identical images (MSE = 0)
    if mse == 0:
        return float('inf')

    # Compute PSNR
    psnr = 10 * np.log10((data_range ** 2) / mse)
    return float(psnr)


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 2.0
) -> float:
    """
    Compute Structural Similarity Index using scikit-image.

    SSIM measures structural similarity between images.
    Range: [0, 1] or [-1, 1] (higher is better, 1.0 for identical images).

    Args:
        img1: First image [C, H, W] or [B, C, H, W] in [-1, 1]
        img2: Second image (same shape) in [-1, 1]
        data_range: Range of values (2.0 for [-1, 1], 1.0 for [0, 1])

    Returns:
        SSIM value (higher is better, 1.0 for identical)

    Example:
        >>> img1 = torch.randn(1, 3, 256, 256).clamp(-1, 1)
        >>> img2 = torch.randn(1, 3, 256, 256).clamp(-1, 1)
        >>> ssim_val = compute_ssim(img1, img2)
    """
    # Validate shapes
    if img1.shape != img2.shape:
        raise ValueError(
            f"Shape mismatch: img1 {img1.shape} vs img2 {img2.shape}"
        )

    # Handle both 3D [C, H, W] and 4D [B, C, H, W]
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    elif img1.ndim != 4:
        raise ValueError(
            f"Expected 3D or 4D tensor, got shape {img1.shape}"
        )

    # Convert to numpy and move channel to last dimension
    # [B, C, H, W] -> [B, H, W, C]
    img1_np = img1.detach().cpu().permute(0, 2, 3, 1).numpy()
    img2_np = img2.detach().cpu().permute(0, 2, 3, 1).numpy()

    # Compute SSIM for each image in batch and average
    ssim_values = []
    for i in range(img1_np.shape[0]):
        # skimage.metrics.structural_similarity expects channel_axis parameter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ssim_val = ssim(
                img1_np[i],
                img2_np[i],
                data_range=data_range,
                channel_axis=2,  # Channel is last dimension
                win_size=11
            )
        ssim_values.append(ssim_val)

    return float(np.mean(ssim_values))


def compute_lpips(
    img1: torch.Tensor,
    img2: torch.Tensor,
    device: Optional[torch.device] = None,
    net: str = 'alex'
) -> float:
    """
    Compute LPIPS perceptual distance.

    LPIPS measures perceptual similarity using deep features.
    Lower values indicate more perceptually similar images.

    Args:
        img1: First image [C, H, W] or [B, C, H, W] in [-1, 1]
        img2: Second image (same shape) in [-1, 1]
        device: Device for computation (default: img1.device)
        net: Network backbone ('alex', 'vgg', 'squeeze')

    Returns:
        LPIPS distance (lower is better, ~0 for identical)

    Example:
        >>> img1 = torch.randn(1, 3, 256, 256).clamp(-1, 1)
        >>> img2 = torch.randn(1, 3, 256, 256).clamp(-1, 1)
        >>> lpips_val = compute_lpips(img1, img2, device='cuda')

    Raises:
        ImportError: If lpips package is not installed
    """
    if not LPIPS_AVAILABLE:
        raise ImportError(
            "lpips package not found. Install with: pip install lpips"
        )

    # Validate shapes
    if img1.shape != img2.shape:
        raise ValueError(
            f"Shape mismatch: img1 {img1.shape} vs img2 {img2.shape}"
        )

    # Handle both 3D [C, H, W] and 4D [B, C, H, W]
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    elif img1.ndim != 4:
        raise ValueError(
            f"Expected 3D or 4D tensor, got shape {img1.shape}"
        )

    # Determine device
    if device is None:
        device = img1.device

    # Get LPIPS model
    lpips_model = _get_lpips_model(device)

    # Move images to device
    img1 = img1.to(device)
    img2 = img2.to(device)

    # Compute LPIPS
    with torch.no_grad():
        lpips_val = lpips_model(img1, img2)

    return float(lpips_val.mean().item())


def compute_all_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
    device: Optional[torch.device] = None,
    data_range: float = 2.0
) -> Dict[str, float]:
    """
    Compute all metrics (PSNR, SSIM, LPIPS) at once.

    Args:
        generated: Generated image [C, H, W] or [B, C, H, W] in [-1, 1]
        target: Target image (same shape) in [-1, 1]
        device: Device for LPIPS computation (default: generated.device)
        data_range: Range of values (2.0 for [-1, 1])

    Returns:
        Dictionary with metric names and values:
        - 'psnr': Peak Signal-to-Noise Ratio (dB, higher is better)
        - 'ssim': Structural Similarity Index (higher is better)
        - 'lpips': Perceptual distance (lower is better)

    Example:
        >>> generated = torch.randn(1, 3, 256, 256).clamp(-1, 1)
        >>> target = torch.randn(1, 3, 256, 256).clamp(-1, 1)
        >>> metrics = compute_all_metrics(generated, target)
        >>> print(f"PSNR: {metrics['psnr']:.2f} dB")
    """
    metrics = {}

    # Compute PSNR
    metrics['psnr'] = compute_psnr(generated, target, data_range=data_range)

    # Compute SSIM
    metrics['ssim'] = compute_ssim(generated, target, data_range=data_range)

    # Compute LPIPS (if available)
    try:
        metrics['lpips'] = compute_lpips(generated, target, device=device)
    except ImportError:
        metrics['lpips'] = None

    return metrics


"""
Visualization utilities for creating comparison panels and loss curves.

Provides functions to create side-by-side comparisons, plot loss curves,
and save image grids for analysis.
"""

from typing import Dict, List, Optional, Any
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (must be before pyplot)
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_comparison_panel(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    save_path: Optional[str] = None
) -> None:
    """
    Create side-by-side comparison panel with difference map.

    Creates a 3-panel visualization:
    - Left: Original image
    - Center: Reconstruction
    - Right: Absolute difference (scaled for visibility)

    Args:
        original: Original image [1, 3, H, W] or [3, H, W] in [-1, 1]
        reconstruction: Reconstructed image [1, 3, H, W] or [3, H, W] in [-1, 1]
        save_path: Path to save panel (required)

    Raises:
        ValueError: If shapes don't match or save_path not provided

    Example:
        >>> create_comparison_panel(orig, recon, 'output/comparison.png')
    """
    if save_path is None:
        raise ValueError("save_path is required")
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove batch dimension if present
    if original.dim() == 4:
        original = original.squeeze(0)
    if reconstruction.dim() == 4:
        reconstruction = reconstruction.squeeze(0)
    
    if original.shape != reconstruction.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstruction.shape}")
    
    # Move to CPU and convert to numpy
    original = original.detach().cpu()
    reconstruction = reconstruction.detach().cpu()
    
    # Denormalize from [-1, 1] to [0, 1]
    original = (original + 1) / 2
    reconstruction = (reconstruction + 1) / 2
    
    # Compute absolute difference
    diff = torch.abs(original - reconstruction)
    
    # Convert to numpy and transpose to HWC format
    original_np = original.permute(1, 2, 0).numpy()
    reconstruction_np = reconstruction.permute(1, 2, 0).numpy()
    diff_np = diff.permute(1, 2, 0).numpy()
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original_np)
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Reconstruction
    axes[1].imshow(reconstruction_np)
    axes[1].set_title('Reconstruction', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Difference (scaled for visibility)
    diff_scaled = diff_np * 5  # Amplify for visibility
    diff_scaled = np.clip(diff_scaled, 0, 1)
    axes[2].imshow(diff_scaled)
    axes[2].set_title('Difference (×5)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.debug(f"Saved comparison panel to: {save_path}")


def create_evolution_panel(
    original: torch.Tensor,
    intermediates: Dict[int, torch.Tensor],
    metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create multi-panel comparison showing evolution over iterations.
    
    Shows original image and reconstructions at different iterations
    (every 100 steps including step 0) with metrics in the title.
    
    Args:
        original: Original image [1, 3, H, W] or [3, H, W] in [-1, 1]
        intermediates: Dict mapping step -> reconstruction tensor
        metrics: Optional metrics dict with PSNR, SSIM, LPIPS
        save_path: Path to save panel (required)
    
    Example:
        >>> intermediates = {0: img0, 100: img100, 200: img200, 300: img300}
        >>> metrics = {'psnr': 25.5, 'ssim': 0.85, 'lpips': 0.15}
        >>> create_evolution_panel(orig, intermediates, metrics, 'out.png')
    """
    if save_path is None:
        raise ValueError("save_path is required")
    
    if not intermediates:
        raise ValueError("intermediates dict is empty")
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove batch dimension if present
    if original.dim() == 4:
        original = original.squeeze(0)
    
    # Move to CPU and denormalize
    original = original.detach().cpu()
    original = (original + 1) / 2
    original_np = original.permute(1, 2, 0).numpy()
    
    # Sort intermediate steps
    steps = sorted(intermediates.keys())
    n_intermediates = len(steps)
    
    # Create figure: original + intermediates
    n_panels = n_intermediates + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 3, 3))
    
    # Handle single panel case
    if n_panels == 1:
        axes = [axes]
    
    # Plot original
    axes[0].imshow(original_np)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot intermediates
    for idx, step in enumerate(steps):
        intermediate = intermediates[step]
        
        # Remove batch dimension and process
        if intermediate.dim() == 4:
            intermediate = intermediate.squeeze(0)
        
        intermediate = intermediate.detach().cpu()
        intermediate = (intermediate + 1) / 2
        intermediate_np = intermediate.permute(1, 2, 0).numpy()
        
        axes[idx + 1].imshow(intermediate_np)
        axes[idx + 1].set_title(f'Step {step}', fontsize=12, fontweight='bold')
        axes[idx + 1].axis('off')
    
    # Add metrics to suptitle if provided
    if metrics:
        psnr = metrics.get('psnr', 0)
        ssim = metrics.get('ssim', 0)
        lpips = metrics.get('lpips', 0)
        title = f'Inversion Evolution  |  PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}  |  LPIPS: {lpips:.4f}'
    else:
        title = 'Inversion Evolution'
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.debug(f"Saved evolution panel to: {save_path}")


def plot_loss_curve(
    history: Dict[str, Any],
    save_path: str,
    title: str = "Inversion Loss Curve"
) -> None:
    """
    Plot and save loss convergence curve.

    Args:
        history: Dictionary with 'loss' key containing loss per step
        save_path: Path to save plot (PNG)
        title: Plot title

    Raises:
        ValueError: If history doesn't contain 'loss' key

    Example:
        >>> history = {'loss': [1.0, 0.8, 0.6, 0.5]}
        >>> plot_loss_curve(history, 'outputs/loss.png')
    """
    if 'loss' not in history:
        raise ValueError("history must contain 'loss' key")
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    losses = history['loss']
    steps = list(range(1, len(losses) + 1))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot loss
    ax.plot(steps, losses, linewidth=2, color='#2E86AB', alpha=0.8)
    ax.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add final loss annotation
    final_loss = losses[-1]
    ax.annotate(
        f'Final: {final_loss:.6f}',
        xy=(len(losses), final_loss),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.debug(f"Saved loss curve to: {save_path}")


def save_grid(
    images: List[torch.Tensor],
    save_path: str,
    nrow: int = 4,
    titles: Optional[List[str]] = None
) -> None:
    """
    Save multiple images in a grid layout.

    Args:
        images: List of image tensors [3, H, W] or [1, 3, H, W]
        save_path: Output path
        nrow: Number of images per row
        titles: Optional titles for each image

    Example:
        >>> images = [torch.randn(3, 256, 256) for _ in range(8)]
        >>> save_grid(images, 'output/grid.png', nrow=4)
    """
    if not images:
        raise ValueError("images list is empty")
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process images
    processed = []
    for img in images:
        if img.dim() == 4:
            img = img.squeeze(0)
        
        img = img.detach().cpu()
        
        # Denormalize if needed
        if img.min() < -0.1:  # Likely in [-1, 1]
            img = (img + 1) / 2
        
        img = img.permute(1, 2, 0).numpy()
        processed.append(img)
    
    # Calculate grid dimensions
    n_images = len(processed)
    ncol = nrow
    nrow_actual = (n_images + ncol - 1) // ncol
    
    # Create figure
    fig, axes = plt.subplots(nrow_actual, ncol, figsize=(ncol * 3, nrow_actual * 3))
    
    # Handle single row/column cases
    if nrow_actual == 1 and ncol == 1:
        axes = np.array([[axes]])
    elif nrow_actual == 1:
        axes = axes.reshape(1, -1)
    elif ncol == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot images
    for idx in range(nrow_actual * ncol):
        row = idx // ncol
        col = idx % ncol
        ax = axes[row, col]
        
        if idx < n_images:
            ax.imshow(processed[idx])
            if titles and idx < len(titles):
                ax.set_title(titles[idx], fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.debug(f"Saved grid to: {save_path}")


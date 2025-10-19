"""
Visualization utilities for creating comparison panels and loss curves.

TODO:
    - Implement create_comparison_panel(original, reconstruction, diff) -> panel
    - Implement plot_loss_curve(history, save_path)
    - Support side-by-side visualizations
    - Add titles and labels to plots
    - Handle different image sizes gracefully
"""

from typing import Dict, List, Optional
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def create_comparison_panel(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    save_path: Optional[str] = None
) -> torch.Tensor:
    """
    Create side-by-side comparison panel with difference map.

    Args:
        original: Original image [3, H, W] in [-1, 1]
        reconstruction: Reconstructed image [3, H, W] in [-1, 1]
        save_path: Optional path to save panel

    Returns:
        Panel tensor [3, H, W*3]

    TODO:
        - Create 3-panel layout: original | reconstruction | abs diff
        - Add labels/titles
        - Optionally save to file
    """
    raise NotImplementedError("create_comparison_panel not yet implemented")


def plot_loss_curve(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Inversion Loss Curve"
) -> None:
    """
    Plot and save loss convergence curve.

    Args:
        history: Dictionary with 'loss' key containing loss per step
        save_path: Path to save plot (PNG)
        title: Plot title

    TODO:
        - Plot loss vs. step
        - Add grid, labels, title
        - Save to file
    """
    raise NotImplementedError("plot_loss_curve not yet implemented")


def save_grid(
    images: List[torch.Tensor],
    save_path: str,
    nrow: int = 4,
    titles: Optional[List[str]] = None
) -> None:
    """
    Save multiple images in a grid layout.

    Args:
        images: List of image tensors [3, H, W]
        save_path: Output path
        nrow: Number of images per row
        titles: Optional titles for each image

    TODO: Implement grid visualization
    """
    raise NotImplementedError("save_grid not yet implemented")


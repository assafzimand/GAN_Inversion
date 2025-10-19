"""
Image I/O utilities for loading and saving images.

TODO:
    - Implement load_image(path, size=(1024, 1024), device) -> tensor
    - Implement save_image(tensor, path)
    - Normalize to [-1, 1] for model input
    - Handle RGB/RGBA conversion
    - Support batch loading from folder
    - Add validation for image sizes and formats
"""

from typing import Optional, Tuple, List
import torch
from pathlib import Path


def load_image(
    image_path: str,
    size: Tuple[int, int] = (1024, 1024),
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Load and preprocess image for GAN inversion.

    Args:
        image_path: Path to image file
        size: Target size (H, W)
        device: Target device

    Returns:
        Image tensor [1, 3, H, W] in [-1, 1]

    TODO: Implement image loading and preprocessing
    """
    raise NotImplementedError("load_image not yet implemented")


def save_image(
    tensor: torch.Tensor,
    save_path: str,
    denormalize: bool = True
) -> None:
    """
    Save tensor as image file.

    Args:
        tensor: Image tensor [1, 3, H, W] or [3, H, W]
        save_path: Output path
        denormalize: If True, convert from [-1, 1] to [0, 255]

    TODO: Implement image saving
    """
    raise NotImplementedError("save_image not yet implemented")


def load_images_from_folder(
    folder_path: str,
    size: Tuple[int, int] = (1024, 1024),
    device: Optional[torch.device] = None
) -> List[Tuple[str, torch.Tensor]]:
    """
    Load all images from a folder.

    Args:
        folder_path: Path to folder containing images
        size: Target size (H, W)
        device: Target device

    Returns:
        List of (filename, tensor) tuples

    TODO: Implement batch loading
    """
    raise NotImplementedError("load_images_from_folder not yet implemented")


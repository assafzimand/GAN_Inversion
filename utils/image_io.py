"""
Image I/O utilities for loading and saving images.

Handles image loading, preprocessing (resizing, normalization),
and saving for GAN inversion pipeline.
"""

from typing import Optional, Tuple, List
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import logging

logger = logging.getLogger(__name__)


def load_image(
    image_path: str,
    size: Tuple[int, int] = (128, 128),
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Load and preprocess image for GAN inversion.

    Loads image, converts to RGB, resizes, and normalizes to [-1, 1].

    Args:
        image_path: Path to image file
        size: Target size (H, W), default (128, 128)
        device: Target device (defaults to CPU)

    Returns:
        Image tensor [1, 3, H, W] in [-1, 1]

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded

    Example:
        >>> img = load_image('data/face.png', size=(128, 128), device='cuda')
        >>> img.shape
        torch.Size([1, 3, 128, 128])
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    logger.debug(f"Loading image: {image_path}")
    
    try:
        # Load image with PIL
        img = Image.open(image_path)
        
        # Convert to RGB (handles RGBA, grayscale, etc.)
        img = img.convert('RGB')
        
        # Create transform pipeline
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),  # Converts to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Converts to [-1, 1]
        ])
        
        # Apply transforms
        tensor = transform(img)  # [3, H, W]
        tensor = tensor.unsqueeze(0)  # [1, 3, H, W]
        
        # Move to device if specified
        if device is not None:
            tensor = tensor.to(device)
        
        logger.debug(f"Loaded image shape: {tensor.shape}, range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        
        return tensor
        
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


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
        denormalize: If True, convert from [-1, 1] to [0, 1]

    Raises:
        ValueError: If tensor shape is invalid

    Example:
        >>> img = torch.randn(1, 3, 256, 256)
        >>> save_image(img, 'output.png')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle batch dimension
    if tensor.dim() == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"Batch size must be 1, got {tensor.shape[0]}")
        tensor = tensor.squeeze(0)  # [3, H, W]
    elif tensor.dim() != 3:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tensor.shape}")
    
    # Move to CPU
    tensor = tensor.detach().cpu()
    
    # Denormalize from [-1, 1] to [0, 1]
    if denormalize:
        tensor = (tensor + 1) / 2
        tensor = tensor.clamp(0, 1)
    
    # Convert to PIL Image
    transform = transforms.ToPILImage()
    img = transform(tensor)
    
    # Save
    img.save(save_path)
    logger.debug(f"Saved image to: {save_path}")


def load_images_from_folder(
    folder_path: str,
    size: Tuple[int, int] = (128, 128),
    device: Optional[torch.device] = None,
    extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
) -> List[Tuple[str, torch.Tensor]]:
    """
    Load all images from a folder.

    Args:
        folder_path: Path to folder containing images
        size: Target size (H, W), default (128, 128)
        device: Target device
        extensions: Tuple of valid file extensions

    Returns:
        List of (filename, tensor) tuples

    Raises:
        FileNotFoundError: If folder doesn't exist
        ValueError: If no valid images found

    Example:
        >>> images = load_images_from_folder('data/samples/', size=(128, 128))
        >>> len(images)
        5
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(folder_path.glob(f'*{ext}'))
        image_files.extend(folder_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        raise ValueError(f"No images found in {folder_path} with extensions {extensions}")
    
    # Sort by filename for consistency
    image_files = sorted(image_files)
    
    logger.info(f"Found {len(image_files)} images in {folder_path}")
    
    # Load all images
    loaded_images = []
    for img_path in image_files:
        try:
            tensor = load_image(str(img_path), size=size, device=device)
            loaded_images.append((img_path.name, tensor))
            logger.debug(f"Loaded: {img_path.name}")
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
    
    if not loaded_images:
        raise ValueError(f"Failed to load any images from {folder_path}")
    
    logger.info(f"Successfully loaded {len(loaded_images)} images")
    
    return loaded_images


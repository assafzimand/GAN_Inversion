"""
Encoder loader for GAN inversion initialization.

Provides encoders that map images to StyleGAN latent codes (W or W+)
for use as initialization in optimization-based inversion.

Supports:
- SimpleEncoder: Custom lightweight encoder for 128×128 images
- e4e/pSp: Pre-trained encoders from 1024×1024 with automatic upscaling
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def upscale_image(
    image: torch.Tensor,
    target_size: int = 1024,
    mode: str = 'bicubic'
) -> torch.Tensor:
    """
    Upscale image for encoder input.
    
    Used to upscale 128×128 images to 1024×1024 for pre-trained encoders.
    
    Args:
        image: Input image [B, C, H, W] in [-1, 1]
        target_size: Target image size (assumes square)
        mode: Interpolation mode ('bicubic', 'bilinear', 'nearest')
    
    Returns:
        Upscaled image [B, C, target_size, target_size] in [-1, 1]
    
    Example:
        >>> img_128 = torch.randn(1, 3, 128, 128)
        >>> img_1024 = upscale_image(img_128, 1024)
        >>> img_1024.shape
        torch.Size([1, 3, 1024, 1024])
    """
    if image.shape[-1] == target_size and image.shape[-2] == target_size:
        return image
    
    # Use align_corners=False for bicubic (recommended for StyleGAN)
    upscaled = F.interpolate(
        image,
        size=(target_size, target_size),
        mode=mode,
        align_corners=False if mode in ['bicubic', 'bilinear'] else None
    )
    
    logger.debug(f"Upscaled image from {image.shape[-1]}×{image.shape[-2]} to {target_size}×{target_size}")
    
    return upscaled


class SimpleEncoder(nn.Module):
    """
    Simple ResNet-based encoder for 128×128 images to W+ latent space.
    
    Maps RGB images to StyleGAN W+ latent codes [num_layers, 512].
    This is a lightweight architecture suitable for 128×128 resolution.
    
    Note: This encoder needs to be trained. Pre-trained weights for 128×128
    StyleGAN encoders are limited. Consider:
    - Training on your specific dataset
    - Fine-tuning from a larger resolution encoder
    - Using synthetic data from the StyleGAN generator
    """
    
    def __init__(
        self,
        input_size: int = 128,
        latent_dim: int = 512,
        num_layers: int = 12,
        output_space: str = "W+"
    ):
        """
        Initialize encoder.
        
        Args:
            input_size: Input image size (assumes square)
            latent_dim: StyleGAN latent dimension (usually 512)
            num_layers: Number of StyleGAN layers (for W+ space)
            output_space: "W" for [1, 512] or "W+" for [num_layers, 512]
        """
        super().__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.output_space = output_space
        
        # Convolutional backbone (ResNet-inspired)
        self.conv_layers = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 32 -> 16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 16 -> 8
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 8 -> 4
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Fully connected layers to latent space
        if output_space == "W+":
            # Output W+ codes: [num_layers, latent_dim]
            self.fc = nn.Linear(512, num_layers * latent_dim)
        else:
            # Output W code: [latent_dim]
            self.fc = nn.Linear(512, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent code.
        
        Args:
            x: Input image [B, 3, H, W] in [-1, 1]
        
        Returns:
            latent: W code [B, 512] or W+ code [B, num_layers, 512]
        """
        # Extract features
        features = self.conv_layers(x)  # [B, 512, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 512]
        
        # Map to latent space
        latent = self.fc(features)  # [B, latent_dim] or [B, num_layers * latent_dim]
        
        if self.output_space == "W+":
            # Reshape to [B, num_layers, latent_dim]
            latent = latent.view(-1, self.num_layers, self.latent_dim)
        
        return latent


def load_encoder(
    encoder_config: Dict[str, Any],
    device: torch.device
) -> nn.Module:
    """
    Load encoder model from configuration.
    
    Args:
        encoder_config: Configuration dict with encoder settings
        device: Target device
    
    Returns:
        Loaded encoder model
    
    Raises:
        ValueError: If encoder type is not supported
        FileNotFoundError: If checkpoint file not found
    
    Example:
        >>> config = {
        ...     'encoder_type': 'simple_encoder',
        ...     'encoder_checkpoint': 'checkpoints/encoder_128.pt',
        ...     'encoder_config': {'input_size': 128, 'num_layers': 12},
        ...     'output_space': 'W+'
        ... }
        >>> encoder = load_encoder(config, torch.device('cuda'))
    """
    encoder_type = encoder_config.get('encoder_type', 'simple_encoder')
    checkpoint_path = encoder_config.get('encoder_checkpoint', None)
    enc_cfg = encoder_config.get('encoder_config', {})
    output_space = encoder_config.get('output_space', 'W+')
    
    logger.info(f"Loading encoder: type={encoder_type}")
    
    if encoder_type == 'simple_encoder':
        # Create simple encoder
        encoder = SimpleEncoder(
            input_size=enc_cfg.get('input_size', 128),
            latent_dim=enc_cfg.get('output_dim', 512),
            num_layers=enc_cfg.get('num_layers', 12),
            output_space=output_space
        )
        
        # Load checkpoint if provided
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                logger.warning(
                    f"Encoder checkpoint not found: {checkpoint_path}\n"
                    f"Using randomly initialized encoder (not recommended for production).\n"
                    f"Consider training an encoder or using optimization-only mode."
                )
            else:
                logger.info(f"Loading encoder weights from: {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location=device)
                
                # Handle different checkpoint formats
                if 'encoder' in state_dict:
                    encoder.load_state_dict(state_dict['encoder'])
                elif 'state_dict' in state_dict:
                    encoder.load_state_dict(state_dict['state_dict'])
                else:
                    encoder.load_state_dict(state_dict)
                
                logger.info("Encoder weights loaded successfully")
        else:
            logger.warning(
                "No checkpoint path provided. Using randomly initialized encoder.\n"
                "This will produce poor results. Please provide trained weights."
            )
    
    elif encoder_type in ['e4e', 'psp']:
        # Load pre-trained e4e or pSp encoder
        if not checkpoint_path:
            raise ValueError(
                f"Encoder checkpoint path required for {encoder_type}.\n"
                f"Please download from:\n"
                f"  e4e: https://github.com/omertov/encoder4editing\n"
                f"  pSp: https://github.com/eladrich/pixel2style2pixel"
            )
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Encoder checkpoint not found: {checkpoint_path}\n"
                f"Please download {encoder_type} weights and place "
                f"in checkpoints/\n"
                f"See README.md for download instructions."
            )
        
        logger.info(f"Loading {encoder_type} encoder from: {checkpoint_path}")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # Check if required dependencies are available
        try:
            import sys
            
            # Add e4e repo to path if it exists
            e4e_path = Path('external/encoder4editing')
            if e4e_path.exists():
                sys.path.insert(0, str(e4e_path.absolute()))
            
            # Try to import pSp model
            try:
                from models.psp import pSp
            except ImportError:
                logger.warning(
                    "e4e/pSp models not found. Please install:\n"
                    "  git clone https://github.com/omertov/encoder4editing.git "
                    "external/encoder4editing\n"
                    "Or see README.md for instructions."
                )
                raise ImportError(
                    f"Cannot load {encoder_type} encoder: pSp model not found.\n"
                    f"Please clone the e4e repository:\n"
                    f"  git clone https://github.com/omertov/encoder4editing.git "
                    f"external/encoder4editing\n"
                    f"Then install its dependencies:\n"
                    f"  pip install -r external/encoder4editing/requirements.txt"
                )
            
            # Load the full pSp model
            model = pSp(ckpt['opts'])
            model.load_state_dict(ckpt['state_dict'], strict=True)
            model.eval()
            model = model.to(device)
            
            # Extract just the encoder part
            encoder_module = model.encoder
            
            logger.info("Successfully loaded encoder using pSp architecture")
        
        except (ImportError, KeyError, AttributeError) as e:
            raise RuntimeError(
                f"Failed to load {encoder_type} encoder: {e}\n\n"
                f"To use encoder-based initialization, you need to:\n"
                f"1. Clone the e4e repository:\n"
                f"   git clone https://github.com/omertov/encoder4editing.git "
                f"external/encoder4editing\n"
                f"2. Install e4e dependencies:\n"
                f"   pip install -r external/encoder4editing/requirements.txt\n"
                f"3. Download encoder weights (already done: {checkpoint_path})\n\n"
                f"Alternatively, use optimization-only modes:\n"
                f"  --preset combo_01  # W space + L2\n"
                f"  --preset combo_02  # W+ space + L2\n"
                f"  --preset combo_03  # W+ space + LPIPS"
            )
        
        # Wrap in a standardized interface
        class E4EEncoderWrapper(nn.Module):
            """Wrapper for e4e/pSp encoders to standardize output."""
            def __init__(self, base_encoder, output_space='W', latent_avg=None):
                super().__init__()
                self.encoder = base_encoder
                self.output_space = output_space
                # Store latent_avg if available
                if latent_avg is not None:
                    self.register_buffer('latent_avg', latent_avg)
                else:
                    self.latent_avg = None
            
            def forward(self, x):
                # Encode image
                latent = self.encoder(x)
                
                # Handle different output formats
                if isinstance(latent, tuple):
                    latent = latent[0]  # Take first output if tuple
                
                # Add latent_avg if available (e4e typically outputs deltas)
                if self.latent_avg is not None and latent.shape[-1] == 512:
                    if latent.dim() == 3 and self.latent_avg.dim() == 2:
                        # Broadcast latent_avg to match
                        latent = latent + self.latent_avg.unsqueeze(0)
                    elif latent.dim() == 2 and self.latent_avg.dim() == 2:
                        # Average latent_avg if needed
                        latent = latent + self.latent_avg.mean(dim=0, keepdim=True)
                
                # Convert to expected format
                if latent.dim() == 3:
                    # W+ output [B, num_layers, 512]
                    if self.output_space == 'W':
                        # Average across layers to get W [B, 512]
                        latent = latent.mean(dim=1)
                elif latent.dim() == 2:
                    # Already W [B, 512]
                    pass
                else:
                    raise ValueError(
                        f"Unexpected encoder output shape: {latent.shape}"
                    )
                
                return latent
        
        # Get latent_avg if available
        latent_avg = ckpt.get('latent_avg', None)
        if latent_avg is not None:
            logger.info(f"Using latent_avg from checkpoint: {latent_avg.shape}")
        
        encoder = E4EEncoderWrapper(
            encoder_module,
            output_space=output_space,
            latent_avg=latent_avg
        )
        encoder = encoder.to(device)
        encoder.eval()
        logger.info(f"Loaded {encoder_type} encoder successfully")
    
    elif encoder_type == 'custom':
        raise NotImplementedError(
            "Custom encoder type not implemented.\n"
            "Please use 'simple_encoder', 'e4e', or 'psp'."
        )
    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}\n"
            f"Supported types: 'simple_encoder', 'e4e', 'psp', 'custom'"
        )
    
    # Move to device and set to eval mode
    encoder = encoder.to(device)
    encoder.eval()
    
    logger.info(f"Encoder loaded and ready (output_space={output_space})")
    
    return encoder


def encode_image(
    encoder: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    encoder_config: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """
    Encode image to latent code with automatic upscaling if needed.
    
    For e4e/pSp encoders trained on 1024×1024, automatically upscales
    128×128 images before encoding.
    
    Args:
        encoder: Encoder model
        image: Input image [1, 3, H, W] in [-1, 1]
        device: Device for computation
        encoder_config: Optional config dict with upscaling settings
    
    Returns:
        latent: Encoded latent code [1, 512] (W space)
    
    Example:
        >>> # Image will be auto-upscaled if needed
        >>> latent = encode_image(encoder, image_128, device, config)
        >>> latent.shape
        torch.Size([1, 512])  # W space
    """
    encoder.eval()
    image = image.to(device)
    
    # Check if upscaling is needed
    if encoder_config:
        enc_cfg = encoder_config.get('encoder_config', {})
        target_size = enc_cfg.get('input_size', 1024)
        our_size = enc_cfg.get('our_image_size', 128)
        upscale_method = encoder_config.get('upscale_method', 'bicubic')
        
        # Upscale if current image is smaller than encoder expects
        if image.shape[-1] == our_size and target_size > our_size:
            logger.info(f"Upscaling image {our_size}×{our_size} → {target_size}×{target_size} for encoder")
            image = upscale_image(image, target_size, mode=upscale_method)
    
    with torch.no_grad():
        latent = encoder(image)
    
    logger.debug(f"Encoded image to latent shape: {latent.shape}")
    
    return latent


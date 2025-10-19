"""
StyleGAN2 model loader and utilities.

Provides abstractions for loading pretrained StyleGAN2 generators
and computing latent statistics (e.g., mean_w).

Supports common StyleGAN2 checkpoint formats:
- HuggingFace Hub models (via huggingface_hub) - **recommended**
- PyTorch .pt/.pth format (rosinality, etc.)
"""

from typing import Optional, Tuple, Union
import sys
import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_hf_stylegan(
    model_id: str,
    device: torch.device
) -> nn.Module:
    """
    Load StyleGAN model from HuggingFace Hub.
    
    Args:
        model_id: HuggingFace model ID (e.g., "hajar001/stylegan2-ffhq-128")
        device: Target device (cpu or cuda)
    
    Returns:
        Loaded StyleGAN model (nn.Module)
    
    Raises:
        ImportError: If huggingface_hub is not installed
        RuntimeError: If model loading fails
    
    Example:
        >>> G = load_hf_stylegan("hajar001/stylegan2-ffhq-128", torch.device('cuda'))
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to load models from HuggingFace.\n"
            "Install with: pip install huggingface-hub"
        )
    
    logger.info(f"Loading StyleGAN from HuggingFace: {model_id}")
    
    try:
        # Download the model architecture file
        model_file = hf_hub_download(
            repo_id=model_id,
            filename="style_gan.py"
        )
        logger.info(f"Downloaded model file: {model_file}")
        
        # Add model directory to path
        model_dir = str(Path(model_file).parent)
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        
        # Import the StyleGAN class
        from style_gan import StyleGAN
        
        # Load pretrained weights
        model = StyleGAN.from_pretrained(model_id)
        model = model.to(device)
        model.eval()
        
        logger.info(f"Successfully loaded HuggingFace model: {model_id}")
        logger.info(f"  - Model has 'mapping': {hasattr(model, 'mapping')}")
        logger.info(f"  - Model has 'synthesis': {hasattr(model, 'synthesis')}")
        
        return model
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load HuggingFace model '{model_id}': {str(e)}\n"
            f"Make sure the model exists and you have internet connectivity."
        )


def load_generator(
    weights_path: str,
    device: torch.device,
    latent_dim: int = 512,
    num_layers: int = 18
) -> nn.Module:
    """
    Load pretrained StyleGAN2 generator from checkpoint.

    Supports multiple checkpoint formats:
    - HuggingFace Hub (prefix with "hf://", e.g., "hf://hajar001/stylegan2-ffhq-128") - **recommended**
    - PyTorch state dict (.pt, .pth)

    Args:
        weights_path: Path to pretrained weights or HuggingFace model ID with "hf://" prefix
        device: Target device (cpu or cuda)
        latent_dim: Latent dimension (default: 512)
        num_layers: Number of style layers (default: 18 for 1024x1024)

    Returns:
        StyleGAN2Wrapper ready for inference

    Raises:
        FileNotFoundError: If weights file doesn't exist
        RuntimeError: If checkpoint format is not recognized

    Example:
        >>> device = torch.device('cuda')
        >>> # Load from HuggingFace
        >>> G = load_generator('hf://hajar001/stylegan2-ffhq-128', device)
        >>> # Load from local file
        >>> G = load_generator('checkpoints/stylegan2-ffhq.pt', device)
        >>> z = torch.randn(1, 512).to(device)
        >>> img = G(z, latent_space='W')
    """
    # Check if this is a HuggingFace model
    if weights_path.startswith("hf://"):
        model_id = weights_path[5:]  # Remove "hf://" prefix
        logger.info(f"Detected HuggingFace model: {model_id}")
        generator = load_hf_stylegan(model_id, device)
        return StyleGAN2Wrapper(generator, device)
    
    # Otherwise treat as local file
    weights_path = Path(weights_path)
    
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}\n"
            f"Please download StyleGAN2 weights and place in checkpoints/\n"
            f"Or use HuggingFace format: hf://model-id"
        )
    
    logger.info(f"Loading StyleGAN2 generator from {weights_path}")
    
    # Try to load based on file extension
    suffix = weights_path.suffix.lower()
    
    try:
        if suffix in ['.pt', '.pth']:
            # PyTorch format
            checkpoint = torch.load(weights_path, map_location='cpu')
            
            # Handle different checkpoint structures
            if isinstance(checkpoint, dict):
                if 'g_ema' in checkpoint:
                    # Rosinality format
                    generator = checkpoint['g_ema']
                elif 'generator' in checkpoint:
                    generator = checkpoint['generator']
                elif 'state_dict' in checkpoint:
                    generator = checkpoint['state_dict']
                else:
                    # Assume checkpoint is the state dict itself
                    generator = checkpoint
            else:
                # Direct model
                generator = checkpoint
        else:
            raise RuntimeError(
                f"Unsupported checkpoint format: {suffix}\n"
                f"Supported formats: .pt, .pth\n"
                f"For HuggingFace models, use 'hf://model-id' format"
            )
        
        # Wrap the generator
        wrapped_generator = StyleGAN2Wrapper(
            generator,
            latent_dim=latent_dim,
            num_layers=num_layers
        )
        
        # Move to device and set to eval mode
        wrapped_generator = wrapped_generator.to(device)
        wrapped_generator.eval()
        
        logger.info(f"Generator loaded successfully on {device}")
        return wrapped_generator
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load generator from {weights_path}: {str(e)}\n"
            f"Please ensure the checkpoint is a valid StyleGAN2 model."
        ) from e


def compute_mean_w(
    generator: nn.Module,
    num_samples: int = 10000,
    batch_size: int = 1000,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Compute mean W latent vector by sampling from Z space.

    Samples random Z vectors from N(0, I), maps them to W space
    through the generator's mapping network, and computes the mean.

    Args:
        generator: StyleGAN2 generator (wrapped)
        num_samples: Number of Z samples for mean estimation
        batch_size: Batch size for efficient sampling
        device: Device to run computation on (default: generator.device)
        seed: Random seed for reproducibility (optional)

    Returns:
        Mean W vector of shape [1, latent_dim] (typically [1, 512])

    Example:
        >>> G = load_generator('checkpoints/stylegan2-ffhq.pt', device='cuda')
        >>> mean_w = compute_mean_w(G, num_samples=10000)
        >>> print(mean_w.shape)  # [1, 512]
    """
    if device is None:
        device = next(generator.parameters()).device
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Get latent dimension
    if isinstance(generator, StyleGAN2Wrapper):
        latent_dim = generator.latent_dim
    else:
        latent_dim = 512  # Default for StyleGAN2
    
    generator.eval()
    
    logger.info(f"Computing mean W from {num_samples} samples...")
    
    w_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            # Calculate actual batch size for this iteration
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # Sample Z from N(0, I)
            z = torch.randn(current_batch_size, latent_dim, device=device)
            
            # Map to W space
            if isinstance(generator, StyleGAN2Wrapper):
                w = generator.map_to_w(z)
            else:
                # Try to call mapping network directly
                if hasattr(generator, 'style'):
                    w = generator.style(z)
                elif hasattr(generator, 'mapping'):
                    w = generator.mapping(z)
                else:
                    raise AttributeError(
                        "Generator doesn't have a recognized mapping network. "
                        "Expected 'style' or 'mapping' attribute."
                    )
            
            w_samples.append(w)
    
    # Concatenate all samples and compute mean
    all_w = torch.cat(w_samples, dim=0)
    mean_w = all_w.mean(dim=0, keepdim=True)
    
    logger.info(f"Mean W computed: shape {mean_w.shape}")
    return mean_w


class StyleGAN2Wrapper(nn.Module):
    """
    Wrapper for StyleGAN2 generator supporting W and W+ spaces.

    This wrapper provides a unified interface for StyleGAN2 generators
    from different implementations, supporting both:
    - W space: Single latent code broadcast to all layers [B, 512]
    - W+ space: Per-layer latent codes [B, num_layers, 512]

    Args:
        generator: StyleGAN2 generator module
        latent_dim: Dimension of latent codes (default: 512)
        num_layers: Number of synthesis layers (default: 18)

    Example:
        >>> G = StyleGAN2Wrapper(raw_generator)
        >>> # W space
        >>> w = torch.randn(1, 512)
        >>> img = G(w, latent_space='W')
        >>> # W+ space
        >>> w_plus = torch.randn(1, 18, 512)
        >>> img = G(w_plus, latent_space='W+')
    """

    def __init__(
        self,
        generator: nn.Module,
        device: torch.device,
        latent_dim: int = 512,
        num_layers: Optional[int] = None
    ):
        """Initialize wrapper with loaded generator."""
        super().__init__()
        self.generator = generator
        self.device = device
        self.latent_dim = latent_dim
        
        # Auto-detect num_layers from generator if not provided
        if num_layers is None:
            if hasattr(generator, 'synthesis') and hasattr(generator.synthesis, 'num_layers'):
                self.num_layers = generator.synthesis.num_layers
                logger.info(f"Auto-detected num_layers={self.num_layers} from generator")
            else:
                self.num_layers = 18  # Default for 1024x1024
                logger.info(f"Using default num_layers={self.num_layers}")
        else:
            self.num_layers = num_layers
        
        # Detect generator structure
        self._detect_generator_structure()
        
        logger.debug(
            f"StyleGAN2Wrapper initialized: "
            f"latent_dim={latent_dim}, num_layers={self.num_layers}"
        )

    def _detect_generator_structure(self):
        """Detect the structure of the loaded generator."""
        # Try to identify mapping and synthesis networks
        
        # Check for standard PyTorch format first (HuggingFace, modern implementations)
        # This checks for mapping and synthesis as direct attributes
        if hasattr(self.generator, 'mapping') and hasattr(self.generator, 'synthesis'):
            self.mapping_network = self.generator.mapping
            self.synthesis_network = self.generator.synthesis
            logger.info("Detected standard format (HuggingFace or modern PyTorch)")
        # Check for Network wrapper with components dict (legacy format)
        elif hasattr(self.generator, 'components') and isinstance(self.generator.components, dict):
            components = self.generator.components
            if 'mapping' in components:
                self.mapping_network = components['mapping']
            if 'synthesis' in components:
                self.synthesis_network = components['synthesis']
            logger.info("Detected NVIDIA Network wrapper structure")
        # Check for rosinality format (style attribute)
        elif hasattr(self.generator, 'style'):
            self.mapping_network = self.generator.style
            self.synthesis_network = self.generator
            logger.info("Detected rosinality format")
        else:
            # Assume the generator is the synthesis network
            self.mapping_network = None
            self.synthesis_network = self.generator
            logger.warning(
                "Could not detect mapping network. "
                "W space may not work correctly."
            )

    def map_to_w(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map Z space to W space through mapping network.

        Args:
            z: Latent codes in Z space [B, latent_dim]

        Returns:
            Latent codes in W space [B, latent_dim]
        """
        if self.mapping_network is None:
            raise RuntimeError("Mapping network not available")
        
        return self.mapping_network(z)

    def forward(
        self,
        latent: torch.Tensor,
        latent_space: str = "W",
        return_latents: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate image from latent code.

        Args:
            latent: Latent code
                - W space: [B, latent_dim]
                - W+ space: [B, num_layers, latent_dim]
            latent_space: "W" or "W+"
            return_latents: If True, return (image, latents)

        Returns:
            Generated image [B, 3, H, W] in [-1, 1]
            or (image, latents) if return_latents=True

        Raises:
            ValueError: If latent_space is not 'W' or 'W+'
            ValueError: If latent shape is incorrect
        """
        if latent_space not in ["W", "W+"]:
            raise ValueError(
                f"latent_space must be 'W' or 'W+', got '{latent_space}'"
            )
        
        # Validate and prepare latents
        if latent_space == "W":
            # W space: [B, latent_dim]
            if latent.ndim != 2:
                raise ValueError(
                    f"W space latent must be 2D [B, {self.latent_dim}], "
                    f"got shape {latent.shape}"
                )
            if latent.shape[1] != self.latent_dim:
                raise ValueError(
                    f"Expected latent_dim={self.latent_dim}, "
                    f"got {latent.shape[1]}"
                )
            # Broadcast to all layers: [B, latent_dim] -> [B, num_layers, latent_dim]
            latent_expanded = latent.unsqueeze(1).repeat(1, self.num_layers, 1)
            
        else:  # W+
            # W+ space: can accept both 2D (will broadcast) or 3D (per-layer)
            if latent.ndim == 2:
                # 2D W vector: broadcast to all layers
                if latent.shape[1] != self.latent_dim:
                    raise ValueError(
                        f"Expected latent_dim={self.latent_dim}, "
                        f"got {latent.shape[1]}"
                    )
                latent_expanded = latent.unsqueeze(1).repeat(1, self.num_layers, 1)
            elif latent.ndim == 3:
                # 3D W+ vector: per-layer codes
                if latent.shape[1] != self.num_layers:
                    raise ValueError(
                        f"Expected {self.num_layers} layers, got {latent.shape[1]}"
                    )
                if latent.shape[2] != self.latent_dim:
                    raise ValueError(
                        f"Expected latent_dim={self.latent_dim}, "
                        f"got {latent.shape[2]}"
                    )
                latent_expanded = latent
            else:
                raise ValueError(
                    f"W+ space latent must be 2D [B, {self.latent_dim}] or "
                    f"3D [B, {self.num_layers}, {self.latent_dim}], "
                    f"got shape {latent.shape}"
                )
        
        # Generate image
        # Try different synthesis calling conventions
        try:
            if hasattr(self.synthesis_network, '__call__'):
                # Try with styles parameter
                image = self.synthesis_network(latent_expanded)
            else:
                raise AttributeError("Synthesis network is not callable")
                
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate image: {str(e)}\n"
                f"Latent shape: {latent_expanded.shape}"
            ) from e
        
        if return_latents:
            return image, latent_expanded
        return image

    def generate_from_z(
        self,
        z: torch.Tensor,
        latent_space: str = "W"
    ) -> torch.Tensor:
        """
        Generate image directly from Z space.

        Args:
            z: Random latent codes [B, latent_dim]
            latent_space: Target latent space ("W" or "W+")

        Returns:
            Generated image [B, 3, H, W] in [-1, 1]
        """
        # Map Z to W
        w = self.map_to_w(z)
        
        # Generate from W (forward will handle broadcasting if W+ is requested)
        return self.forward(w, latent_space=latent_space)

    def __repr__(self):
        return (
            f"StyleGAN2Wrapper(\n"
            f"  latent_dim={self.latent_dim},\n"
            f"  num_layers={self.num_layers},\n"
            f"  has_mapping={'Yes' if self.mapping_network else 'No'}\n"
            f")"
        )


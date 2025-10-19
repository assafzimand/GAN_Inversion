"""
StyleGAN2 model loader and utilities.

Provides abstractions for loading pretrained StyleGAN2 generators
and computing latent statistics (e.g., mean_w).

Supports common StyleGAN2 checkpoint formats:
- Official NVIDIA pkl format (via pickle)
- PyTorch .pt/.pth format (rosinality, etc.)
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import pickle
import io
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LegacyUnpickler(pickle.Unpickler):
    """
    Custom unpickler for loading NVIDIA StyleGAN2 .pkl files.
    
    NVIDIA's pickle files contain references to their dnnlib and torch_utils
    modules, which aren't available in our environment. This unpickler
    creates stub modules/classes to bypass those dependencies while
    extracting the actual PyTorch generator model.
    """
    
    def find_class(self, module, name):
        """
        Override find_class to handle missing NVIDIA modules.
        
        Args:
            module: Module name (e.g., 'dnnlib.tflib.network')
            name: Class name (e.g., 'Network')
        
        Returns:
            Class or stub class for missing modules
        """
        # Handle missing dnnlib modules
        if module.startswith('dnnlib'):
            # Create a stub module namespace
            import sys
            import types
            
            # Split module path
            parts = module.split('.')
            
            # Create nested module structure
            current_module = None
            for i, part in enumerate(parts):
                module_name = '.'.join(parts[:i+1])
                if module_name not in sys.modules:
                    sys.modules[module_name] = types.ModuleType(module_name)
                current_module = sys.modules[module_name]
            
            # Create a stub class if it doesn't exist
            if not hasattr(current_module, name):
                # Create a stub class that can hold PyTorch modules and is callable
                class StubNetwork:
                    """Stub class for NVIDIA's Network wrapper - callable for gradient flow"""
                    def __init__(self, *args, **kwargs):
                        pass
                    
                    def __setstate__(self, state):
                        # Restore state from pickle
                        self.__dict__.update(state)
                    
                    def __getstate__(self):
                        return self.__dict__
                    
                    def __call__(self, *args, **kwargs):
                        """
                        Make Network callable - delegates to components recursively.
                        This allows gradient flow through the PyTorch modules inside.
                        """
                        # If this is a synthesis/mapping network, delegate to its components
                        if hasattr(self, 'components') and isinstance(self.components, dict):
                            # First, try common high-level keys
                            for key in ['synthesis', 'mapping', 'main']:
                                if key in self.components:
                                    component = self.components[key]
                                    if callable(component):
                                        return component(*args, **kwargs)
                            
                            # If no known key, try to find any nn.Module or callable
                            # Check for actual PyTorch modules first (deepest level)
                            import torch.nn as nn
                            for key, component in self.components.items():
                                if isinstance(component, nn.Module):
                                    # Found actual PyTorch module - call it
                                    return component(*args, **kwargs)
                            
                            # Otherwise, try any callable component (might be nested Network)
                            for key, component in self.components.items():
                                if callable(component):
                                    return component(*args, **kwargs)
                        
                        # If no components, maybe it's already a callable module
                        raise RuntimeError(
                            f"Network wrapper is not properly callable. "
                            f"Attributes: {list(self.__dict__.keys()) if hasattr(self, '__dict__') else 'none'}, "
                            f"Components: {list(self.components.keys()) if hasattr(self, 'components') and isinstance(self.components, dict) else 'none'}"
                        )
                
                stub_class = type(name, (StubNetwork,), {})
                setattr(current_module, name, stub_class)
            
            return getattr(current_module, name)
        
        # Handle missing torch_utils modules (similar to dnnlib)
        if module.startswith('torch_utils'):
            import sys
            import types
            
            parts = module.split('.')
            current_module = None
            for i, part in enumerate(parts):
                module_name = '.'.join(parts[:i+1])
                if module_name not in sys.modules:
                    sys.modules[module_name] = types.ModuleType(module_name)
                current_module = sys.modules[module_name]
            
            if not hasattr(current_module, name):
                # Create a stub class that can hold PyTorch modules and is callable
                class StubNetwork:
                    """Stub class for torch_utils classes - callable for gradient flow"""
                    def __init__(self, *args, **kwargs):
                        pass
                    
                    def __setstate__(self, state):
                        self.__dict__.update(state)
                    
                    def __getstate__(self):
                        return self.__dict__
                    
                    def __call__(self, *args, **kwargs):
                        """Delegate to components if available - recursively handles nested Networks"""
                        if hasattr(self, 'components') and isinstance(self.components, dict):
                            # Try common high-level keys
                            for key in ['synthesis', 'mapping', 'main']:
                                if key in self.components:
                                    component = self.components[key]
                                    if callable(component):
                                        return component(*args, **kwargs)
                            
                            # Check for actual PyTorch modules first
                            import torch.nn as nn
                            for key, component in self.components.items():
                                if isinstance(component, nn.Module):
                                    return component(*args, **kwargs)
                            
                            # Try any callable component
                            for key, component in self.components.items():
                                if callable(component):
                                    return component(*args, **kwargs)
                        
                        raise RuntimeError(
                            f"torch_utils wrapper not callable. "
                            f"Attributes: {list(self.__dict__.keys()) if hasattr(self, '__dict__') else 'none'}, "
                            f"Components: {list(self.components.keys()) if hasattr(self, 'components') and isinstance(self.components, dict) else 'none'}"
                        )
                
                stub_class = type(name, (StubNetwork,), {})
                setattr(current_module, name, stub_class)
            
            return getattr(current_module, name)
        
        # For everything else, use the default behavior
        return super().find_class(module, name)


def load_pkl_legacy(weights_path: Path) -> nn.Module:
    """
    Load NVIDIA StyleGAN2 .pkl file using custom unpickler.
    
    Args:
        weights_path: Path to .pkl file
    
    Returns:
        Generator model (PyTorch nn.Module)
    
    Raises:
        RuntimeError: If loading fails
    """
    try:
        with open(weights_path, 'rb') as f:
            # Use custom unpickler
            unpickler = LegacyUnpickler(f)
            checkpoint = unpickler.load()
        
        # Extract generator from checkpoint structure
        if isinstance(checkpoint, tuple):
            # NVIDIA format: (G, D, Gs) where Gs is the EMA generator
            if len(checkpoint) >= 3:
                generator = checkpoint[2]  # Gs (EMA generator)
                logger.info("Extracted Gs (EMA generator) from tuple format")
            elif len(checkpoint) >= 1:
                generator = checkpoint[0]  # Fallback to G
                logger.info("Extracted G from tuple format")
            else:
                raise RuntimeError("Empty tuple in checkpoint")
        elif isinstance(checkpoint, dict):
            # Try common keys
            if 'G_ema' in checkpoint:
                generator = checkpoint['G_ema']
            elif 'G' in checkpoint:
                generator = checkpoint['G']
            elif 'generator' in checkpoint:
                generator = checkpoint['generator']
            else:
                # Assume the dict itself is the checkpoint
                generator = checkpoint
        else:
            # Direct model object
            generator = checkpoint
        
        # Handle NVIDIA Network wrapper
        # NEW APPROACH: Return the Network wrapper directly (it's now callable)
        # The StubNetwork class has __call__ that delegates to components
        if not isinstance(generator, nn.Module):
            # Check instance attributes
            inst_attrs = list(generator.__dict__.keys()) if hasattr(generator, '__dict__') else []
            logger.info(f"Network wrapper has attributes: {inst_attrs}")
            
            if hasattr(generator, 'components') and isinstance(generator.components, dict):
                components = generator.components
                logger.info(f"Found components: {list(components.keys())}")
                logger.info("Returning Network wrapper directly (now callable via __call__)")
                # Return the Network wrapper as-is - it's callable now
                # Gradients will flow through to the PyTorch modules in components
            else:
                raise RuntimeError(
                    f"Cannot use {type(generator).__name__} - no components dict found. "
                    f"Instance attributes: {inst_attrs}"
                )
        
        # COMMENTED OUT: Old extraction approach (kept for backup)
        # if not isinstance(generator, nn.Module):
        #     # Extract PyTorch modules from components dict
        #     if hasattr(generator, 'components') and isinstance(generator.components, dict):
        #         components = generator.components
        #         logger.info(f"Found components: {list(components.keys())}")
        #         
        #         # Create a simple wrapper to hold mapping and synthesis
        #         class ExtractedGenerator(nn.Module):
        #             def __init__(self, components_dict):
        #                 super().__init__()
        #                 # Extract mapping and synthesis networks from components
        #                 if 'mapping' in components_dict:
        #                     self.mapping = components_dict['mapping']
        #                 if 'synthesis' in components_dict:
        #                     self.synthesis = components_dict['synthesis']
        #             
        #             def forward(self, *args, **kwargs):
        #                 # Forward to synthesis
        #                 if hasattr(self, 'synthesis'):
        #                     return self.synthesis(*args, **kwargs)
        #                 else:
        #                     raise RuntimeError("No synthesis network found in components")
        #         
        #         generator = ExtractedGenerator(components)
        #         logger.info("Successfully extracted generator from Network components")
        
        logger.info("Successfully loaded legacy .pkl format")
        return generator
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load legacy .pkl file: {str(e)}\n"
            f"The file might be corrupted or in an unsupported format."
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
    - PyTorch state dict (.pt, .pth)
    - Official NVIDIA pickle format (.pkl)

    Args:
        weights_path: Path to pretrained weights
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
        >>> G = load_generator('checkpoints/stylegan2-ffhq.pt', device)
        >>> z = torch.randn(1, 512).to(device)
        >>> img = G(z, latent_space='W')
    """
    weights_path = Path(weights_path)
    
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}\n"
            f"Please download StyleGAN2-FFHQ weights and place in checkpoints/"
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
                
        elif suffix == '.pkl':
            # NVIDIA pickle format (use legacy unpickler)
            generator = load_pkl_legacy(weights_path)
        else:
            raise RuntimeError(
                f"Unsupported checkpoint format: {suffix}\n"
                f"Supported formats: .pt, .pth, .pkl"
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
        latent_dim: int = 512,
        num_layers: int = 18
    ):
        """Initialize wrapper with loaded generator."""
        super().__init__()
        self.generator = generator
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Detect generator structure
        self._detect_generator_structure()
        
        logger.debug(
            f"StyleGAN2Wrapper initialized: "
            f"latent_dim={latent_dim}, num_layers={num_layers}"
        )

    def _detect_generator_structure(self):
        """Detect the structure of the loaded generator."""
        # Try to identify mapping and synthesis networks
        
        # Check for Network wrapper with components dict (NVIDIA .pkl format)
        if hasattr(self.generator, 'components') and isinstance(self.generator.components, dict):
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
        # Check for standard PyTorch format (mapping and synthesis as attributes)
        elif hasattr(self.generator, 'mapping') and hasattr(self.generator, 'synthesis'):
            self.mapping_network = self.generator.mapping
            self.synthesis_network = self.generator.synthesis
            logger.info("Detected standard format")
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


"""
Core inversion engine for optimization-based GAN inversion.

Implements optimization-based latent code search to reconstruct
target images using a pretrained StyleGAN2 generator.
"""

from typing import Dict, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
from torch.optim import Adam
import logging
import time

# Import loss functions
from losses.l2 import L2Loss
from losses.lpips_loss import LPIPSLoss
from models.stylegan2_loader import compute_mean_w

logger = logging.getLogger(__name__)


def run_inversion(
    generator: nn.Module,
    target_image: torch.Tensor,
    config: Dict[str, Any],
    mean_w: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Run optimization-based inversion to find latent code.

    Minimizes loss(G(z), target) using Adam optimizer to find the
    optimal latent code z* that reconstructs the target image.

    Args:
        generator: StyleGAN2 generator (StyleGAN2Wrapper)
        target_image: Target image [1, 3, H, W] in [-1, 1]
        config: Inversion configuration dictionary
        mean_w: Precomputed mean W vector (optional, computed if needed)

    Returns:
        z_star: Optimized latent code [1, 512] for W or [1, num_layers, 512] for W+
        reconstruction: Generated image G(z_star) [1, 3, H, W]
        history: Dict with:
            - 'loss': List of loss values per step
            - 'steps': Total steps run
            - 'time': Total optimization time (seconds)
            - 'early_stopped': Whether early stopping triggered

    Raises:
        ValueError: If config is invalid or unsupported options specified

    Example:
        >>> from models.stylegan2_loader import load_generator
        >>> G = load_generator('checkpoints/stylegan2-ffhq.pt', device)
        >>> target = torch.randn(1, 3, 1024, 1024).to(device)
        >>> config = {'latent_space': 'W', 'init_method': 'mean_w', 'steps': 300}
        >>> z_star, recon, hist = run_inversion(G, target, config)
    """
    # Extract config parameters
    if isinstance(config, dict):
        cfg = config
    else:
        cfg = config.__dict__
    
    # Validate and extract parameters
    device = target_image.device
    latent_space = cfg.get('latent_space', 'W')
    init_method = cfg.get('init_method', 'mean_w')
    loss_type = cfg.get('loss_type', 'l2')
    steps = cfg.get('steps', 300)
    learning_rate = cfg.get('learning_rate', 0.01)
    log_interval = cfg.get('log_interval', 50)
    betas = cfg.get('betas', [0.9, 0.999])
    
    # Early stopping parameters
    enable_early_stop = cfg.get('enable_early_stop', False)
    early_stop_patience = cfg.get('early_stop_patience', 50)
    early_stop_threshold = cfg.get('early_stop_threshold', 1e-5)
    
    # Validate inputs
    if latent_space not in ['W', 'W+']:
        raise ValueError(f"latent_space must be 'W' or 'W+', got '{latent_space}'")
    if init_method not in ['mean_w', 'random']:
        raise ValueError(f"init_method must be 'mean_w' or 'random', got '{init_method}'")
    if loss_type not in ['l2', 'lpips']:
        raise ValueError(f"loss_type must be 'l2' or 'lpips', got '{loss_type}'")
    
    if target_image.dim() != 4:
        raise ValueError(f"target_image must be 4D [B, C, H, W], got shape {target_image.shape}")
    if target_image.shape[0] != 1:
        raise ValueError(f"target_image batch size must be 1, got {target_image.shape[0]}")
    
    logger.info(f"Starting inversion with {latent_space} space, {init_method} init, {loss_type} loss")
    logger.info(f"Steps: {steps}, LR: {learning_rate}")
    
    # Initialize latent code
    latent = initialize_latent(
        generator=generator,
        latent_space=latent_space,
        init_method=init_method,
        device=device,
        mean_w=mean_w,
        config=cfg
    )
    latent.requires_grad_(True)
    
    # Create loss function
    if loss_type == 'l2':
        loss_fn = L2Loss(reduction='mean')
    elif loss_type == 'lpips':
        lpips_net = cfg.get('lpips_net', 'alex')
        loss_fn = LPIPSLoss(net=lpips_net, device=device)
    
    # Create optimizer
    optimizer = Adam([latent], lr=learning_rate, betas=betas)
    
    # Optimization loop
    history = {
        'loss': [],
        'steps': 0,
        'time': 0.0,
        'early_stopped': False,
        'intermediates': {}  # Store intermediate reconstructions
    }
    
    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    # Save interval for intermediate reconstructions
    save_interval = cfg.get('save_interval', 100)
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Forward pass
        generated = generator(latent, latent_space=latent_space)
        
        # Compute loss
        loss = loss_fn(generated, target_image)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track history
        loss_value = loss.item()
        history['loss'].append(loss_value)
        
        # Save intermediate reconstruction every save_interval steps or at step 0
        if step % save_interval == 0 or step == 0:
            with torch.no_grad():
                intermediate = generator(latent, latent_space=latent_space)
                history['intermediates'][step] = intermediate.detach().cpu()
        
        # Logging
        if (step + 1) % log_interval == 0 or step == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Step [{step + 1}/{steps}] Loss: {loss_value:.6f} "
                f"Time: {elapsed:.2f}s"
            )
        
        # Early stopping check
        if enable_early_stop:
            if loss_value < best_loss - early_stop_threshold:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered at step {step + 1}")
                history['early_stopped'] = True
                break
    
    # Final metrics
    history['steps'] = step + 1
    history['time'] = time.time() - start_time
    
    # Generate final reconstruction
    with torch.no_grad():
        reconstruction = generator(latent, latent_space=latent_space)
        # Also save final reconstruction as intermediate
        if step not in history['intermediates']:
            history['intermediates'][step] = reconstruction.detach().cpu()
    
    logger.info(f"Inversion complete. Final loss: {history['loss'][-1]:.6f}")
    logger.info(f"Total time: {history['time']:.2f}s")
    logger.info(f"Saved {len(history['intermediates'])} intermediate reconstructions")
    
    return latent.detach(), reconstruction, history


def initialize_latent(
    generator: nn.Module,
    latent_space: str,
    init_method: str,
    device: torch.device,
    mean_w: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """
    Initialize latent code for optimization.

    For W space: returns [1, 512]
    For W+ space: returns [1, 512] which will be broadcast by StyleGAN2Wrapper

    Args:
        generator: StyleGAN2 generator (StyleGAN2Wrapper)
        latent_space: "W" or "W+"
        init_method: "mean_w" or "random"
        device: Target device
        mean_w: Precomputed mean W vector [1, 512] (optional)
        config: Additional config parameters for initialization

    Returns:
        Initialized latent code [1, 512]

    Raises:
        ValueError: If init_method is not supported or required params missing

    Example:
        >>> latent = initialize_latent(G, 'W', 'mean_w', device)
        >>> latent.shape
        torch.Size([1, 512])
    """
    if config is None:
        config = {}
    
    latent_dim = config.get('latent_dim', 512)
    
    if init_method == 'mean_w':
        # Use precomputed mean_w or compute it
        if mean_w is None:
            logger.info("Computing mean_w (not provided)")
            num_samples = config.get('mean_w_num_samples', 10000)
            batch_size = config.get('mean_w_batch_size', 1000)
            mean_w = compute_mean_w(
                generator,
                num_samples=num_samples,
                batch_size=batch_size,
                device=device
            )
        
        # Ensure mean_w is on the correct device and has batch dimension
        if mean_w.dim() == 1:
            mean_w = mean_w.unsqueeze(0)  # [512] -> [1, 512]
        
        latent = mean_w.clone().to(device)
        logger.info(f"Initialized latent with mean_w: shape {latent.shape}")
        
    elif init_method == 'random':
        # Random initialization from standard normal
        std = config.get('random_w_std', 1.0)
        mean = config.get('random_w_mean', 0.0)
        
        latent = torch.randn(1, latent_dim, device=device) * std + mean
        logger.info(f"Initialized latent randomly: shape {latent.shape}, std={std}, mean={mean}")
        
    else:
        raise ValueError(
            f"Unknown init_method: '{init_method}'. "
            f"Supported: 'mean_w', 'random'"
        )
    
    # For both W and W+, we return [1, 512]
    # The StyleGAN2Wrapper will handle broadcasting to [1, num_layers, 512] for W+
    return latent


"""
CLI entry point for optimization-based GAN inversion.

Usage:
    # Run with preset
    python invert.py --input data/face.png --preset combo_01
    
    # Run with custom settings
    python invert.py --input data/face.png --latent_space W+ --loss lpips --steps 600
    
    # Run on folder
    python invert.py --input data/samples/ --preset combo_02
"""

from typing import Optional, Dict, Any
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import yaml
import json
import torch

# Import project modules
from models.stylegan2_loader import load_generator, compute_mean_w
from engine.inverter import run_inversion
from engine.metrics import compute_all_metrics
from utils.seed import set_seed
from utils.image_io import load_image, save_image, load_images_from_folder
from utils.viz import create_comparison_panel, create_evolution_panel, plot_loss_curve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Optimization-based GAN Inversion using StyleGAN2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input image file or folder'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (defaults to outputs/run_<timestamp>)'
    )
    
    # Model
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to pretrained StyleGAN2 weights (defaults to config file)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=None,
        help='Image resolution (defaults to config file, currently 128×128)'
    )
    
    # Inversion Settings
    parser.add_argument(
        '--latent_space',
        type=str,
        choices=['W', 'W+'],
        default=None,
        help='Latent space for optimization (defaults to preset config)'
    )
    parser.add_argument(
        '--init_method',
        type=str,
        choices=['mean_w', 'random'],
        default=None,
        help='Initialization method (defaults to preset config)'
    )
    parser.add_argument(
        '--loss',
        type=str,
        choices=['l2', 'lpips'],
        default=None,
        help='Loss function (defaults to preset config)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Number of optimization steps (defaults to preset config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (defaults to preset config)'
    )
    
    # Experiment Preset
    parser.add_argument(
        '--preset',
        type=str,
        default=None,
        help='Load experiment preset from configs/experiment.yaml (e.g., combo_01, combo_02, combo_03)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base.yaml',
        help='Path to base config file'
    )
    
    # System
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=50,
        help='Logging frequency (steps)'
    )
    
    # Debugging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Load and merge configuration from files and CLI arguments.
    
    Priority: CLI args > preset config > base config
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Merged configuration dictionary
    """
    config = {}
    
    # Load base config
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded base config from {args.config}")
    else:
        logger.warning(f"Base config not found: {args.config}, using defaults")
    
    # Load preset if specified
    if args.preset:
        preset_path = Path('configs/experiment.yaml')
        if preset_path.exists():
            with open(preset_path, 'r') as f:
                presets = yaml.safe_load(f)
            
            if args.preset in presets:
                preset_config = presets[args.preset]
                config.update(preset_config)
                logger.info(f"Loaded preset '{args.preset}' from {preset_path}")
            else:
                available = ', '.join(presets.keys())
                raise ValueError(f"Preset '{args.preset}' not found. Available: {available}")
        else:
            raise FileNotFoundError(f"Preset config not found: {preset_path}")
    
    # Override with CLI arguments
    cli_overrides = {
        'input_path': args.input,
        'output_dir': args.output,
        'weights_path': args.weights,
        'image_size': args.image_size,
        'latent_space': args.latent_space,
        'init_method': args.init_method,
        'loss_type': args.loss,
        'steps': args.steps,
        'learning_rate': args.lr,
        'device': args.device,
        'seed': args.seed,
        'log_interval': args.log_interval,
    }
    
    # Only override if not using preset defaults
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value
    
    # Set default output dir if not specified or is generic "outputs"
    # (meaning no specific output dir was set)
    if config.get('output_dir') is None or config.get('output_dir') == 'outputs':
        # Extract combo name from experiment_name or preset
        combo_name = config.get('experiment_name', 'default')
        if args.preset:
            combo_name = args.preset
        
        # Get image name from input path (will be set later if folder)
        image_name = 'multi'
        if args.input and Path(args.input).is_file():
            image_name = Path(args.input).stem
        
        # Create unique directory name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config['output_dir'] = f"outputs/{combo_name}_{image_name}_{timestamp}"
    
    return config


def process_single_image(
    image_name: str,
    image_tensor: torch.Tensor,
    generator: torch.nn.Module,
    config: Dict[str, Any],
    output_dir: Path,
    mean_w: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Run inversion on a single image and save outputs.
    
    Args:
        image_name: Name of the image file
        image_tensor: Image tensor [1, 3, H, W]
        generator: StyleGAN2 generator
        config: Configuration dictionary
        output_dir: Output directory
        mean_w: Precomputed mean W (optional)
    
    Returns:
        Results dictionary with metrics and paths
    """
    logger.info(f"Processing: {image_name}")
    
    # Create subdirectories
    recon_dir = output_dir / 'reconstructions'
    diff_dir = output_dir / 'comparisons'
    recon_dir.mkdir(parents=True, exist_ok=True)
    diff_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inversion
    logger.info(f"Running inversion with {config['latent_space']} space, {config['loss_type']} loss...")
    z_star, reconstruction, history = run_inversion(
        generator=generator,
        target_image=image_tensor,
        config=config,
        mean_w=mean_w
    )
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_all_metrics(
        reconstruction,
        image_tensor,
        device=config['device']
    )
    
    # Save reconstruction
    stem = Path(image_name).stem
    recon_path = recon_dir / f"{stem}_recon.png"
    save_image(reconstruction, str(recon_path))
    logger.info(f"Saved reconstruction: {recon_path}")
    
    # Save evolution panel (showing progress over iterations with metrics)
    comp_path = diff_dir / f"{stem}_evolution.png"
    intermediates = history.get('intermediates', {})
    create_evolution_panel(image_tensor, intermediates, metrics, str(comp_path))
    logger.info(f"Saved evolution panel: {comp_path}")
    
    # Log metrics
    logger.info(
        f"Metrics - PSNR: {metrics['psnr']:.2f}, "
        f"SSIM: {metrics['ssim']:.4f}, "
        f"LPIPS: {metrics['lpips']:.4f}"
    )
    
    return {
        'image_name': image_name,
        'metrics': metrics,
        'history': history,
        'paths': {
            'reconstruction': str(recon_path),
            'comparison': str(comp_path)
        }
    }


def main():
    """Main CLI entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args)
    
    # Set random seed
    logger.info(f"Setting random seed: {config['seed']}")
    set_seed(config['seed'], deterministic=config.get('deterministic', True))
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Save config
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved config to: {config_path}")
    
    # Setup device
    device = torch.device(config['device'])
    logger.info(f"Using device: {device}")
    
    # Load generator
    logger.info(f"Loading StyleGAN2 generator from {config['weights_path']}...")
    generator = load_generator(
        weights_path=config['weights_path'],
        device=device,
        latent_dim=config.get('latent_dim', 512),
        num_layers=config.get('num_layers', 18)
    )
    generator.eval()
    logger.info("Generator loaded successfully")
    
    # Compute mean_w if needed
    mean_w = None
    if config['init_method'] == 'mean_w':
        logger.info("Computing mean W...")
        mean_w = compute_mean_w(
            generator,
            num_samples=config.get('mean_w_num_samples', 10000),
            batch_size=config.get('mean_w_batch_size', 1000),
            device=device
        )
        logger.info(f"Mean W computed: shape {mean_w.shape}")
    
    # Load input image(s)
    input_path = Path(config['input_path'])
    image_size = (config['image_size'], config['image_size'])
    
    if input_path.is_file():
        # Single image
        logger.info(f"Loading single image: {input_path}")
        image_tensor = load_image(str(input_path), size=image_size, device=device)
        images = [(input_path.name, image_tensor)]
    elif input_path.is_dir():
        # Folder of images
        logger.info(f"Loading images from folder: {input_path}")
        images = load_images_from_folder(str(input_path), size=image_size, device=device)
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")
    
    logger.info(f"Loaded {len(images)} image(s)")
    
    # Save original images
    orig_dir = output_dir / 'originals'
    orig_dir.mkdir(parents=True, exist_ok=True)
    for img_name, img_tensor in images:
        save_image(img_tensor, str(orig_dir / img_name))
    
    # Process all images
    all_results = []
    for img_name, img_tensor in images:
        try:
            result = process_single_image(
                image_name=img_name,
                image_tensor=img_tensor,
                generator=generator,
                config=config,
                output_dir=output_dir,
                mean_w=mean_w
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {img_name}: {e}")
            raise
    
    # Save aggregated metrics
    metrics_path = output_dir / 'metrics.json'
    metrics_summary = {
        res['image_name']: res['metrics']
        for res in all_results
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Saved metrics to: {metrics_path}")
    
    # Save loss curves for each image
    for result in all_results:
        stem = Path(result['image_name']).stem
        
        # Plot loss curve
        loss_curve_path = output_dir / f"{stem}_loss_curve.png"
        plot_loss_curve(
            result['history'],
            str(loss_curve_path),
            title=f"Loss Curve - {result['image_name']}"
        )
        
        # Save raw loss history (exclude intermediates - they're images, not JSON-serializable)
        loss_history_path = output_dir / f"{stem}_loss_history.json"
        history_for_json = {k: v for k, v in result['history'].items() if k != 'intermediates'}
        with open(loss_history_path, 'w') as f:
            json.dump(history_for_json, f, indent=2)
        
        logger.info(f"Saved loss outputs for {result['image_name']}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("INVERSION COMPLETE")
    logger.info("="*60)
    logger.info(f"Processed {len(all_results)} image(s)")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration: {config['latent_space']} space, {config['loss_type']} loss, {config['steps']} steps")
    
    # Average metrics
    if all_results:
        avg_psnr = sum(r['metrics']['psnr'] for r in all_results) / len(all_results)
        avg_ssim = sum(r['metrics']['ssim'] for r in all_results) / len(all_results)
        avg_lpips = sum(r['metrics']['lpips'] for r in all_results) / len(all_results)
        
        logger.info("\nAverage Metrics:")
        logger.info(f"  PSNR:  {avg_psnr:.2f}")
        logger.info(f"  SSIM:  {avg_ssim:.4f}")
        logger.info(f"  LPIPS: {avg_lpips:.4f}")
    
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


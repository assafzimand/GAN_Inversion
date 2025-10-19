"""
CLI entry point for optimization-based GAN inversion.

Usage:
    python invert.py --input path/to/image.png --output outputs/run_01

TODO:
    - Parse CLI arguments (input, output, config overrides, seed, device)
    - Load config from YAML
    - Set global seed
    - Load pretrained StyleGAN2 generator
    - Load and preprocess input image(s)
    - Run inversion via engine.inverter
    - Save outputs: reconstructions, diffs, metrics, loss curves
    - Handle errors gracefully with non-zero exit codes
"""

from typing import Optional
import argparse
import sys


def main():
    """Main CLI entry point."""
    # TODO: Implement argument parsing
    # TODO: Implement config loading and merging
    # TODO: Implement seeding
    # TODO: Implement inversion pipeline
    # TODO: Implement output saving
    pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


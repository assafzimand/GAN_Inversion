"""
Data preparation script for organizing sample images.

This script can:
1. Copy images from a source folder to data/samples/
2. Generate synthetic test images for testing the pipeline
3. Provide instructions for downloading real FFHQ/CelebA-HQ images

Usage:
    # Generate synthetic test images
    python scripts/prepare_data.py --mode generate --count 5
    
    # Copy images from a folder
    python scripts/prepare_data.py --mode copy --source /path/to/images
    
    # Show download instructions
    python scripts/prepare_data.py --mode instructions
"""

import argparse
import sys
from pathlib import Path
import shutil
from typing import List

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def create_synthetic_image(
    size: int = 1024,
    index: int = 0,
    pattern: str = 'gradient'
) -> 'Image.Image':
    """
    Create a synthetic test image with distinct patterns.
    
    Args:
        size: Image size (square)
        index: Image index for variation
        pattern: Pattern type ('gradient', 'checkerboard', 'solid')
    
    Returns:
        PIL Image
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL/Pillow is required. Install with: pip install pillow")
    
    img = Image.new('RGB', (size, size))
    draw = ImageDraw.Draw(img)
    
    # Color schemes for variation
    colors = [
        ((255, 100, 100), (100, 100, 255)),  # Red to Blue
        ((100, 255, 100), (255, 255, 100)),  # Green to Yellow
        ((255, 150, 100), (150, 100, 255)),  # Orange to Purple
        ((100, 255, 255), (255, 100, 255)),  # Cyan to Magenta
        ((255, 200, 100), (100, 200, 255)),  # Gold to Sky Blue
    ]
    
    color_pair = colors[index % len(colors)]
    
    if pattern == 'gradient':
        # Create gradient
        for y in range(size):
            ratio = y / size
            r = int(color_pair[0][0] * (1 - ratio) + color_pair[1][0] * ratio)
            g = int(color_pair[0][1] * (1 - ratio) + color_pair[1][1] * ratio)
            b = int(color_pair[0][2] * (1 - ratio) + color_pair[1][2] * ratio)
            draw.line([(0, y), (size, y)], fill=(r, g, b))
    
    elif pattern == 'checkerboard':
        # Create checkerboard pattern
        square_size = size // 8
        for i in range(8):
            for j in range(8):
                color = color_pair[(i + j) % 2]
                x1, y1 = i * square_size, j * square_size
                x2, y2 = x1 + square_size, y1 + square_size
                draw.rectangle([x1, y1, x2, y2], fill=color)
    
    elif pattern == 'solid':
        # Solid color with border
        draw.rectangle([0, 0, size, size], fill=color_pair[0])
        border_width = size // 20
        draw.rectangle(
            [border_width, border_width, size - border_width, size - border_width],
            outline=color_pair[1],
            width=border_width
        )
    
    # Add text label
    try:
        font_size = size // 20
        # Try to use default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        text = f"Test Image {index + 1}"
        # Get text bbox for centering
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (size - text_width) // 2
        y = size // 2 - text_height // 2
        
        # Draw text with shadow for visibility
        draw.text((x + 2, y + 2), text, fill=(0, 0, 0), font=font)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
    except:
        pass  # Skip text if font handling fails
    
    return img


def generate_synthetic_images(output_dir: Path, count: int = 5):
    """
    Generate synthetic test images.
    
    Args:
        output_dir: Output directory
        count: Number of images to generate
    """
    if not PIL_AVAILABLE:
        print("Error: PIL/Pillow is required for image generation")
        print("Install with: pip install pillow")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patterns = ['gradient', 'checkerboard', 'solid']
    
    print(f"Generating {count} synthetic test images...")
    for i in range(count):
        pattern = patterns[i % len(patterns)]
        img = create_synthetic_image(size=1024, index=i, pattern=pattern)
        
        filename = f"test_image_{i + 1:02d}.png"
        filepath = output_dir / filename
        img.save(filepath)
        print(f"  ✓ Created: {filepath}")
    
    print(f"\n✓ Generated {count} test images in {output_dir}")


def copy_images_from_folder(source_dir: Path, output_dir: Path):
    """
    Copy images from source folder to data/samples.
    
    Args:
        source_dir: Source directory with images
        output_dir: Output directory
    """
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Valid image extensions
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    
    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(source_dir.glob(f'*{ext}'))
        image_files.extend(source_dir.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"Error: No images found in {source_dir}")
        print(f"Looking for extensions: {', '.join(extensions)}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images in {source_dir}")
    print(f"Copying to {output_dir}...")
    
    copied = 0
    for img_file in image_files:
        dest_path = output_dir / img_file.name
        shutil.copy2(img_file, dest_path)
        print(f"  ✓ Copied: {img_file.name}")
        copied += 1
    
    print(f"\n✓ Copied {copied} images to {output_dir}")


def show_download_instructions():
    """Show instructions for downloading real face images."""
    print("="*70)
    print("SAMPLE DATA DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print()
    print("For realistic face images, you can download from:")
    print()
    print("Option 1: FFHQ Dataset (Flickr-Faces-HQ)")
    print("  • Official: https://github.com/NVlabs/ffhq-dataset")
    print("  • Kaggle: https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq")
    print("  • Note: Full dataset is large (70GB+), download only thumbnails or samples")
    print()
    print("Option 2: CelebA-HQ")
    print("  • Official: https://github.com/tkarras/progressive_growing_of_gans")
    print("  • Kaggle: https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256")
    print()
    print("Option 3: Use Your Own Images")
    print("  • Take/collect 3-5 face images (front-facing, well-lit)")
    print("  • Crop to roughly square aspect ratio")
    print("  • Place in data/samples/")
    print()
    print("Quick Setup:")
    print("  1. Download 3-5 sample images (any resolution)")
    print("  2. Place them in: data/samples/")
    print("  3. The inversion script will automatically resize them to 1024x1024")
    print()
    print("Alternative - Generate Test Images:")
    print("  python scripts/prepare_data.py --mode generate --count 5")
    print()
    print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Prepare sample data for GAN inversion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5 synthetic test images
  python scripts/prepare_data.py --mode generate --count 5
  
  # Copy images from a folder
  python scripts/prepare_data.py --mode copy --source ~/Pictures/faces
  
  # Show download instructions for real face images
  python scripts/prepare_data.py --mode instructions
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['generate', 'copy', 'instructions'],
        default='generate',
        help='Preparation mode'
    )
    parser.add_argument(
        '--source',
        type=str,
        default=None,
        help='Source directory for copy mode'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/samples',
        help='Output directory (default: data/samples)'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=5,
        help='Number of images to generate (generate mode only)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    if args.mode == 'generate':
        generate_synthetic_images(output_dir, args.count)
    
    elif args.mode == 'copy':
        if not args.source:
            print("Error: --source is required for copy mode")
            sys.exit(1)
        source_dir = Path(args.source)
        copy_images_from_folder(source_dir, output_dir)
    
    elif args.mode == 'instructions':
        show_download_instructions()
    
    # Show summary
    if args.mode != 'instructions':
        print()
        print("Next steps:")
        print("  1. Verify images: ls -lh data/samples/")
        print("  2. Run inversion: python invert.py --input data/samples/ --preset combo_01")


if __name__ == "__main__":
    main()


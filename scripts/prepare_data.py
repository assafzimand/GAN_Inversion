"""
Data preparation script for downloading and organizing sample images.

TODO:
    - Download sample FFHQ/CelebA-HQ images (3-5 images)
    - Place in data/samples/
    - Validate image formats and sizes
    - Create data manifest (optional)
"""

from pathlib import Path


def prepare_data():
    """
    Download and organize sample data.

    TODO: Implement data preparation
    """
    print("TODO: Implement prepare_data()")
    print("This script should download 3-5 sample face images to data/samples/")
    print("For now, please manually add sample images to data/samples/")
    
    # Create data directory structure
    data_dir = Path("data/samples")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory: {data_dir}")


if __name__ == "__main__":
    prepare_data()


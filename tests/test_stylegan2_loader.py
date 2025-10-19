"""
Unit tests for StyleGAN2 loader.

Tests the wrapper functionality without requiring actual pretrained weights.
Uses mock generators to verify structure and behavior.
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.stylegan2_loader import (
    StyleGAN2Wrapper,
    compute_mean_w,
    load_generator
)


class MockMappingNetwork(nn.Module):
    """Mock mapping network for testing."""
    
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, z):
        return self.fc(z)


class MockSynthesisNetwork(nn.Module):
    """Mock synthesis network for testing."""
    
    def __init__(self, latent_dim=512, num_layers=18, img_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.img_size = img_size
        
        # Simple conv layer to generate images
        self.conv = nn.Conv2d(latent_dim, 3, 1)
    
    def forward(self, latents):
        # latents: [B, num_layers, latent_dim]
        # For simplicity, use first layer latent
        batch_size = latents.shape[0]
        
        # Create a spatial map from latents
        w = latents[:, 0, :]  # [B, latent_dim]
        spatial = w.view(batch_size, self.latent_dim, 1, 1)
        spatial = spatial.expand(-1, -1, self.img_size, self.img_size)
        
        # Generate image
        img = self.conv(spatial)
        return torch.tanh(img)  # [-1, 1] range


class MockGenerator(nn.Module):
    """Mock StyleGAN2 generator for testing."""
    
    def __init__(self, latent_dim=512, num_layers=18, img_size=256):
        super().__init__()
        self.mapping = MockMappingNetwork(latent_dim)
        self.synthesis = MockSynthesisNetwork(latent_dim, num_layers, img_size)
        self.latent_dim = latent_dim
        self.num_layers = num_layers
    
    def forward(self, latents):
        return self.synthesis(latents)


class TestStyleGAN2Wrapper:
    """Test cases for StyleGAN2Wrapper."""
    
    def test_wrapper_initialization(self):
        """Test wrapper initializes correctly."""
        generator = MockGenerator()
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        assert wrapper.latent_dim == 512
        assert wrapper.num_layers == 18
        assert wrapper.mapping_network is not None
        assert wrapper.synthesis_network is not None
    
    def test_w_space_forward(self):
        """Test forward pass with W space latent."""
        generator = MockGenerator(latent_dim=512, num_layers=18, img_size=64)
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        # W space: [B, latent_dim]
        w = torch.randn(2, 512)
        img = wrapper(w, latent_space='W')
        
        assert img.shape == (2, 3, 64, 64)
        assert img.min() >= -1.0 and img.max() <= 1.0
    
    def test_w_plus_space_forward(self):
        """Test forward pass with W+ space latent."""
        generator = MockGenerator(latent_dim=512, num_layers=18, img_size=64)
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        # W+ space: [B, num_layers, latent_dim]
        w_plus = torch.randn(2, 18, 512)
        img = wrapper(w_plus, latent_space='W+')
        
        assert img.shape == (2, 3, 64, 64)
        assert img.min() >= -1.0 and img.max() <= 1.0
    
    def test_w_space_shape_validation(self):
        """Test W space shape validation."""
        generator = MockGenerator()
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        # Wrong number of dimensions
        with pytest.raises(ValueError, match="W space latent must be 2D"):
            wrapper(torch.randn(2, 18, 512), latent_space='W')
        
        # Wrong latent dim
        with pytest.raises(ValueError, match="Expected latent_dim=512"):
            wrapper(torch.randn(2, 256), latent_space='W')
    
    def test_w_plus_space_shape_validation(self):
        """Test W+ space shape validation."""
        generator = MockGenerator()
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        # Wrong number of dimensions (1D or 4D+)
        with pytest.raises(ValueError, match="W\\+ space latent must be 2D"):
            wrapper(torch.randn(512), latent_space='W+')
        
        # Wrong number of layers (for 3D input)
        with pytest.raises(ValueError, match="Expected 18 layers"):
            wrapper(torch.randn(2, 14, 512), latent_space='W+')
        
        # Wrong latent dim (for 3D input)
        with pytest.raises(ValueError, match="Expected latent_dim=512"):
            wrapper(torch.randn(2, 18, 256), latent_space='W+')
        
        # Wrong latent dim (for 2D input)
        with pytest.raises(ValueError, match="Expected latent_dim=512"):
            wrapper(torch.randn(2, 256), latent_space='W+')
    
    def test_invalid_latent_space(self):
        """Test error for invalid latent space."""
        generator = MockGenerator()
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        with pytest.raises(ValueError, match="latent_space must be"):
            wrapper(torch.randn(2, 512), latent_space='Z')
    
    def test_return_latents(self):
        """Test returning latents along with image."""
        generator = MockGenerator(latent_dim=512, num_layers=18, img_size=64)
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        w = torch.randn(2, 512)
        img, latents = wrapper(w, latent_space='W', return_latents=True)
        
        assert img.shape == (2, 3, 64, 64)
        assert latents.shape == (2, 18, 512)  # Expanded to W+
    
    def test_map_to_w(self):
        """Test Z to W mapping."""
        generator = MockGenerator()
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        z = torch.randn(4, 512)
        w = wrapper.map_to_w(z)
        
        assert w.shape == (4, 512)
    
    def test_generate_from_z(self):
        """Test direct generation from Z space."""
        generator = MockGenerator(latent_dim=512, num_layers=18, img_size=64)
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        z = torch.randn(2, 512)
        
        # Test W space (forward will broadcast)
        img_w = wrapper.generate_from_z(z, latent_space='W')
        assert img_w.shape == (2, 3, 64, 64)
        
        # Test W+ space (forward will broadcast W to W+)
        img_wplus = wrapper.generate_from_z(z, latent_space='W+')
        assert img_wplus.shape == (2, 3, 64, 64)
    
    def test_device_handling(self):
        """Test that wrapper works with different devices."""
        generator = MockGenerator(latent_dim=512, num_layers=18, img_size=64)
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        # CPU
        w_cpu = torch.randn(1, 512)
        img_cpu = wrapper(w_cpu, latent_space='W')
        assert img_cpu.device.type == 'cpu'
        
        # GPU (if available)
        if torch.cuda.is_available():
            wrapper_gpu = wrapper.cuda()
            w_gpu = torch.randn(1, 512).cuda()
            img_gpu = wrapper_gpu(w_gpu, latent_space='W')
            assert img_gpu.device.type == 'cuda'
    
    def test_repr(self):
        """Test string representation."""
        generator = MockGenerator()
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        repr_str = repr(wrapper)
        assert 'StyleGAN2Wrapper' in repr_str
        assert 'latent_dim=512' in repr_str
        assert 'num_layers=18' in repr_str


class TestComputeMeanW:
    """Test cases for compute_mean_w function."""
    
    def test_compute_mean_w_shape(self):
        """Test mean_w has correct shape."""
        generator = MockGenerator()
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        mean_w = compute_mean_w(wrapper, num_samples=100, batch_size=50)
        
        assert mean_w.shape == (1, 512)
    
    def test_compute_mean_w_reproducibility(self):
        """Test mean_w is reproducible with seed."""
        generator = MockGenerator()
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        mean_w1 = compute_mean_w(wrapper, num_samples=100, batch_size=50, seed=42)
        mean_w2 = compute_mean_w(wrapper, num_samples=100, batch_size=50, seed=42)
        
        assert torch.allclose(mean_w1, mean_w2, atol=1e-6)
    
    def test_compute_mean_w_different_seeds(self):
        """Test different seeds give different results."""
        generator = MockGenerator()
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        mean_w1 = compute_mean_w(wrapper, num_samples=100, batch_size=50, seed=42)
        mean_w2 = compute_mean_w(wrapper, num_samples=100, batch_size=50, seed=123)
        
        assert not torch.allclose(mean_w1, mean_w2, atol=1e-2)
    
    def test_compute_mean_w_device(self):
        """Test mean_w computation on different devices."""
        generator = MockGenerator()
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        # CPU
        mean_w_cpu = compute_mean_w(
            wrapper, num_samples=100, batch_size=50, device=torch.device('cpu')
        )
        assert mean_w_cpu.device.type == 'cpu'
        
        # GPU (if available)
        if torch.cuda.is_available():
            wrapper_gpu = wrapper.cuda()
            mean_w_gpu = compute_mean_w(
                wrapper_gpu, num_samples=100, batch_size=50, device=torch.device('cuda')
            )
            assert mean_w_gpu.device.type == 'cuda'
    
    def test_compute_mean_w_batch_sizes(self):
        """Test with different batch sizes."""
        generator = MockGenerator()
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        # Same seed, different batch sizes should give same result
        mean_w1 = compute_mean_w(wrapper, num_samples=100, batch_size=25, seed=42)
        mean_w2 = compute_mean_w(wrapper, num_samples=100, batch_size=50, seed=42)
        
        # Should be approximately equal (minor differences due to batching)
        assert torch.allclose(mean_w1, mean_w2, atol=1e-5)


class TestLoadGenerator:
    """Test cases for load_generator function."""
    
    def test_file_not_found(self):
        """Test error when weights file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Weights file not found"):
            load_generator('nonexistent.pt', torch.device('cpu'))
    
    def test_unsupported_format(self):
        """Test error for unsupported file format."""
        # Create a temporary file with unsupported extension
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(RuntimeError, match="Unsupported checkpoint format"):
                load_generator(temp_path, torch.device('cpu'))
        finally:
            Path(temp_path).unlink()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_batch_size_one(self):
        """Test with batch size of 1."""
        generator = MockGenerator(latent_dim=512, num_layers=18, img_size=64)
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        w = torch.randn(1, 512)
        img = wrapper(w, latent_space='W')
        
        assert img.shape == (1, 3, 64, 64)
    
    def test_large_batch(self):
        """Test with larger batch size."""
        generator = MockGenerator(latent_dim=512, num_layers=18, img_size=64)
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        w = torch.randn(16, 512)
        img = wrapper(w, latent_space='W')
        
        assert img.shape == (16, 3, 64, 64)
    
    def test_gradient_flow(self):
        """Test gradients flow through wrapper."""
        generator = MockGenerator(latent_dim=512, num_layers=18, img_size=64)
        wrapper = StyleGAN2Wrapper(generator, latent_dim=512, num_layers=18)
        
        w = torch.randn(1, 512, requires_grad=True)
        img = wrapper(w, latent_space='W')
        loss = img.mean()
        loss.backward()
        
        assert w.grad is not None
        assert w.grad.shape == w.shape


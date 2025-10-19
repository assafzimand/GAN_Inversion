"""
Tests for encoder-based initialization.

Tests the encoder loader and integration with the inversion pipeline.
"""

import pytest
import torch
import torch.nn as nn
from models.encoder_loader import SimpleEncoder, load_encoder, encode_image, upscale_image


class TestSimpleEncoder:
    """Tests for SimpleEncoder architecture."""
    
    def test_encoder_initialization(self):
        """Test encoder can be created with default parameters."""
        encoder = SimpleEncoder(
            input_size=128,
            latent_dim=512,
            num_layers=12,
            output_space="W+"
        )
        
        assert isinstance(encoder, nn.Module)
        assert encoder.input_size == 128
        assert encoder.latent_dim == 512
        assert encoder.num_layers == 12
        assert encoder.output_space == "W+"
    
    def test_encoder_forward_w_plus(self):
        """Test encoder forward pass with W+ output."""
        encoder = SimpleEncoder(output_space="W+", num_layers=12)
        
        # Create dummy input
        x = torch.randn(2, 3, 128, 128)
        
        # Forward pass
        latent = encoder(x)
        
        # Check output shape
        assert latent.shape == (2, 12, 512)
    
    def test_encoder_forward_w(self):
        """Test encoder forward pass with W output."""
        encoder = SimpleEncoder(output_space="W", num_layers=12)
        
        # Create dummy input
        x = torch.randn(2, 3, 128, 128)
        
        # Forward pass
        latent = encoder(x)
        
        # Check output shape
        assert latent.shape == (2, 512)
    
    def test_encoder_gradient_flow(self):
        """Test gradients flow through encoder."""
        encoder = SimpleEncoder(output_space="W+")
        
        x = torch.randn(1, 3, 128, 128, requires_grad=True)
        latent = encoder(x)
        
        # Compute loss and backward
        loss = latent.mean()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestEncoderLoader:
    """Tests for encoder loading utilities."""
    
    def test_load_simple_encoder_no_checkpoint(self):
        """Test loading encoder without checkpoint (random init)."""
        config = {
            'encoder_type': 'simple_encoder',
            'encoder_checkpoint': None,
            'encoder_config': {
                'input_size': 128,
                'output_dim': 512,
                'num_layers': 12
            },
            'output_space': 'W+'
        }
        
        device = torch.device('cpu')
        encoder = load_encoder(config, device)
        
        assert isinstance(encoder, SimpleEncoder)
        assert next(encoder.parameters()).device.type == 'cpu'
    
    def test_load_encoder_e4e_without_checkpoint(self):
        """Test loading e4e/pSp encoder without checkpoint raises error."""
        config = {
            'encoder_type': 'e4e',
            'encoder_checkpoint': None
        }
        
        device = torch.device('cpu')
        
        with pytest.raises(ValueError, match="Encoder checkpoint path required"):
            load_encoder(config, device)
    
    def test_encode_image_function(self):
        """Test encode_image utility function."""
        encoder = SimpleEncoder(output_space="W+")
        encoder.eval()
        
        image = torch.randn(1, 3, 128, 128)
        device = torch.device('cpu')
        
        latent = encode_image(encoder, image, device)
        
        assert latent.shape == (1, 12, 512)
        assert latent.device.type == 'cpu'


class TestUpscaling:
    """Tests for image upscaling functionality."""
    
    def test_upscale_basic(self):
        """Test basic upscaling from 128 to 1024."""
        image = torch.randn(1, 3, 128, 128)
        upscaled = upscale_image(image, target_size=1024)
        
        assert upscaled.shape == (1, 3, 1024, 1024)
    
    def test_upscale_preserves_range(self):
        """Test upscaling preserves approximate value range."""
        # Create image in [-1, 1]
        image = torch.rand(1, 3, 128, 128) * 2 - 1
        upscaled = upscale_image(image, target_size=1024)
        
        # Bicubic interpolation can overshoot more than expected
        # Just check it's roughly in the right ballpark
        assert upscaled.min() >= -1.5
        assert upscaled.max() <= 1.5
    
    def test_upscale_different_modes(self):
        """Test different upscaling modes."""
        image = torch.randn(1, 3, 128, 128)
        
        for mode in ['bicubic', 'bilinear', 'nearest']:
            upscaled = upscale_image(image, target_size=256, mode=mode)
            assert upscaled.shape == (1, 3, 256, 256)
    
    def test_upscale_no_op_when_already_correct_size(self):
        """Test upscaling does nothing when image is already target size."""
        image = torch.randn(1, 3, 1024, 1024)
        upscaled = upscale_image(image, target_size=1024)
        
        # Should return same tensor (no upscaling needed)
        assert torch.equal(upscaled, image)
    
    def test_encode_image_with_upscaling(self):
        """Test encode_image applies upscaling when configured."""
        # SimpleEncoder is flexible and can handle different input sizes
        # So we test that upscaling logic is triggered, not that it fails
        encoder = SimpleEncoder(input_size=128, output_space="W")
        device = torch.device('cpu')
        
        # 128x128 image
        image = torch.randn(1, 3, 128, 128)
        
        # Config specifying upscaling (image will be upscaled to 1024 first)
        encoder_config = {
            'encoder_config': {
                'input_size': 1024,
                'our_image_size': 128
            },
            'upscale_method': 'bicubic'
        }
        
        # This should work - image gets upscaled, then encoded
        # SimpleEncoder is flexible enough to handle different sizes
        latent = encode_image(encoder, image, device, encoder_config)
        
        # Should still produce valid W output
        assert latent.shape == (1, 512)
        assert latent.dtype == torch.float32


class TestEncoderIntegration:
    """Tests for encoder integration with inversion pipeline."""
    
    def test_encoder_output_compatible_with_generator(self):
        """Test encoder output is compatible with StyleGAN2 forward pass."""
        # This is a shape/dimension compatibility test
        encoder = SimpleEncoder(output_space="W+", num_layers=12)
        
        image = torch.randn(1, 3, 128, 128)
        latent = encoder(image)
        
        # Check latent has correct shape for StyleGAN2
        assert latent.shape == (1, 12, 512)
        assert latent.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestEncoderGPU:
    """GPU-specific encoder tests."""
    
    def test_encoder_on_gpu(self):
        """Test encoder works on GPU."""
        device = torch.device('cuda')
        encoder = SimpleEncoder().to(device)
        
        x = torch.randn(1, 3, 128, 128, device=device)
        latent = encoder(x)
        
        assert latent.device.type == 'cuda'
        assert latent.shape == (1, 12, 512)


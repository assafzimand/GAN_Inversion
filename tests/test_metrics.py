"""
Unit tests for metrics computation.

Tests PSNR, SSIM, and LPIPS metrics with known values and edge cases.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.metrics import (
    compute_psnr,
    compute_ssim,
    compute_lpips,
    compute_all_metrics,
    LPIPS_AVAILABLE
)


class TestPSNR:
    """Test cases for PSNR metric."""

    def test_identical_images(self):
        """PSNR should be inf for identical images."""
        img = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        psnr = compute_psnr(img, img)
        assert psnr == float('inf'), "PSNR should be infinite for identical images"

    def test_known_values(self):
        """Test PSNR with known MSE values."""
        # Create images with known difference
        img1 = torch.zeros(1, 3, 10, 10)  # All zeros
        img2 = torch.ones(1, 3, 10, 10) * 0.1  # All 0.1

        # MSE = (0.1)^2 = 0.01
        # PSNR = 10 * log10(4 / 0.01) = 10 * log10(400) ≈ 26.02 dB
        psnr = compute_psnr(img1, img2, data_range=2.0)
        expected_psnr = 10 * np.log10(4.0 / 0.01)
        assert abs(psnr - expected_psnr) < 0.01, f"Expected {expected_psnr:.2f}, got {psnr:.2f}"

    def test_3d_input(self):
        """Should handle 3D [C, H, W] inputs."""
        img1 = torch.randn(3, 32, 32).clamp(-1, 1)
        img2 = torch.randn(3, 32, 32).clamp(-1, 1)
        psnr = compute_psnr(img1, img2)
        assert isinstance(psnr, float), "PSNR should return float"
        assert psnr > 0, "PSNR should be positive for different images"

    def test_4d_input(self):
        """Should handle 4D [B, C, H, W] inputs."""
        img1 = torch.randn(2, 3, 32, 32).clamp(-1, 1)
        img2 = torch.randn(2, 3, 32, 32).clamp(-1, 1)
        psnr = compute_psnr(img1, img2)
        assert isinstance(psnr, float), "PSNR should return float"
        assert psnr > 0, "PSNR should be positive"

    def test_shape_mismatch(self):
        """Should raise error for mismatched shapes."""
        img1 = torch.randn(1, 3, 32, 32)
        img2 = torch.randn(1, 3, 64, 64)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_psnr(img1, img2)

    def test_different_data_ranges(self):
        """PSNR should scale with data range."""
        img1 = torch.zeros(1, 3, 10, 10)
        img2 = torch.ones(1, 3, 10, 10) * 0.1

        psnr_range2 = compute_psnr(img1, img2, data_range=2.0)
        psnr_range1 = compute_psnr(img1, img2, data_range=1.0)

        # Larger data range should give higher PSNR
        assert psnr_range2 > psnr_range1, "PSNR should increase with data range"

    def test_noise_levels(self):
        """Higher noise should give lower PSNR."""
        img = torch.zeros(1, 3, 64, 64)
        
        # Small noise
        img_small_noise = img + torch.randn_like(img) * 0.01
        psnr_small = compute_psnr(img, img_small_noise)
        
        # Large noise
        img_large_noise = img + torch.randn_like(img) * 0.1
        psnr_large = compute_psnr(img, img_large_noise)
        
        assert psnr_small > psnr_large, "Lower noise should give higher PSNR"


class TestSSIM:
    """Test cases for SSIM metric."""

    def test_identical_images(self):
        """SSIM should be 1.0 for identical images."""
        img = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        ssim = compute_ssim(img, img)
        assert abs(ssim - 1.0) < 1e-5, f"SSIM should be 1.0 for identical images, got {ssim}"

    def test_range(self):
        """SSIM should be in appropriate range."""
        img1 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        ssim = compute_ssim(img1, img2)
        # SSIM can be negative for very different images
        assert -1 <= ssim <= 1, f"SSIM should be in [-1, 1], got {ssim}"

    def test_3d_input(self):
        """Should handle 3D [C, H, W] inputs."""
        img1 = torch.randn(3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(3, 64, 64).clamp(-1, 1)
        ssim = compute_ssim(img1, img2)
        assert isinstance(ssim, float), "SSIM should return float"

    def test_4d_input(self):
        """Should handle 4D [B, C, H, W] inputs."""
        img1 = torch.randn(2, 3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(2, 3, 64, 64).clamp(-1, 1)
        ssim = compute_ssim(img1, img2)
        assert isinstance(ssim, float), "SSIM should return float"

    def test_shape_mismatch(self):
        """Should raise error for mismatched shapes."""
        img1 = torch.randn(1, 3, 32, 32)
        img2 = torch.randn(1, 3, 64, 64)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_ssim(img1, img2)

    def test_structural_similarity(self):
        """Structurally similar images should have higher SSIM."""
        # Create base image
        base = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        
        # Slightly shifted version (structurally similar)
        similar = base + torch.randn_like(base) * 0.01
        
        # Random image (structurally different)
        different = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        
        ssim_similar = compute_ssim(base, similar)
        ssim_different = compute_ssim(base, different)
        
        assert ssim_similar > ssim_different, "Structurally similar images should have higher SSIM"

    def test_uniform_images(self):
        """Uniform images should have SSIM = 1.0."""
        img1 = torch.ones(1, 3, 64, 64) * 0.5
        img2 = torch.ones(1, 3, 64, 64) * 0.5
        ssim = compute_ssim(img1, img2)
        assert abs(ssim - 1.0) < 1e-5, "Identical uniform images should have SSIM = 1.0"


@pytest.mark.skipif(not LPIPS_AVAILABLE, reason="lpips package not installed")
class TestLPIPSMetric:
    """Test cases for LPIPS metric."""

    def test_identical_images(self):
        """LPIPS should be near 0 for identical images."""
        img = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        lpips = compute_lpips(img, img, device=torch.device("cpu"))
        assert lpips < 1e-5, f"LPIPS should be near 0 for identical images, got {lpips}"

    def test_different_images(self):
        """LPIPS should be positive for different images."""
        img1 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        lpips = compute_lpips(img1, img2, device=torch.device("cpu"))
        assert lpips > 0, "LPIPS should be positive for different images"

    def test_3d_input(self):
        """Should handle 3D [C, H, W] inputs."""
        img1 = torch.randn(3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(3, 64, 64).clamp(-1, 1)
        lpips = compute_lpips(img1, img2, device=torch.device("cpu"))
        assert isinstance(lpips, float), "LPIPS should return float"
        assert lpips >= 0, "LPIPS should be non-negative"

    def test_4d_input(self):
        """Should handle 4D [B, C, H, W] inputs."""
        img1 = torch.randn(2, 3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(2, 3, 64, 64).clamp(-1, 1)
        lpips = compute_lpips(img1, img2, device=torch.device("cpu"))
        assert isinstance(lpips, float), "LPIPS should return float"

    def test_shape_mismatch(self):
        """Should raise error for mismatched shapes."""
        img1 = torch.randn(1, 3, 32, 32)
        img2 = torch.randn(1, 3, 64, 64)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_lpips(img1, img2, device=torch.device("cpu"))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_execution(self):
        """Should work on GPU."""
        img1 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        lpips = compute_lpips(img1, img2, device=torch.device("cuda"))
        assert isinstance(lpips, float), "LPIPS should return float"
        assert lpips >= 0, "LPIPS should be non-negative"

    def test_perceptual_similarity(self):
        """Perceptually similar images should have lower LPIPS."""
        base = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        
        # Small perturbation (perceptually similar)
        similar = base + torch.randn_like(base) * 0.01
        similar = similar.clamp(-1, 1)
        
        # Large perturbation (perceptually different)
        different = base + torch.randn_like(base) * 0.3
        different = different.clamp(-1, 1)
        
        lpips_similar = compute_lpips(base, similar, device=torch.device("cpu"))
        lpips_different = compute_lpips(base, different, device=torch.device("cpu"))
        
        assert lpips_similar < lpips_different, "Perceptually similar images should have lower LPIPS"


class TestAllMetrics:
    """Test cases for compute_all_metrics."""

    def test_returns_dict(self):
        """Should return dict with all metrics."""
        img1 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        metrics = compute_all_metrics(img1, img2)
        assert isinstance(metrics, dict), "Should return dictionary"

    def test_keys_present(self):
        """Dict should contain psnr, ssim, lpips keys."""
        img1 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        metrics = compute_all_metrics(img1, img2)
        
        assert 'psnr' in metrics, "Should contain psnr"
        assert 'ssim' in metrics, "Should contain ssim"
        assert 'lpips' in metrics, "Should contain lpips"

    def test_identical_images_all_metrics(self):
        """Test all metrics on identical images."""
        img = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        metrics = compute_all_metrics(img, img)
        
        assert metrics['psnr'] == float('inf'), "PSNR should be inf"
        assert abs(metrics['ssim'] - 1.0) < 1e-5, "SSIM should be 1.0"
        if LPIPS_AVAILABLE:
            assert metrics['lpips'] < 1e-5, "LPIPS should be near 0"

    def test_metric_values_reasonable(self):
        """All metrics should have reasonable values."""
        img1 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(1, 3, 64, 64).clamp(-1, 1)
        metrics = compute_all_metrics(img1, img2)
        
        assert metrics['psnr'] > 0, "PSNR should be positive"
        assert -1 <= metrics['ssim'] <= 1, "SSIM should be in [-1, 1]"
        if LPIPS_AVAILABLE and metrics['lpips'] is not None:
            assert metrics['lpips'] >= 0, "LPIPS should be non-negative"

    def test_3d_inputs(self):
        """Should work with 3D inputs."""
        img1 = torch.randn(3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(3, 64, 64).clamp(-1, 1)
        metrics = compute_all_metrics(img1, img2)
        
        assert isinstance(metrics['psnr'], float)
        assert isinstance(metrics['ssim'], float)

    def test_batch_inputs(self):
        """Should work with batched inputs."""
        img1 = torch.randn(4, 3, 64, 64).clamp(-1, 1)
        img2 = torch.randn(4, 3, 64, 64).clamp(-1, 1)
        metrics = compute_all_metrics(img1, img2)
        
        # Should average over batch
        assert isinstance(metrics['psnr'], float)
        assert isinstance(metrics['ssim'], float)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_device(self):
        """Should work with GPU tensors."""
        img1 = torch.randn(1, 3, 64, 64).clamp(-1, 1).cuda()
        img2 = torch.randn(1, 3, 64, 64).clamp(-1, 1).cuda()
        metrics = compute_all_metrics(img1, img2, device=torch.device("cuda"))
        
        assert isinstance(metrics['psnr'], float)
        assert isinstance(metrics['ssim'], float)
        if LPIPS_AVAILABLE:
            assert isinstance(metrics['lpips'], float)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_black_images(self):
        """Test with all-black images."""
        img1 = torch.ones(1, 3, 64, 64) * -1.0  # All black in [-1, 1]
        img2 = torch.ones(1, 3, 64, 64) * -1.0
        
        psnr = compute_psnr(img1, img2)
        ssim = compute_ssim(img1, img2)
        
        assert psnr == float('inf'), "Identical black images should have inf PSNR"
        assert abs(ssim - 1.0) < 1e-5, "Identical black images should have SSIM = 1.0"

    def test_all_white_images(self):
        """Test with all-white images."""
        img1 = torch.ones(1, 3, 64, 64) * 1.0  # All white in [-1, 1]
        img2 = torch.ones(1, 3, 64, 64) * 1.0
        
        psnr = compute_psnr(img1, img2)
        ssim = compute_ssim(img1, img2)
        
        assert psnr == float('inf'), "Identical white images should have inf PSNR"
        assert abs(ssim - 1.0) < 1e-5, "Identical white images should have SSIM = 1.0"

    def test_black_vs_white(self):
        """Test maximum difference: all black vs all white."""
        img_black = torch.ones(1, 3, 64, 64) * -1.0
        img_white = torch.ones(1, 3, 64, 64) * 1.0
        
        psnr = compute_psnr(img_black, img_white, data_range=2.0)
        ssim = compute_ssim(img_black, img_white, data_range=2.0)
        
        # Maximum MSE = (2.0)^2 = 4.0
        # PSNR = 10 * log10(4/4) = 0 dB
        assert abs(psnr - 0.0) < 0.1, "Black vs white should give ~0 dB PSNR"
        assert ssim < 1.0, "Black vs white should have SSIM < 1.0"

    def test_small_images(self):
        """Test with very small images."""
        # Note: LPIPS requires minimum image size (at least 32x32 for AlexNet)
        img1 = torch.randn(1, 3, 32, 32).clamp(-1, 1)
        img2 = torch.randn(1, 3, 32, 32).clamp(-1, 1)
        
        # Test PSNR and SSIM only (LPIPS needs larger images)
        psnr = compute_psnr(img1, img2)
        ssim = compute_ssim(img1, img2)
        
        assert isinstance(psnr, float), "PSNR should return float"
        assert isinstance(ssim, float), "SSIM should return float"


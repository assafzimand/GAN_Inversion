"""
Unit tests for metrics computation.

TODO:
    - Test PSNR with known ground truth values
    - Test SSIM with known ground truth values
    - Test LPIPS metric computation
    - Test compute_all_metrics returns correct dict
    - Test edge cases (all black, all white, etc.)
"""

import pytest
import torch
import numpy as np


class TestPSNR:
    """Test cases for PSNR metric."""

    def test_identical_images(self):
        """PSNR should be inf for identical images."""
        # TODO: Implement test
        pytest.skip("Not yet implemented")

    def test_known_values(self):
        """Test PSNR with known ground truth."""
        # TODO: Implement test with calculated values
        pytest.skip("Not yet implemented")


class TestSSIM:
    """Test cases for SSIM metric."""

    def test_identical_images(self):
        """SSIM should be 1.0 for identical images."""
        # TODO: Implement test
        pytest.skip("Not yet implemented")

    def test_range(self):
        """SSIM should be in [0, 1]."""
        # TODO: Implement test
        pytest.skip("Not yet implemented")


class TestLPIPSMetric:
    """Test cases for LPIPS metric."""

    def test_identical_images(self):
        """LPIPS should be near 0 for identical images."""
        # TODO: Implement test
        pytest.skip("Not yet implemented")


class TestAllMetrics:
    """Test cases for compute_all_metrics."""

    def test_returns_dict(self):
        """Should return dict with all metrics."""
        # TODO: Implement test
        pytest.skip("Not yet implemented")

    def test_keys_present(self):
        """Dict should contain psnr, ssim, lpips keys."""
        # TODO: Implement test
        pytest.skip("Not yet implemented")


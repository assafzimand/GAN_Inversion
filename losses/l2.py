"""
L2 (MSE) pixel-wise loss for image reconstruction.

TODO:
    - Implement L2Loss class with proper normalization
    - Validate input shapes (must match)
    - Validate value ranges (expect [-1, 1] or [0, 1])
    - Handle batch dimensions correctly
    - Add unit tests
"""

import torch
import torch.nn as nn


class L2Loss(nn.Module):
    """
    Mean Squared Error loss for image reconstruction.

    Computes pixel-wise MSE between generated and target images.
    Assumes inputs are normalized to the same range.

    TODO:
        - Implement __init__ with optional reduction parameter
        - Implement forward(generated, target) -> loss scalar
        - Add shape and range validation
    """

    def __init__(self, reduction: str = "mean"):
        """
        Initialize L2 loss.

        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction
        # TODO: Implement initialization

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L2 loss between generated and target images.

        Args:
            generated: Generated images [B, C, H, W]
            target: Target images [B, C, H, W]

        Returns:
            Loss scalar (if reduction='mean')

        TODO: Implement loss computation
        """
        raise NotImplementedError("L2Loss.forward not yet implemented")


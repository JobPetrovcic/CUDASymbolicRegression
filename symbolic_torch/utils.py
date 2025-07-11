from typing import Any
import torch
from ._C import Operator

# This value must match the one in operators.h

def create_constants(ops_tensor: torch.Tensor, method: str = 'randn', **kwargs : Any) -> torch.Tensor:
    """
    Creates a tensor of constants for a given Ops tensor.

    Args:
        ops_tensor (torch.Tensor): The tensor defining the operations.
        method (str): The method to use for creating constants. One of ['randn', 'range'].
        **kwargs: Keyword arguments for the chosen method.
            For 'randn': mean (float, default 0.0), std (float, default 1.0).
            For 'range': lower (float or torch.Tensor, default 0.0), upper (float or torch.Tensor, default 1.0).

    Returns:
        torch.Tensor: A 1D tensor of constants.
    """
    num_constants = int((ops_tensor == int(Operator.LEARNABLE_CONSTANT)).sum().item())
    device = ops_tensor.device

    if method == 'randn':
        mean = kwargs.get('mean', 0.0)
        std = kwargs.get('std', 1.0)
        return torch.randn(num_constants, dtype=torch.float32, device=device) * std + mean
    elif method == 'range':
        lower = kwargs.get('lower', 0.0)
        upper = kwargs.get('upper', 1.0)
        return torch.rand(num_constants, dtype=torch.float32, device=device) * (upper - lower) + lower
    else:
        raise ValueError(f"Unknown method for constant creation: {method}")

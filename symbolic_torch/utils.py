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
        torch.Tensor: A 2D tensor of constants with the same shape as ops_tensor.
    """
    const_mask = (ops_tensor == int(Operator.LEARNABLE_CONSTANT))
    num_constants = int(const_mask.sum().item())
    device = ops_tensor.device
    dtype = torch.float32 # Default dtype

    constants_flat: torch.Tensor

    if method == 'randn':
        mean = kwargs.get('mean', 0.0)
        std = kwargs.get('std', 1.0)
        constants_flat = torch.randn(num_constants, dtype=dtype, device=device) * std + mean
    elif method == 'range':
        lower = kwargs.get('lower', 0.0)
        upper = kwargs.get('upper', 1.0)
        constants_flat = torch.rand(num_constants, dtype=dtype, device=device) * (upper - lower) + lower
    else:
        raise ValueError(f"Unknown method for constant creation: {method}")
    # TODO: maybe set others to NaN
    output = torch.zeros_like(ops_tensor, dtype=dtype, device=device)
    if num_constants > 0:
        output[const_mask] = constants_flat
    return output
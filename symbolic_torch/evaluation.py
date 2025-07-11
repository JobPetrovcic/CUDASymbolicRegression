import torch
from ._C import evaluate_backend 

def evaluate(
    X: torch.Tensor, # (N, n_x)
    Ops: torch.Tensor, # (B, M)
    Ch: torch.Tensor, # (B, M, MAX_ARITY)
    C: torch.Tensor # (SC, )
) -> torch.Tensor:
    """
    Python wrapper for the evaluate function that handles dimension swapping.
    PyTorch uses batch-first (B, ...), while the C++ implementation expects (..., B)
    for better memory layout and parallelization.

    Args:
        X (torch.Tensor): Input tensor for the variables of shape (N, n_x).
        Ops (torch.Tensor): Tensor of operations of shape (B, M).
        Ch (torch.Tensor): Tensor of children indices of shape (B, M, MAX_ARITY).
        C (torch.Tensor, optional): Tensor of learnable constants of shape (SC,). Defaults to None.

    Returns:
        torch.Tensor: The result of the evaluation of shape (B, N).
    """
    # Permute Ops and Ch from (B, M, ...) to (M, B, ...)
    Ops = Ops.permute(1, 0)
    Ch = Ch.permute(1, 0, 2)

    # The C++ function returns a tensor of shape (M, N, B)
    result = evaluate_backend(X, Ops, Ch, C)

    result = result.permute(2, 0, 1)  # Change to (B, M, N)
    
    return result

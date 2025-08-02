import torch
from ._C import evaluate_backend, evaluate_multiple_constant_backend

def evaluate(
    X: torch.Tensor, # (N, n_x)
    Ops: torch.Tensor, # (B, M)
    Ch: torch.Tensor, # (B, M, MAX_ARITY)
    C: torch.Tensor # (B, M)
) -> torch.Tensor:
    """
    Python wrapper for the evaluate function that handles dimension swapping.
    PyTorch uses batch-first (B, ...), while the C++ implementation expects (..., B)
    for better memory layout and parallelization.

    Args:
        X (torch.Tensor): Input tensor for the variables of shape (N, n_x).
        Ops (torch.Tensor): Tensor of operations of shape (B, M).
        Ch (torch.Tensor): Tensor of children indices of shape (B, M, MAX_ARITY).
        C (torch.Tensor): Tensor of learnable constants of shape (B, M).

    Returns:
        torch.Tensor: The result of the evaluation of shape (B, M, N).
    """
    # Permute Ops and Ch from (B, M, ...) to (M, B, ...)
    Ops = Ops.permute(1, 0)
    Ch = Ch.permute(1, 0, 2)
    C = C.permute(1, 0)

    # The C++ function returns a tensor of shape (M, N, B)
    result = evaluate_backend(X, Ops, Ch, C)

    result = result.permute(2, 0, 1)  # Change to (B, M, N)
    
    return result

def evaluate_multiple(
    X: torch.Tensor,      # (N, n_x)
    Ops: torch.Tensor,    # (B, M)
    Ch: torch.Tensor,     # (B, M, 2)
    C: torch.Tensor       # (B, M, K)
) -> torch.Tensor:
    """
    Python wrapper for evaluating expressions with multiple constant sets.
    Handles dimension swapping between PyTorch's (B, ...) and the backend's (..., B) layout.

    Args:
        X (torch.Tensor): Input variables, shape (N, n_x).
        Ops (torch.Tensor): Operations tensor, shape (B, M).
        Ch (torch.Tensor): Children tensor, shape (B, M, 2).
        C (torch.Tensor): Multiple constant sets tensor, shape (B, M, K).

    Returns:
        torch.Tensor: The evaluation result cache, shape (B, M, N, K).
    """
    # Permute from (B, M, ...) to (M, B, ...)
    Ops_permuted = Ops.permute(1, 0).contiguous()
    Ch_permuted = Ch.permute(1, 0, 2).contiguous()
    C_permuted = C.permute(1, 0, 2).contiguous() # (B, M, K) -> (M, B, K)

    # Backend returns a tensor of shape (M, N, B, K)
    result_cache = evaluate_multiple_constant_backend(X, Ops_permuted, Ch_permuted, C_permuted)
    
    # Permute back to a user-friendly (B, ...) layout
    # (M, N, B, K) -> (B, M, N, K)
    return result_cache.permute(2, 0, 1, 3)

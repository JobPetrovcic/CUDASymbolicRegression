# symbolic_torch/optimization.py
import torch
import torch.nn as nn
from ._C import Operator
from .evaluation import evaluate
from .lbfgs_batch import LBFGS_batch
from typing import Literal, Tuple

def optimize(
    X: torch.Tensor,
    Y: torch.Tensor,
    Ops: torch.Tensor,
    Ch: torch.Tensor,
    C_init: torch.Tensor,
    max_iter: int = 100,
    history_size: int = 10,
    tolerance_grad: float = 1e-7,
    line_search_fn: Literal['strong_wolfe', 'backtracking'] = 'strong_wolfe'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimizes the constants in a batch of expressions using the L-BFGS algorithm.

    This function uses an efficient (M, B) layout internally for C++ evaluation
    and the L-BFGS optimizer, but accepts and returns tensors in the standard
    PyTorch (B, M) layout.

    Args:
        X (torch.Tensor): Input data tensor of shape (N, n_x).
        Y (torch.Tensor): Target data tensor of shape (B, N).
        Ops (torch.Tensor): Operations tensor of shape (B, M).
        Ch (torch.Tensor): Children tensor of shape (B, M, 2).
        C_init (torch.Tensor): Initial constants tensor of shape (B, M).
        max_iter (int): Maximum number of iterations for L-BFGS.
        history_size (int): L-BFGS history size.
        tolerance_grad (float): L-BFGS termination tolerance on the gradient norm.
        line_search_fn (str): Line search method, 'strong_wolfe' or 'backtracking'.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The optimized constants tensor of shape (B, M).
            - The final predicted Y values of shape (B, N) using the optimized constants.
    """
    # Get shapes and device
    B, _ = Ops.shape
    N = X.shape[0]

    # --- Internal Layout: M-first for performance ---
    # LBFGS_batch expects parameters as (n_variables, n_problems)
    # Here, n_variables = M (all possible constant slots)
    # n_problems = B (batch size)
    C_opt = C_init.permute(1, 0).clone().detach().requires_grad_(True) # -> (M, B)
    Ops_internal = Ops.permute(1, 0) # -> (M, B)

    # Find the root node for each expression in the batch (for postfix)
    is_op = Ops != int(Operator.NO_OP)
    root_indices = is_op.sum(dim=1) - 1
    # Handle empty expressions
    root_indices[root_indices < 0] = 0

    # Create a mask for learnable constants to zero out gradients for non-constants
    const_mask_internal = (Ops_internal == int(Operator.LEARNABLE_CONSTANT)) # (M, B)

    optimizer = LBFGS_batch(
        [C_opt],
        max_iter=max_iter,
        history_size=history_size,
        tolerance_grad=tolerance_grad,
        line_search_fn=line_search_fn,
    )
    loss_function = nn.MSELoss(reduction='none')  

    def closure():
        optimizer.zero_grad()

        # `evaluate` wrapper expects (B, M) layout
        C_current = C_opt.permute(1, 0) # -> (B, M)

        # y_cache has shape (B, M, N)
        y_cache = evaluate(X, Ops, Ch, C_current)

        # Gather the final result for each expression from the cache
        # indices shape: (B, 1, 1) -> expanded to (B, 1, N) for gather
        indices = root_indices.view(B, 1, 1).expand(B, 1, N)
        y_pred = y_cache.gather(1, indices).squeeze(1) # -> (B, N)

        # LBFGS batch expects a per-problem loss vector (B,)
        loss_per_sample = loss_function(y_pred, Y) # -> (B, N)
        loss = loss_per_sample.mean(dim=1) # -> (B,)

        # Backpropagate on the sum of losses
        loss.sum().backward()

        # Manually zero out gradients for non-learnable constants
        # This ensures the L-BFGS search direction is sparse.
        with torch.no_grad():
            if C_opt.grad is not None:
                C_opt.grad[~const_mask_internal] = 0.0

        return loss

    optimizer.step(closure)

    # --- Reshape back to standard PyTorch layout ---
    C_final = C_opt.detach().permute(1, 0) # -> (B, M)
    
    # Calculate final Y prediction with optimized constants
    with torch.no_grad():
        y_cache_final = evaluate(X, Ops, Ch, C_final)
        indices = root_indices.view(B, 1, 1).expand(B, 1, N)
        Y_final = y_cache_final.gather(1, indices).squeeze(1) # -> (B, N)

    return C_final, Y_final
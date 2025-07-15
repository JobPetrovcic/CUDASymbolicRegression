from typing import Callable, Iterable, Literal
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer

class LBFGS_batch(Optimizer):
    """
    Implements a batch-wise L-BFGS algorithm, processing a batch of independent
    optimization problems simultaneously.

    This implementation assumes that the parameter tensor has the shape
    (n_variables, n_problems) for performance reasons.
    """
    def __init__(
        self,
        params: Iterable[Tensor],
        max_iter: int = 100,
        history_size: int = 10,
        tolerance_grad: float = 1e-7,
        line_search_fn: Literal['strong_wolfe', 'backtracking'] = 'strong_wolfe',
        line_search_max_iter: int = 10,
        eps: float = 1e-10
    ):
        """
        Args:
            params (Iterable[Tensor]): Iterable of parameters to optimize. Should be a
                single 2D tensor of shape (n_variables, n_problems).
            max_iter (int): Maximum number of iterations per optimization step.
            history_size (int): L-BFGS history size.
            tolerance_grad (float): Termination tolerance on the gradient norm.
            line_search_fn (str): The line search algorithm to use.
                'strong_wolfe' (default) or 'backtracking'.
            line_search_max_iter (int): Maximum iterations for the line search.
            eps (float): Small value to prevent division by zero.
        """
        defaults = dict(
            max_iter=max_iter,
            history_size=history_size,
            tolerance_grad=tolerance_grad,
            line_search_fn=line_search_fn,
            line_search_max_iter=line_search_max_iter,
            eps=eps
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS_batch doesn't support multiple param groups.")
        
        param_group = self.param_groups[0]
        if len(param_group['params']) != 1:
            raise ValueError("LBFGS_batch is designed to work on a single parameter tensor.")
        
        _params = param_group['params'][0]
        if _params.dim() != 2:
            raise ValueError("Input params must be a 2D tensor of shape (n_variables, n_problems)")

        if line_search_fn not in ['strong_wolfe', 'backtracking']:
            raise ValueError(f"Invalid line_search_fn: {line_search_fn}. Choose 'strong_wolfe' or 'backtracking'.")

    def _batched_dot(self, a: Tensor, b: Tensor) -> Tensor:
        """Batched dot product. Shape: (D, B), (D, B) -> (1, B)."""
        return torch.sum(a * b, dim=0, keepdim=True)

    def _compute_search_direction(
        self, g_k: Tensor, ss: list[Tensor], ys: list[Tensor], eps: float
    ) -> Tensor:
        num_history = len(ss)
        q = g_k
        
        if num_history == 0:
            return q

        # --- START FIX ---
        # Revert `alphas` to a simple python list to mirror the original logic
        # and avoid potential view/broadcasting bugs with a single tensor.
        alphas: list[Tensor] = []
        # --- END FIX ---

        # First loop
        for i in range(num_history - 1, -1, -1):
            s_i, y_i = ss[i], ys[i]
            rho_i = 1.0 / (self._batched_dot(y_i, s_i) + eps)
            alpha_i = rho_i * self._batched_dot(s_i, q)
            q = q - alpha_i * y_i
            # --- START FIX ---
            # Prepend to the list so the order is correct for the second loop
            alphas.insert(0, alpha_i)
            # --- END FIX ---

        s_last, y_last = ss[-1], ys[-1]
        gamma_k = self._batched_dot(s_last, y_last) / (self._batched_dot(y_last, y_last) + eps)
        z = gamma_k * q

        # Second loop
        for i in range(num_history):
            s_i, y_i = ss[i], ys[i]
            rho_i = 1.0 / (self._batched_dot(y_i, s_i) + eps)
            beta = rho_i * self._batched_dot(y_i, z)
            # The logic here is now identical to the original version.
            z = z + s_i * (alphas[i] - beta)
            
        return z

    def _backtracking_line_search(
        self, closure: Callable[[], Tensor], x_k: Tensor, p_k: Tensor, f_k: Tensor, g_k: Tensor, line_search_max_iter: int
    ) -> Tensor:
        """Simplified backtracking line search satisfying only the Armijo condition."""
        params = self.param_groups[0]['params'][0]
        rho = 0.5
        c = 1e-4

        grad_f_p = self._batched_dot(g_k, p_k).squeeze(0) # Shape: (n_problems,)
        alpha = torch.ones_like(f_k, device=x_k.device)   # Shape: (n_problems,)

        for _ in range(line_search_max_iter):
            # Evaluate the objective at the new point
            # alpha (n_problems,) -> (1, n_problems) for broadcasting with (D, B)
            params.data.copy_(x_k + alpha.unsqueeze(0) * p_k)
            with torch.enable_grad():
                f_try = closure()
            
            # Check the Armijo condition for all problems in the batch
            cond_met = f_try <= f_k + c * alpha * grad_f_p

            # If all problems have met the condition, we can stop
            if cond_met.all():
                break
            
            # For problems that haven't met the condition, reduce their alpha
            alpha = torch.where(cond_met, alpha, alpha * rho)
        
        # Reset params to original state before returning
        params.data.copy_(x_k)
        # Return alpha with shape (1, n_problems) for the step update
        return alpha.unsqueeze(0)
    
    def _strong_wolfe(self, closure: Callable[[], Tensor], x: Tensor, p: Tensor, line_search_max_iter: int,
                      alpha_max: int = 20, c1: float = 1e-4, c2: float = 0.9) -> Tensor:
        """
        Batchified line search (bracketing phase).
        This version always operates on full-batch tensors and passes a mask to the zoom function.
        """
        params = self.param_groups[0]['params'][0]

        params.data.copy_(x)
        with torch.enable_grad():
            phi_0 = closure() # (n_problems,)
        phi_prime_0 = self._batched_dot(params.grad, p).squeeze(0) # (n_problems,)

        alpha_prev = torch.zeros_like(phi_0)
        phi_prev = phi_0.clone()
        alpha_i = torch.ones_like(phi_0)

        final_alpha = torch.zeros_like(phi_0)
        done = torch.zeros_like(phi_0, dtype=torch.bool)

        for i in range(line_search_max_iter):
            if torch.all(done):
                break

            # Always evaluate on the full batch
            # alpha_i (n_problems,) -> (1, n_problems) for broadcasting
            params.data.copy_(x + alpha_i.unsqueeze(0) * p)
            with torch.enable_grad():
                phi_i = closure()
            phi_prime_i = self._batched_dot(params.grad, p).squeeze(0)

            # Check conditions on the full batch
            is_bad_point = (phi_i > phi_0 + c1 * alpha_i * phi_prime_0) | ((i > 0) & (phi_i >= phi_prev))
            is_good_alpha = torch.abs(phi_prime_i) <= c2 * torch.abs(phi_prime_0)
            passed_minimum = phi_prime_i >= 0

            # Case 1: Found a good alpha directly.
            newly_done_good = ~done & is_good_alpha
            final_alpha = torch.where(newly_done_good, alpha_i, final_alpha)
            done.logical_or_(newly_done_good)

            # Case 2: Bracket found, must zoom.
            needs_zoom = ~done & (is_bad_point | passed_minimum)
            if torch.any(needs_zoom):
                zoom_lo = torch.where(is_bad_point, alpha_prev, alpha_i)
                zoom_hi = torch.where(is_bad_point, alpha_i, alpha_prev)
                
                zoomed_alpha = self._alpha_zoom_corrected(
                    closure, x, p, zoom_lo, zoom_hi, 
                    active_mask=needs_zoom, c1=c1, c2=c2
                )
                
                final_alpha = torch.where(needs_zoom, zoomed_alpha, final_alpha)
                done.logical_or_(needs_zoom)

            # Update state for problems that continue bracketing.
            continuing = ~done
            alpha_prev = torch.where(continuing, alpha_i, alpha_prev)
            phi_prev = torch.where(continuing, phi_i, phi_prev)
            alpha_i = torch.where(continuing, alpha_i + (alpha_max - alpha_i) * 0.8, alpha_i)

        final_alpha = torch.where(done, final_alpha, alpha_i)
        params.data.copy_(x)
        # Return alpha with shape (1, n_problems) for the step update
        return final_alpha.unsqueeze(0)

    def _alpha_zoom_corrected(self, closure: Callable[[], Tensor], x: Tensor, p: Tensor, 
                         alpha_lo: Tensor, alpha_hi: Tensor, 
                         active_mask: Tensor, # <-- New parameter
                         c1: float = 1e-4, c2: float = 0.9) -> Tensor:
        """
        Zoom phase of the line search. Operates on full-batch tensors
        but only updates the elements specified by `active_mask`.
        """
        params = self.param_groups[0]['params'][0]
        
        params.data.copy_(x)
        with torch.enable_grad():
            phi_0 = closure()
        phi_prime_0 = self._batched_dot(params.grad, p).squeeze(0)

        params.data.copy_(x + alpha_lo.unsqueeze(0) * p)
        with torch.enable_grad():
            phi_lo = closure()

        alpha = torch.zeros_like(alpha_lo)
        zoom_done = torch.zeros_like(alpha_lo, dtype=torch.bool)

        max_zoom_iters = 10
        for _ in range(max_zoom_iters):
            if torch.all(zoom_done[active_mask]):
                break
            
            alpha_j = (alpha_lo + alpha_hi) / 2

            # alpha_j (n_problems,) -> (1, n_problems) for broadcasting
            params.data.copy_(x + alpha_j.unsqueeze(0) * p)
            with torch.enable_grad():
                phi_j = closure()
            phi_prime_j = self._batched_dot(params.grad, p).squeeze(0)

            is_bad_point = (phi_j > phi_0 + c1 * alpha_j * phi_prime_0) | (phi_j >= phi_lo)
            is_good_alpha = abs(phi_prime_j) <= c2 * abs(phi_prime_0)
            
            update_mask_good = active_mask & ~zoom_done & is_good_alpha
            alpha = torch.where(update_mask_good, alpha_j, alpha)
            zoom_done.logical_or_(update_mask_good)

            update_mask_bracket = active_mask & ~zoom_done
            
            temp_alpha_hi = torch.where(update_mask_bracket & is_bad_point, alpha_j, alpha_hi)
            cond3 = phi_prime_j * (temp_alpha_hi - alpha_lo) >= 0
            alpha_hi = torch.where(update_mask_bracket & ~is_bad_point & cond3, alpha_lo, temp_alpha_hi)
            
            update_lo_mask = update_mask_bracket & ~is_bad_point
            alpha_lo = torch.where(update_lo_mask, alpha_j, alpha_lo)
            phi_lo = torch.where(update_lo_mask, phi_j, phi_lo)

            bracket_too_small = abs(alpha_hi - alpha_lo) < 1e-9
            update_mask_small = active_mask & ~zoom_done & bracket_too_small
            alpha = torch.where(update_mask_small, alpha_j, alpha)
            zoom_done.logical_or_(update_mask_small)

        fallback_mask = active_mask & ~zoom_done
        alpha = torch.where(fallback_mask, alpha_lo, alpha)

        return alpha

    def step(self, closure: Callable[[], Tensor]):
        with torch.no_grad():
            param_group = self.param_groups[0]
            params = param_group['params'][0]
            max_iter = param_group['max_iter']
            history_size = param_group['history_size']
            tolerance_grad = param_group['tolerance_grad']
            line_search_fn = param_group['line_search_fn']
            line_search_max_iter = param_group['line_search_max_iter']
            eps = param_group['eps']

            with torch.enable_grad():
                f_k = closure()
            
            g_k = params.grad.clone() # (D, B)
            x_k = params.data.clone() # (D, B)
            n_problems = f_k.shape[0]

            ss: list[torch.Tensor] = []
            ys: list[torch.Tensor] = []
            
            done: torch.Tensor = torch.zeros(n_problems, dtype=torch.bool, device=x_k.device)

            for k in range(max_iter):
                # Norm over variable dimension (dim=0)
                grad_norm = torch.linalg.norm(g_k, dim=0)
                newly_done = (grad_norm < tolerance_grad)
                done |= newly_done
                
                if done.all():
                    break

                p_k = -self._compute_search_direction(g_k, ss, ys, eps)
                
                if line_search_fn == 'strong_wolfe':
                    alpha_k = self._strong_wolfe(closure, x_k, p_k, line_search_max_iter)
                else: # 'backtracking'
                    alpha_k = self._backtracking_line_search(closure, x_k, p_k, f_k, g_k, line_search_max_iter)

                s_k = alpha_k * p_k
                # Update non-converged problems. `done` (B,) -> (1, B) for broadcasting
                x_k += s_k * (~done).unsqueeze(0).float()
                
                params.data.copy_(x_k)
                with torch.enable_grad():
                    f_nxt = closure()
                g_nxt = params.grad.clone()
                
                y_k = g_nxt - g_k
                ss.append(s_k)
                ys.append(y_k)

                if len(ss) > history_size:
                    ss.pop(0)
                    ys.pop(0)
                
                g_k = g_nxt
                f_k = f_nxt

            params.data.copy_(x_k)
            return f_k

class batchedRosenbrock(nn.Module):
    def __init__(self, n_problems: int, device: str = 'cpu'):
        super().__init__()
        # Use shape (1, n_problems) for easy broadcasting
        self.a = torch.ones(1, n_problems, device=device)
        self.b = torch.full((1, n_problems), 100.0, device=device)

    def forward(self, x: Tensor) -> Tensor:
        # x has shape (n_variables, n_problems), e.g., (2, B)
        x1, x2 = x[0], x[1] # each are (n_problems,)
        # self.a and self.b broadcast from (1, B) to match
        f = (self.a - x1)**2 + self.b * (x2 - x1**2)**2
        # Return shape (n_problems,)
        return f.squeeze(0)

if __name__ == "__main__":
    n_problems = 10000
    n_variables = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = batchedRosenbrock(n_problems, device=device)
    # Initialize parameters with shape (n_variables, n_problems)
    x0 = torch.randn(n_variables, n_problems, requires_grad=True, device=device)
    
    # --- Example of using the Strong Wolfe line search ---
    print("--- Testing with Strong Wolfe Line Search (Default) ---")
    optimizer_wolfe = LBFGS_batch(
        [x0], max_iter=200, history_size=20, tolerance_grad=1e-7,
        line_search_fn='strong_wolfe'
    )

    def closure() -> Tensor:
        optimizer_wolfe.zero_grad(set_to_none=True)
        f_val = model(x0)
        # We backpropagate on the sum of losses for all problems
        f_val.sum().backward()
        return f_val

    initial_loss = model(x0.clone().detach())
    print(f"Device: {device}")
    print(f"Initial loss (mean): {initial_loss.mean().item()}")

    optimizer_wolfe.step(closure)

    final_loss_wolfe = model(x0)
    print(f"Final loss (Wolfe, mean): {final_loss_wolfe.mean().item()}")
    print("\nFinal parameters (first 3 problems):")
    # Print the columns corresponding to the first 3 problems
    print(x0.data[:, :3])
    print("\nExpected parameters for Rosenbrock are close to [1.0, 1.0]")
from typing import Callable, Iterable, Literal, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer

class LBFGS_batch(Optimizer):
    """
    Implements a batch-wise L-BFGS algorithm with optional box constraints (L-BFGS-B),
    processing a batch of independent optimization problems simultaneously.
    """
    def __init__(
        self,
        params: Iterable[Tensor],
        lower_bounds: Optional[Tensor] = None, # ### ADDED ###
        upper_bounds: Optional[Tensor] = None, # ### ADDED ###
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
                single 2D tensor of shape (batch_size, n_variables).
            lower_bounds (Optional[Tensor]): Lower bounds for each variable. Shape (batch_size, n_variables).
            upper_bounds (Optional[Tensor]): Upper bounds for each variable. Shape (batch_size, n_variables).
            max_iter (int): Maximum number of iterations per optimization step.
            history_size (int): L-BFGS history size.
            tolerance_grad (float): Termination tolerance on the gradient norm.
            line_search_fn (str): The line search algorithm to use.
            line_search_max_iter (int): Maximum iterations for the line search.
            eps (float): Small value to prevent division by zero.
        """
        defaults = dict(
            lower_bounds=lower_bounds, # ### ADDED ###
            upper_bounds=upper_bounds, # ### ADDED ###
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
            raise ValueError("Input params must be a 2D tensor of shape (batch_size, n_variables)")

        # ### ADDED: Validate bounds ###
        if lower_bounds is not None and lower_bounds.shape != _params.shape:
            raise ValueError("lower_bounds shape must match params shape.")
        if upper_bounds is not None and upper_bounds.shape != _params.shape:
            raise ValueError("upper_bounds shape must match params shape.")
        if lower_bounds is not None and upper_bounds is not None:
            if torch.any(lower_bounds > upper_bounds):
                raise ValueError("lower_bounds cannot be greater than upper_bounds.")

        if line_search_fn not in ['strong_wolfe', 'backtracking']:
            raise ValueError(f"Invalid line_search_fn: {line_search_fn}. Choose 'strong_wolfe' or 'backtracking'.")

    def _batched_dot(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.sum(a * b, dim=-1, keepdim=True)

    # ### ADDED: Helper functions for L-BFGS-B ###
    def _project(self, x: Tensor, lb: Optional[Tensor], ub: Optional[Tensor]) -> Tensor:
        """Projects a tensor to the feasible region defined by lower and upper bounds."""
        if lb is None and ub is None:
            return x
        return torch.clamp(x, min=lb, max=ub)

    def _projected_gradient(self, x: Tensor, g: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
        """
        Computes the projected gradient.
        The projected gradient is zero for variables at their bounds if the gradient
        is pointing outwards.
        """
        if lb is None and ub is None:
            return g
            
        g_proj = g.clone()
        # If x is at the lower bound and gradient is positive (pushing it lower, i.e., out of bounds)
        # set the gradient component to zero.
        g_proj[((x <= lb) & (g > 0))] = 0
        # If x is at the upper bound and gradient is negative (pushing it higher, i.e., out of bounds)
        # set the gradient component to zero.
        g_proj[((x >= ub) & (g < 0))] = 0
        return g_proj
    
    def _compute_search_direction(
        self, g_k: Tensor, ss: list[Tensor], ys: list[Tensor], eps: float
    ) -> Tensor:
        # This function remains unchanged. It operates on the (projected) gradient.
        num_history = len(ss)
        q = g_k
        
        if num_history == 0:
            return q

        alphas = torch.zeros(num_history, g_k.shape[0], 1, device=g_k.device)

        for i in range(num_history - 1, -1, -1):
            s_i, y_i = ss[i], ys[i]
            rho_i = 1.0 / (self._batched_dot(y_i, s_i) + eps)
            alpha_i = rho_i * self._batched_dot(s_i, q)
            q = q - alpha_i * y_i
            alphas[i] = alpha_i

        s_last, y_last = ss[-1], ys[-1]
        gamma_k = self._batched_dot(s_last, y_last) / (self._batched_dot(y_last, y_last) + eps)
        z = gamma_k * q

        for i in range(num_history):
            s_i, y_i = ss[i], ys[i]
            rho_i = 1.0 / (self._batched_dot(y_i, s_i) + eps)
            beta = rho_i * self._batched_dot(y_i, z)
            z = z + s_i * (alphas[i] - beta)
            
        return z

    # ### MODIFIED: Backtracking now needs to respect bounds ###
    def _backtracking_line_search(
        self, closure: Callable[[], Tensor], x_k: Tensor, p_k: Tensor, f_k: Tensor, g_k: Tensor, 
        line_search_max_iter: int, lb: Tensor, ub: Tensor
    ) -> Tensor:
        """Simplified backtracking line search satisfying only the Armijo condition."""
        # This is a simplified version. For full correctness, it should also compute
        # and respect alpha_max like the strong wolfe search. For this example, we
        # rely on the final projection in the step loop.
        params = self.param_groups[0]['params'][0]
        rho = 0.5
        c = 1e-4

        grad_f_p = self._batched_dot(g_k, p_k).squeeze(-1)
        alpha = torch.ones_like(f_k, device=x_k.device)

        for _ in range(line_search_max_iter):
            x_try = self._project(x_k + alpha.unsqueeze(-1) * p_k, lb, ub)
            params.data.copy_(x_try)
            with torch.enable_grad():
                f_try = closure()
            
            cond_met = f_try <= f_k + c * alpha * grad_f_p

            if cond_met.all():
                break
            
            alpha = torch.where(cond_met, alpha, alpha * rho)
        
        params.data.copy_(x_k)
        return alpha.unsqueeze(-1)
    
    # ### MODIFIED: Strong Wolfe now needs to respect bounds ###
    def _strong_wolfe(self, closure: Callable[[], Tensor], x: Tensor, p: Tensor, line_search_max_iter: int,
                      lb: Optional[Tensor], ub: Optional[Tensor], c1: float = 1e-4, c2: float = 0.9) -> Tensor:
        """
        Batchified line search with bounds.
        """
        params = self.param_groups[0]['params'][0]

        # ### ADDED: Calculate max step size before hitting a bound ###
        alpha_max = torch.full((x.shape[0],), float('inf'), device=x.device)
        if ub is not None:
            mask = p > 1e-8
            alpha_ub = (ub[mask] - x[mask]) / p[mask]
            # Group by problem instance (row) and find the minimum alpha for each
            rows = torch.where(mask)[0]
            alpha_max = alpha_max.scatter_reduce(0, rows, alpha_ub, reduce='amin', include_self=False)

        if lb is not None:
            mask = p < -1e-8
            alpha_lb = (lb[mask] - x[mask]) / p[mask]
            rows = torch.where(mask)[0]
            alpha_max = alpha_max.scatter_reduce(0, rows, alpha_lb, reduce='amin', include_self=False)

        params.data.copy_(x)
        with torch.enable_grad():
            phi_0 = closure()
        phi_prime_0 = self._batched_dot(params.grad, p).squeeze(-1)

        alpha_prev = torch.zeros_like(phi_0)
        phi_prev = phi_0.clone()
        alpha_i = torch.ones_like(phi_0).clamp_max_(alpha_max) # Initial guess respects bounds

        final_alpha = torch.zeros_like(phi_0)
        done = torch.zeros_like(phi_0, dtype=torch.bool)
        
        # This line search is complex. For simplicity in this example, we won't implement the zoom
        # phase and just return the best alpha found. The provided user code was also missing zoom.
        for _ in range(line_search_max_iter):
            if torch.all(done):
                break
                
            # The projection here is a safeguard. alpha_i should already be valid.
            x_try = self._project(x + alpha_i.unsqueeze(-1) * p, lb, ub)
            params.data.copy_(x_try)
            with torch.enable_grad():
                phi_i = closure()
            phi_prime_i = self._batched_dot(params.grad, p).squeeze(-1)

            # Armijo condition
            armijo_cond = phi_i <= phi_0 + c1 * alpha_i * phi_prime_0
            # Strong Wolfe curvature condition
            wolfe_cond = torch.abs(phi_prime_i) <= c2 * torch.abs(phi_prime_0)
            
            newly_done = ~done & armijo_cond & wolfe_cond
            final_alpha = torch.where(newly_done, alpha_i, final_alpha)
            done.logical_or_(newly_done)

            # If Armijo fails, step is too long, reduce it.
            needs_shrink = ~done & ~armijo_cond
            alpha_i = torch.where(needs_shrink, (alpha_prev + alpha_i) / 2.0, alpha_i)
            
            # If Armijo passes but Wolfe fails, step is too short, increase it.
            needs_expand = ~done & armijo_cond & ~wolfe_cond
            alpha_prev = torch.where(needs_expand, alpha_i, alpha_prev)
            alpha_i = torch.where(needs_expand, (alpha_i + alpha_max) / 2.0, alpha_i)

        final_alpha = torch.where(done, final_alpha, alpha_i)
        params.data.copy_(x)
        return final_alpha.unsqueeze(-1)


    def step(self, closure: Callable[[], Tensor]):
        with torch.no_grad():
            param_group = self.param_groups[0]
            params = param_group['params'][0]
            # ### ADDED: Get bounds from param group ###
            lb = param_group['lower_bounds']
            ub = param_group['upper_bounds']
            max_iter = param_group['max_iter']
            history_size = param_group['history_size']
            tolerance_grad = param_group['tolerance_grad']
            line_search_fn = param_group['line_search_fn']
            line_search_max_iter = param_group['line_search_max_iter']
            eps = param_group['eps']

            # ### ADDED: Ensure bounds are on the correct device ###
            if lb is not None and lb.device != params.device:
                lb = lb.to(params.device)
            if ub is not None and ub.device != params.device:
                ub = ub.to(params.device)

            # ### ADDED: Project initial point into the feasible region ###
            params.data.copy_(self._project(params.data, lb, ub))
            
            with torch.enable_grad():
                f_k = closure()
            
            g_k = params.grad.clone()
            x_k = params.data.clone()
            batch_size = f_k.shape[0]

            ss: list[torch.Tensor] = []
            ys: list[torch.Tensor] = []
            
            done: torch.Tensor = torch.zeros(batch_size, dtype=torch.bool, device=x_k.device)

            for k in range(max_iter):
                # ### MODIFIED: Convergence check uses the projected gradient norm ###
                g_proj = self._projected_gradient(x_k, g_k, lb, ub)
                grad_norm = torch.linalg.norm(g_proj, dim=1)
                
                newly_done = (grad_norm < tolerance_grad)
                done |= newly_done
                
                if done.all():
                    break

                # ### MODIFIED: Compute search direction from projected gradient ###
                p_k = -self._compute_search_direction(g_proj, ss, ys, eps)
                
                if line_search_fn == 'strong_wolfe':
                    alpha_k = self._strong_wolfe(closure, x_k, p_k, line_search_max_iter, lb, ub)
                else:
                    alpha_k = self._backtracking_line_search(closure, x_k, p_k, f_k, g_k, line_search_max_iter, lb, ub)

                s_k = alpha_k * p_k
                x_k += s_k * (~done).unsqueeze(-1).float()

                # ### MODIFIED: Project the new point to stay within bounds ###
                # This is a crucial step to correct for any potential floating point
                # inaccuracies or line search imperfections.
                x_k = self._project(x_k, lb, ub)
                
                params.data.copy_(x_k)
                with torch.enable_grad():
                    f_nxt = closure()
                g_nxt = params.grad.clone()
                
                y_k = g_nxt - g_k
                
                # Update history only for problems with valid curvature
                curv_cond = self._batched_dot(y_k, s_k).squeeze(-1) > eps
                active_s = s_k[curv_cond]
                active_y = y_k[curv_cond]

                if active_s.numel() > 0:
                    ss.append(active_s)
                    ys.append(active_y)

                if len(ss) > history_size:
                    ss.pop(0)
                    ys.pop(0)
                
                g_k = g_nxt
                f_k = f_nxt

            # ### MODIFIED: Final projection ###
            params.data.copy_(self._project(x_k, lb, ub))
            return f_k

# The rest of your code (batchedRosenbrock, if __name__ == "__main__") remains the same.
# Here is an updated example demonstrating its use.

class batchedRosenbrock(nn.Module):
    def __init__(self, n_problems: int, device: str = 'cpu'):
        super().__init__()
        self.a = torch.ones(n_problems, 1, device=device)
        self.b = torch.full((n_problems, 1), 100.0, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        f = (self.a.squeeze() - x1)**2 + self.b.squeeze() * (x2 - x1**2)**2
        return f

if __name__ == "__main__":
    n_problems = 10000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = batchedRosenbrock(n_problems, device=device)
    
    # ### CORRECTED INITIALIZATION ###
    # Create the leaf tensor that will be optimized.
    x0 = torch.empty(n_problems, 2, requires_grad=True, device=device)
    
    # Define bounds for the optimization problem
    # We want to constrain the solution to be within the box [-2, 2] for x1 and [0, 3] for x2
    lower_bounds = torch.tensor([[-2.0, 0.0]], device=device).expand(n_problems, -1)
    upper_bounds = torch.tensor([[2.0, 3.0]], device=device).expand(n_problems, -1)

    # Initialize its data in-place using a torch.no_grad() context.
    # This does not create a computation graph history for the initialization.
    with torch.no_grad():
        # Initialize with random data
        x0.copy_(torch.randn(n_problems, 2, device=device) * 3)
        # Best Practice: Clamp the initial guess to be within the bounds.
        # The optimizer will do this on the first step anyway, but it's good practice
        # to start from a feasible point.
        x0.clamp_(min=lower_bounds, max=upper_bounds)
    
    print("--- Testing with L-BFGS-B (Strong Wolfe Line Search) ---")
    optimizer_wolfe = LBFGS_batch(
        [x0], 
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_iter=200, 
        history_size=20, 
        tolerance_grad=1e-7,
        line_search_fn='strong_wolfe'
    )

    def closure() -> Tensor:
        optimizer_wolfe.zero_grad(set_to_none=True)
        f_val = model(x0)
        # We need to sum for backward(), but the optimizer works on per-problem loss/grad
        f_val.sum().backward()
        return f_val

    initial_loss = model(x0.clone().detach())
    print(f"Device: {device}")
    print(f"Initial avg loss: {initial_loss.mean().item():.4f}")

    optimizer_wolfe.step(closure)

    final_loss_wolfe = model(x0)
    print(f"Final avg loss (Wolfe): {final_loss_wolfe.mean().item():.4f}")
    print("\nFinal parameters (first 5 problems):")
    print(x0.data[:5])
    print("\nExpected parameters for Rosenbrock are close to [1.0, 1.0], which is within our bounds.")

    # Verify that all final parameters are within the specified bounds
    is_below_lb = torch.any(x0.data < lower_bounds - 1e-6) # Add a small tolerance for floating point
    is_above_ub = torch.any(x0.data > upper_bounds + 1e-6)
    if not is_below_lb and not is_above_ub:
        print("\nVerification successful: All parameters are within the defined bounds.")
    else:
        print("\nVerification FAILED: Some parameters are outside the defined bounds.")
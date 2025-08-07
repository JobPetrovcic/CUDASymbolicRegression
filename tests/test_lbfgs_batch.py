# test_lbfgs.py

from typing import Literal
import torch
import torch.nn as nn
from torch import Tensor
import pytest

# Import the class to be tested
from symbolic_torch.lbfgs_batch import LBFGS_batch
from .utils import get_cuda_device_with_min_memory


# --- Test Helper Fixtures ---

@pytest.fixture(params=['cpu', 'cuda'], scope="module")
def device(request: pytest.FixtureRequest) -> str:
    """Parametrized fixture for CPU and CUDA devices."""
    if request.param == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this system.")
    index = get_cuda_device_with_min_memory()
    return f"cuda:{index}" if request.param == "cuda" else "cpu"

class batchedRosenbrock(nn.Module):
    """Test function for optimization. Minimum is at (1, 1)."""
    def __init__(self, n_problems: int, device: str):
        super().__init__() # type: ignore
        self.a = torch.ones(1, n_problems, device=device)
        self.b = torch.full((1, n_problems), 100.0, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x[0, :], x[1, :]
        f = (self.a.squeeze(0) - x1)**2 + self.b.squeeze(0) * (x2 - x1**2)**2
        return f


# --- Test Functions ---

@pytest.mark.parametrize("line_search_fn", ["strong_wolfe", "backtracking"])
def test_correctness_rosenbrock(device : str, line_search_fn : Literal['strong_wolfe', 'backtracking']):
    """
    Tests if the optimizer correctly minimizes the Rosenbrock function for a small batch.
    This test is parameterized to run for both line search methods.
    """
    n_problems = 10
    model = batchedRosenbrock(n_problems, device=device)
    x0 = torch.randn(2, n_problems, requires_grad=True, device=device)
    
    optimizer = LBFGS_batch(
        [x0], 
        max_iter=250, 
        history_size=15, 
        tolerance_grad=1e-7,
        line_search_fn=line_search_fn
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        f_val = model(optimizer.param_groups[0]['params'][0])
        f_val.sum().backward()
        return f_val

    # Run the optimization
    optimizer.step(closure)
    
    # --- Assertions ---
    final_loss = model(x0)
    expected_solution = torch.ones_like(x0)
    
    # 1. The final loss should be very close to zero.
    assert final_loss.mean() < 1e-4, f"Final loss was not close to zero for {line_search_fn}, got {final_loss.mean().item()}"
    
    # 2. The final parameters should be very close to the known minimum [1.0, 1.0].
    assert torch.allclose(x0, expected_solution, atol=1e-3), \
        f"Solution did not converge to [1, 1] for {line_search_fn}"


@pytest.mark.large
def test_benchmark_large_batch(benchmark, device : str):
    """
    Benchmarks the optimizer on a large number of problems.
    This test is marked as 'large' and can be skipped with `pytest -m "not large"`.
    Requires `pytest-benchmark` to be installed.
    """
    n_problems = 50_000
    max_iter = 50
    
    if device == 'cpu' and n_problems > 10_000:
        pytest.skip("Skipping large benchmark on CPU to save time.")
    
    if device == 'cuda':
        index = get_cuda_device_with_min_memory()
        device = f"cuda:{index}"

    model = batchedRosenbrock(n_problems, device=device)
    x0 = torch.randn(2, n_problems, requires_grad=True, device=device)

    optimizer = LBFGS_batch(
        [x0], 
        max_iter=max_iter, 
        history_size=10, 
        tolerance_grad=1e-7,
        line_search_fn='strong_wolfe'
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        f_val = model(optimizer.param_groups[0]['params'][0])
        f_val.sum().backward()
        return f_val

    def target_func():
        # Reset parameters for each benchmark round to ensure fairness
        x0.data.copy_(torch.randn(2, n_problems, device=device))
        optimizer.step(closure)
        # For GPU, wait for all kernels to finish
        if device == 'cuda':
            torch.cuda.synchronize(device=device)

    # The benchmark fixture handles timing, warm-up, and statistics
    benchmark(target_func)
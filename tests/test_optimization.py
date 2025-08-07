# tests/test_optimization.py
import time
import torch
import torch.nn as nn
import pytest
from symbolic_torch import Operator, create_constants, optimize
from symbolic_torch.evaluation import evaluate
from .utils import get_cuda_device_with_min_memory

@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_optimize_constants(device_str : str):
    """
    Tests if the optimize function can correctly find the constants for a
    batched linear expression y = c1*x + c2 and that the final returned
    Y value is correct.
    """
    if not torch.cuda.is_available() and device_str == "cuda":
        raise ValueError("CUDA is not available on this system.")
    if device_str == "cpu":
        device = torch.device("cpu")
    else:
        index = get_cuda_device_with_min_memory()
        device = torch.device(f"cuda:{index}")

    # --- Setup ---
    B, N, M, n_x = 2, 100, 5, 1

    # Target function: y = c1 * X_0 + c2
    # Two different problems in the batch
    c1_true = torch.tensor([2.5, -1.5], device=device).view(B, 1)
    c2_true = torch.tensor([0.5, 3.0], device=device).view(B, 1)

    X = torch.randn(N, n_x, device=device)
    # Y_true must have shape (B, N)
    Y_true = c1_true * X[:, 0].T + c2_true

    # Expression in postfix: X_0 C * C +
    # Ops shape (B, M)
    Ops = torch.full((B, M), int(Operator.NO_OP), dtype=torch.int64, device=device)
    Ops[:, 0] = int(Operator.VAR_START_ID)
    Ops[:, 1] = int(Operator.LEARNABLE_CONSTANT) # This will be c1
    Ops[:, 2] = int(Operator.MUL)
    Ops[:, 3] = int(Operator.LEARNABLE_CONSTANT) # This will be c2
    Ops[:, 4] = int(Operator.ADD)

    # Ch shape (B, M, 2)
    Ch = torch.full((B, M, 2), -1, dtype=torch.int64, device=device)
    Ch[:, 2, 0] = 0 # MUL children: X_0 and C1
    Ch[:, 2, 1] = 1
    Ch[:, 4, 0] = 2 # ADD children: (X_0*C1) and C2
    Ch[:, 4, 1] = 3

    # C_init shape (B, M)
    # Initialize constants away from the true values
    C_init = create_constants(Ops, method='randn', mean=0.0, std=1.0)

    # --- Run Optimization ---
    C_final, Y_final = optimize(
        X, Y_true, Ops, Ch, C_init,
    )

    # --- Assertions ---
    # 1. Check if the optimized constants are close to the true values
    c1_found = C_final[:, 1]
    c2_found = C_final[:, 3]

    assert torch.allclose(c1_found, c1_true.squeeze(), atol=1e-3), \
        f"c1 failed to converge. Found: {c1_found}, Expected: {c1_true.squeeze()}"
    assert torch.allclose(c2_found, c2_true.squeeze(), atol=1e-3), \
        f"c2 failed to converge. Found: {c2_found}, Expected: {c2_true.squeeze()}"

    # 2. Check that other constants in C_final (which are not learnable) remain unchanged
    const_mask = (Ops == int(Operator.LEARNABLE_CONSTANT))
    assert torch.allclose(C_final[~const_mask], C_init[~const_mask]), \
        "Non-learnable constants were modified during optimization."

    # 3. Check that the returned Y_final is very close to Y_true
    assert Y_final.shape == Y_true.shape, "Shape of final Y prediction is incorrect."
    assert torch.allclose(Y_final, Y_true, atol=1e-3), \
        f"Final Y prediction is not close to true Y. Max error: {torch.max(torch.abs(Y_final - Y_true))}"
    
@pytest.mark.large
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_large_batch_optimization(device_str: str):
    """
    Benchmarks the optimization performance on a large batch of expressions.
    This test is marked as 'large' and can be skipped with `pytest -m "not large"`.
    """
    if not torch.cuda.is_available() and device_str == "cuda":
        raise ValueError("CUDA is not available on this system.")

    if device_str == "cpu":
        device = torch.device("cpu")
    else:
        index = get_cuda_device_with_min_memory()
        device = f"cuda:{index}"
        print(f"Using CUDA device: {device}")
        device = torch.device(device)

    B = 10000
    if device.type == 'cpu' and B >= 10000:
        pytest.skip("Skipping large optimization benchmark on CPU to save time.")

    # --- Setup ---
    N, M, n_x = 50, 6, 1

    # Target function: y = c1 * sin(X_0) + c2
    c1_true = torch.randn(B, 1, device=device) * 2 # Random true constants per problem
    c2_true = torch.randn(B, 1, device=device)
    
    X = torch.rand(N, n_x, device=device) * 4 - 2 # Range [-2, 2]
    Y_true = c1_true * torch.sin(X[:, 0]).T + c2_true

    # Expression in postfix: X_0 sin C * C +
    # Ops shape (B, M)
    Ops = torch.full((B, M), int(Operator.NO_OP), dtype=torch.int64, device=device)
    Ops[:, 0] = int(Operator.VAR_START_ID)
    Ops[:, 1] = int(Operator.SIN)
    Ops[:, 2] = int(Operator.LEARNABLE_CONSTANT) # c1
    Ops[:, 3] = int(Operator.MUL)
    Ops[:, 4] = int(Operator.LEARNABLE_CONSTANT) # c2
    Ops[:, 5] = int(Operator.ADD)
    
    # Ch shape (B, M, 2)
    Ch = torch.full((B, M, 2), -1, dtype=torch.int64, device=device)
    Ch[:, 1, 0] = 0 # sin(X_0)
    Ch[:, 3, 0] = 1 # mul(sin(X_0), C1)
    Ch[:, 3, 1] = 2
    Ch[:, 5, 0] = 3 # add(mul(...), C2)
    Ch[:, 5, 1] = 4
    
    # C_init shape (B, M)
    C_init = create_constants(Ops, method='randn')
    
    loss_fn = nn.MSELoss() # Use standard MSE for the final check

    # --- Time the optimization ---
    start_time = time.time()

    _, Y_final = optimize(
        X, Y_true, Ops, Ch, C_init
    )
    
    if device.type == 'cuda':
        torch.cuda.synchronize(device=device)
    
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\n[Large Batch Optimization Benchmark] Device: {device_str}, Batch Size: {B}, Duration: {duration:.4f}s")
    
    # --- Assertions ---
    # Calculate initial loss to ensure optimization improved the result.
    is_op = Ops != int(Operator.NO_OP)
    root_indices = is_op.sum(dim=1) - 1
    root_indices[root_indices < 0] = 0
    
    with torch.no_grad():
        y_cache_init = evaluate(X, Ops, Ch, C_init)
        indices = root_indices.view(B, 1, 1).expand(B, 1, N)
        Y_init = y_cache_init.gather(1, indices).squeeze(1)

    initial_mse = loss_fn(Y_init, Y_true)
    final_mse = loss_fn(Y_final, Y_true)
    
    print(f"Initial MSE: {initial_mse.item():.6f}, Final MSE: {final_mse.item():.6f}")
    
    assert final_mse < initial_mse, "Optimization did not reduce the loss."
    assert final_mse < initial_mse * 0.1, "Optimization did not significantly reduce the loss."

@pytest.mark.large
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_very_large_noisy_optimization(device_str: str):
    """
    Benchmarks optimization on a large batch of complex expressions with noisy targets.
    This tests the optimizer's ability to find a best fit in a realistic scenario.
    """
    if not torch.cuda.is_available() and device_str == "cuda":
        raise ValueError("CUDA is not available on this system.")
    if device_str == "cpu":
        device = torch.device("cpu")
    else:
        index = get_cuda_device_with_min_memory()
        device_str = f"cuda:{index}"
        print(f"Using CUDA device: {device_str}")
        
    device = torch.device(device_str)
    B = 10000
    if device.type == 'cpu' and B >= 10000:
        pytest.skip("Skipping very large noisy benchmark on CPU to save time.")
        
    # --- Setup ---
    N, M, n_x = 100, 16, 4

    # Target function: y = c1*sin(X0+c2) + c3*log(|X1*X2 + c4|^2)
    # Using random true constants for each problem in the batch
    c1_true = torch.randn(B, 1, device=device)
    c2_true = torch.randn(B, 1, device=device)
    c3_true = torch.randn(B, 1, device=device)
    # Shift c4_true by a large amount to ensure the argument to log is always positive
    # and far from zero, avoiding numerical instability without needing to clamp.
    c4_true = torch.randn(B, 1, device=device) + 10.0
    
    X = torch.randn(N, n_x, device=device) * 2

    # Calculate clean Y_true, matching the symbolic expression exactly.
    # The symbolic model evaluates log(square(...)), not log(square(...) + eps).
    term_for_log = torch.square(X[:, 1].unsqueeze(0) * X[:, 2].unsqueeze(0) + c4_true)
    Y_true = (c1_true * torch.sin(X[:, 0].unsqueeze(0) + c2_true) + 
              c3_true * torch.log(term_for_log))
    
    # Add noise
    noise_level = 0.05 * torch.std(Y_true)
    Y_noisy = Y_true + torch.randn_like(Y_true) * noise_level

    # Expression in postfix: C1 X0 C2 + sin * C3 X1 X2 * C4 + SQUARE log * +
    Ops = torch.full((B, M), int(Operator.NO_OP), dtype=torch.int64, device=device)
    Ops[:, 0]  = int(Operator.LEARNABLE_CONSTANT) # C1
    Ops[:, 1]  = int(Operator.VAR_START_ID) + 0    # X0
    Ops[:, 2]  = int(Operator.LEARNABLE_CONSTANT) # C2
    Ops[:, 3]  = int(Operator.ADD)
    Ops[:, 4]  = int(Operator.SIN)
    Ops[:, 5]  = int(Operator.MUL)
    Ops[:, 6]  = int(Operator.LEARNABLE_CONSTANT) # C3
    Ops[:, 7]  = int(Operator.VAR_START_ID) + 1    # X1
    Ops[:, 8]  = int(Operator.VAR_START_ID) + 2    # X2
    Ops[:, 9]  = int(Operator.MUL)
    Ops[:, 10] = int(Operator.LEARNABLE_CONSTANT) # C4
    Ops[:, 11] = int(Operator.ADD)
    Ops[:, 12] = int(Operator.SQUARE)
    Ops[:, 13] = int(Operator.LOG)
    Ops[:, 14] = int(Operator.MUL)
    Ops[:, 15] = int(Operator.ADD)

    Ch = torch.full((B, M, 2), -1, dtype=torch.int64, device=device)
    Ch[:, 3, 0], Ch[:, 3, 1] = 1, 2
    Ch[:, 4, 0] = 3
    Ch[:, 5, 0], Ch[:, 5, 1] = 0, 4
    Ch[:, 9, 0], Ch[:, 9, 1] = 7, 8
    Ch[:, 11, 0], Ch[:, 11, 1] = 9, 10
    Ch[:, 12, 0] = 11
    Ch[:, 13, 0] = 12
    Ch[:, 14, 0], Ch[:, 14, 1] = 6, 13
    Ch[:, 15, 0], Ch[:, 15, 1] = 5, 14
    
    C_init = create_constants(Ops, method='randn')
    loss_fn_mse = nn.MSELoss()
    
    # --- Time the optimization ---
    start_time = time.time()
    _, Y_final = optimize(
        X, Y_noisy, Ops, Ch, C_init,
        line_search_fn='strong_wolfe',
        max_iter=300 # Increased iterations for this complex, noisy problem
    )
    if device.type == 'cuda':
        torch.cuda.synchronize(device=device)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\n[Very Large Noisy Benchmark] Device: {device_str}, Batch Size: {B}, Duration: {duration:.4f}s")
    
    # --- Assertions ---
    # We want to check that we recovered the original signal, so we calculate loss against Y_true
    with torch.no_grad():
        is_op = Ops != int(Operator.NO_OP)
        root_indices = is_op.sum(dim=1) - 1
        root_indices[root_indices < 0] = 0
        y_cache_init = evaluate(X, Ops, Ch, C_init)
        indices = root_indices.view(B, 1, 1).expand(B, 1, N)
        Y_init = y_cache_init.gather(1, indices).squeeze(1)
        initial_mse_vs_true = loss_fn_mse(Y_init, Y_true)
    
    final_mse_vs_true = loss_fn_mse(Y_final, Y_true)
    
    print(f"Initial MSE vs. True Signal: {initial_mse_vs_true.item():.6f}")
    print(f"Final MSE vs. True Signal: {final_mse_vs_true.item():.6f}")
    
    assert final_mse_vs_true < initial_mse_vs_true, "Optimization did not reduce the loss towards the true signal."
    assert final_mse_vs_true < 0.1, "Optimizer failed to recover the true signal from noise."
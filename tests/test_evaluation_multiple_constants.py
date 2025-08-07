# tests/test_evaluation_multiple_constants.py
import pytest
import torch

import symbolic_torch
from symbolic_torch import Operator
from symbolic_torch.evaluation import evaluate_backend
# Import the new backend function
from symbolic_torch._C import evaluate_multiple_constant_backend
from tests.utils import get_cuda_device_with_min_memory

NULL_CHILD = -1

@pytest.fixture(params=['cpu', 'cuda'])
def device(request: pytest.FixtureRequest) -> str:
    """Parametrized fixture for CPU and CUDA devices."""
    if request.param == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this system.")
    index = get_cuda_device_with_min_memory()
    return f"cuda:{index}" if request.param == "cuda" else "cpu"

# Test 1: Ensure expressions without constants are unaffected by the K dimension
def test_no_constant_expressions_multiple_sets(device: str):
    """
    Tests that for an expression without any LEARNABLE_CONSTANT, evaluating it with K different
    constant sets produces K identical outputs.
    """
    B, N, K, n_x = 4, 10, 10, 2
    
    # Expression: X_0 + X_1
    ops_tensor = torch.tensor([
        [int(Operator.VAR_START_ID)],      # M=0: X0
        [int(Operator.VAR_START_ID) + 1],  # M=1: X1
        [int(Operator.ADD)],               # M=2: X0 + X1
    ], dtype=torch.int64).expand(-1, B)
    M = ops_tensor.shape[0]

    ch_tensor = -torch.ones(M, B, 2, dtype=torch.int64)
    ch_tensor[2, :, 0] = 0
    ch_tensor[2, :, 1] = 1

    # C tensor with K different sets of random constants
    c_tensor_multiple = torch.randn(M, B, K, dtype=torch.float32)
    x_tensor = torch.randn(N, n_x, dtype=torch.float32)

    # Move all to the target device
    ops_tensor = ops_tensor.to(device)
    ch_tensor = ch_tensor.to(device)
    c_tensor_multiple = c_tensor_multiple.to(device)
    x_tensor = x_tensor.to(device)

    # Evaluate using the new function
    result_cache = evaluate_multiple_constant_backend(x_tensor, ops_tensor, ch_tensor, c_tensor_multiple)
    
    # Extract the final result for each of the K constant sets
    # Shape: (N, B, K)
    final_results = result_cache[-1]

    # All K results for any given batch item should be identical
    for b in range(B):
        first_result = final_results[:, b, 0]
        for k in range(1, K):
            other_result = final_results[:, b, k]
            assert torch.allclose(first_result, other_result), \
                f"Mismatch for batch item {b} between constant sets 0 and {k}"

# Test 2: Compare against looped single evaluation for simple expressions
unary_ops_params = [
    (Operator.SIN, "sin(C)"),
    (Operator.COS, "cos(C)"),
    (Operator.EXP, "exp(C)"),
    (Operator.LOG, "log(C)"),
    (Operator.SQRT, "sqrt(C)"),
    (Operator.TAN, "tan(C)"),
    (Operator.ARCSIN, "arcsin(C)"),
    (Operator.ARCCOS, "arccos(C)"),
    (Operator.ARCTAN, "arctan(C)"),
    (Operator.SINH, "sinh(C)"),
    (Operator.COSH, "cosh(C)"),
    (Operator.TANH, "tanh(C)"),
    (Operator.FLOOR, "floor(C)"),
    (Operator.CEIL, "ceil(C)"),
    (Operator.LN, "ln(C)"),
    (Operator.LOG10, "log10(C)"),
    (Operator.NEG, "neg(C)"),
    (Operator.INV, "inv(C)"),
    (Operator.CUBE, "cube(C)"),
    (Operator.FOURTH, "fourth(C)"),
    (Operator.FIFTH, "fifth(C)"),
]

@pytest.mark.parametrize("op, test_id", unary_ops_params, ids=[p[1] for p in unary_ops_params])
def test_unary_op_multiple_constants(device: str, op: Operator, test_id: str):
    """
    Compares evaluate_multiple_constant_backend with a loop of evaluate_backend for unary ops.
    Expression: op(C)
    """
    B, N, K = 8, 16, 10
    
    ops_tensor = torch.tensor([
        [int(Operator.LEARNABLE_CONSTANT)], # M=0: C
        [int(op)],                          # M=1: op(C)
    ], dtype=torch.int64).expand(-1, B)
    M = ops_tensor.shape[0]
    
    ch_tensor = -torch.ones(M, B, 2, dtype=torch.int64)
    ch_tensor[1, :, 0] = 0

    c_tensor_multiple = torch.randn(M, B, K, dtype=torch.float32)
    x_tensor = torch.randn(N, 1, dtype=torch.float32) # n_x=1, though not used

    # Move all to target device
    ops_tensor_dev = ops_tensor.to(device)
    ch_tensor_dev = ch_tensor.to(device)
    c_tensor_multiple_dev = c_tensor_multiple.to(device)
    x_tensor_dev = x_tensor.to(device)
    
    # Run new function
    result_multiple = evaluate_multiple_constant_backend(x_tensor_dev, ops_tensor_dev, ch_tensor_dev, c_tensor_multiple_dev)

    # Run old function in a loop
    looped_results : list[torch.Tensor] = []
    for k in range(K):
        c_k = c_tensor_multiple[:, :, k].to(device)
        result_k = evaluate_backend(x_tensor_dev, ops_tensor_dev, ch_tensor_dev, c_k)
        looped_results.append(result_k)
    
    result_looped = torch.stack(looped_results, dim=3) # Stack on a new 4th dim -> (M, N, B, K)

    assert torch.allclose(result_multiple, result_looped, atol=1e-6, equal_nan=True)

binary_ops_params = [
    (Operator.ADD, "X_0 + C"),
    (Operator.SUB, "X_0 - C"),
    (Operator.MUL, "X_0 * C"),
    (Operator.DIV, "X_0 / C"),
]
@pytest.mark.parametrize("op, test_id", binary_ops_params, ids=[p[1] for p in binary_ops_params])
def test_binary_op_multiple_constants(device: str, op: Operator, test_id: str):
    """
    Compares evaluate_multiple_constant_backend with a loop of evaluate_backend for binary ops.
    Expression: X_0 op C
    """
    B, N, K = 8, 16, 10
    
    ops_tensor = torch.tensor([
        [int(Operator.VAR_START_ID)],       # M=0: X_0
        [int(Operator.LEARNABLE_CONSTANT)], # M=1: C
        [int(op)],                          # M=2: X_0 op C
    ], dtype=torch.int64).expand(-1, B)
    M = ops_tensor.shape[0]

    ch_tensor = -torch.ones(M, B, 2, dtype=torch.int64)
    ch_tensor[2, :, 0] = 0
    ch_tensor[2, :, 1] = 1

    c_tensor_multiple = torch.randn(M, B, K, dtype=torch.float32)
    x_tensor = torch.randn(N, 1, dtype=torch.float32) # n_x=1

    # Move all to target device
    ops_tensor_dev = ops_tensor.to(device)
    ch_tensor_dev = ch_tensor.to(device)
    c_tensor_multiple_dev = c_tensor_multiple.to(device)
    x_tensor_dev = x_tensor.to(device)
    
    # Run new function
    result_multiple = evaluate_multiple_constant_backend(x_tensor_dev, ops_tensor_dev, ch_tensor_dev, c_tensor_multiple_dev)
    
    # Run old function in a loop
    looped_results : list[torch.Tensor] = []
    for k in range(K):
        c_k = c_tensor_multiple[:, :, k].to(device)
        result_k = evaluate_backend(x_tensor_dev, ops_tensor_dev, ch_tensor_dev, c_k)
        looped_results.append(result_k)
        
    result_looped = torch.stack(looped_results, dim=3)

    assert torch.allclose(result_multiple, result_looped, atol=1e-6, equal_nan=True)


# Test 3: Compare against looped single evaluation for complex expressions
def test_complex_expression_multiple_constants(device: str):
    """
    Compares evaluate_multiple_constant_backend with a loop of evaluate_backend for a complex expression.
    Expression: sin(X_0 * C_1) + C_2
    """
    B, N, K, n_x = 8, 16, 10, 1

    ops_tensor = torch.tensor([
        [int(Operator.VAR_START_ID)],       # M=0: X_0
        [int(Operator.LEARNABLE_CONSTANT)], # M=1: C_1
        [int(Operator.MUL)],                # M=2: X_0 * C_1
        [int(Operator.SIN)],                # M=3: sin(X_0 * C_1)
        [int(Operator.LEARNABLE_CONSTANT)], # M=4: C_2
        [int(Operator.ADD)],                # M=5: sin(...) + C_2
    ], dtype=torch.int64).expand(-1, B)
    M = ops_tensor.shape[0]
    
    ch_tensor = -torch.ones(M, B, 2, dtype=torch.int64)
    ch_tensor[2, :, 0], ch_tensor[2, :, 1] = 0, 1
    ch_tensor[3, :, 0] = 2
    ch_tensor[5, :, 0], ch_tensor[5, :, 1] = 3, 4
    
    c_tensor_multiple = torch.randn(M, B, K, dtype=torch.float32)
    x_tensor = torch.randn(N, n_x, dtype=torch.float32)

    # Move all to target device
    ops_tensor_dev = ops_tensor.to(device)
    ch_tensor_dev = ch_tensor.to(device)
    c_tensor_multiple_dev = c_tensor_multiple.to(device)
    x_tensor_dev = x_tensor.to(device)

    # Run new function
    result_multiple = evaluate_multiple_constant_backend(x_tensor_dev, ops_tensor_dev, ch_tensor_dev, c_tensor_multiple_dev)

    # Run old function in a loop
    looped_results : list[torch.Tensor] = []
    for k in range(K):
        c_k = c_tensor_multiple[:, :, k].to(device)
        result_k = evaluate_backend(x_tensor_dev, ops_tensor_dev, ch_tensor_dev, c_k)
        looped_results.append(result_k)

    result_looped = torch.stack(looped_results, dim=3)

    assert torch.allclose(result_multiple, result_looped, atol=1e-6, equal_nan=True)

    # Also test CPU vs GPU match if on CUDA
    if device == "cuda":
        result_multiple_cpu = evaluate_multiple_constant_backend(
            x_tensor.to("cpu"), ops_tensor.to("cpu"), ch_tensor.to("cpu"), c_tensor_multiple.to("cpu")
        )
        assert torch.allclose(result_multiple.cpu(), result_multiple_cpu, atol=1e-6, equal_nan=True)

# Test 4: Verify gradients
def test_gradients_complex_expression_multiple_constants(device: str):
    """
    Compares the gradients from evaluate_multiple_constant_backend with the accumulated
    gradients from a loop of evaluate_backend calls.
    """
    B, N, K, n_x = 4, 8, 5, 1
    ops_tensor = torch.tensor([
        [int(Operator.VAR_START_ID)],
        [int(Operator.LEARNABLE_CONSTANT)],
        [int(Operator.MUL)],
        [int(Operator.SIN)],
        [int(Operator.LEARNABLE_CONSTANT)],
        [int(Operator.ADD)],
    ], dtype=torch.int64).expand(-1, B)
    M = ops_tensor.shape[0]

    ch_tensor = -torch.ones(M, B, 2, dtype=torch.int64)
    ch_tensor[2, :, 0], ch_tensor[2, :, 1] = 0, 1
    ch_tensor[3, :, 0] = 2
    ch_tensor[5, :, 0], ch_tensor[5, :, 1] = 3, 4
    
    # Base tensors for comparison
    x_tensor_base = torch.randn(N, n_x, device=device)
    c_tensor_multiple_base = torch.randn(M, B, K, device=device)
    grad_output = torch.randn(M, N, B, K, device=device)

    # --- Method A: New function ---
    x_new = x_tensor_base.clone().requires_grad_(True)
    c_new = c_tensor_multiple_base.clone().requires_grad_(True)
    result_multiple = evaluate_multiple_constant_backend(x_new, ops_tensor.to(device), ch_tensor.to(device), c_new)
    result_multiple.backward(grad_output)

    # --- Method B: Looped old function ---
    x_looped = x_tensor_base.clone().requires_grad_(True)
    c_looped = c_tensor_multiple_base.clone().requires_grad_(True)
    
    # Manually accumulate gradients
    for k in range(K):
        c_k = c_looped[:, :, k]
        grad_k = grad_output[:, :, :, k]
        
        # We need to clone inputs for each backward pass in the loop
        # if we were not accumulating into the same .grad attributes.
        # But since we are, PyTorch handles accumulation correctly.
        result_k = evaluate_backend(x_looped, ops_tensor.to(device), ch_tensor.to(device), c_k)
        result_k.backward(grad_k)

    # --- Assertions ---
    assert x_new.grad is not None and x_looped.grad is not None, "Gradients for X should not be None"
    assert c_new.grad is not None and c_looped.grad is not None, "Gradients for C should not be None"

    # The gradient for X should be summed over the K dimension
    assert torch.allclose(x_new.grad, x_looped.grad, atol=1e-5), "Gradients for X do not match"
    
    # The gradient for C should match across all K sets
    assert torch.allclose(c_new.grad, c_looped.grad, atol=1e-5), "Gradients for C do not match"
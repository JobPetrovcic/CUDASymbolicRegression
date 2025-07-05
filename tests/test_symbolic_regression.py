import pytest
import torch
from torch import Tensor
import time
from typing import List, Optional

# Import the compiled extension
import symbolic_torch
#torch.autograd.set_detect_anomaly(True)

# Define operators from operators.h
NO_OP: int = 0
LEARNABLE_CONSTANT: int = 1
CONST_1: int = 2
SIN: int = 6
COS: int = 7
EXP: int = 8
LOG: int = 9
SQUARE: int = 10
SQRT: int = 11
ADD: int = 12
SUB: int = 13
MUL: int = 14
DIV: int = 15
VAR_START_ID: int = 16

unary_ops: List[int] = [SIN, COS, EXP, LOG, SQUARE, SQRT]
binary_ops: List[int] = [ADD, SUB, MUL, DIV]
all_ops: List[int] = unary_ops + binary_ops

def get_arity(op : int) -> int:
    if op in unary_ops:
        return 1
    if op in binary_ops:
        return 2
    return 0

def build_and_run(x_tensor: Tensor, ops_tensor: Tensor, ch_tensor: Tensor, c_tensor: Tensor, device: str = 'cpu') -> Tensor:
    x_tensor = x_tensor.to(device)
    ops_tensor = ops_tensor.to(device)
    ch_tensor = ch_tensor.to(device)
    c_tensor = c_tensor.to(device)
    
    y = symbolic_torch.evaluate(x_tensor, ops_tensor, ch_tensor, c_tensor)
    return y

def build_and_run_with_grad(x_tensor: Tensor, ops_tensor: Tensor, ch_tensor: Tensor, c_tensor: Tensor, grad_output : Tensor, null_nan_output_grad : bool, device: str = 'cpu', use_custom_kernel: bool = True):
    x_tensor = x_tensor.clone().detach().to(device).requires_grad_(True)
    ops_tensor = ops_tensor.to(device)
    ch_tensor = ch_tensor.to(device)
    c_tensor = c_tensor.clone().detach().to(device).requires_grad_(True)

    if use_custom_kernel:
        y = symbolic_torch.evaluate(x_tensor, ops_tensor, ch_tensor, c_tensor)
    else:
        y = manual_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    
    # Use the last node's output for grad calculation
    output = y[-1]
    # check that grad_output is 0 where output is NaN
    grad_output = grad_output.to(device)

    print(f"Grad output shape: {grad_output.shape}, Output shape: {output.shape}, ops_tensor shape: {ops_tensor.shape}, ch_tensor shape: {ch_tensor.shape}, c_tensor shape: {c_tensor.shape}")

    if null_nan_output_grad:
        grad_output[torch.isnan(output)] = 0
    
    output.backward(gradient=grad_output)
    
    return x_tensor.grad, c_tensor.grad

def manual_run(x_tensor: Tensor, ops_tensor: Tensor, ch_tensor: Tensor, c_tensor: Tensor, device: str = 'cpu') -> Tensor:
    M, B = ops_tensor.shape
    N, _ = x_tensor.shape
    
    x_tensor = x_tensor.to(device)
    ops_tensor = ops_tensor.to(device)
    ch_tensor = ch_tensor.to(device)
    c_tensor = c_tensor.to(device)

    cache_list: List[Tensor] = []
    
    # Map learnable constant indices
    posC: Tensor = -torch.ones_like(ops_tensor)
    const_indices = (ops_tensor == LEARNABLE_CONSTANT).nonzero(as_tuple=True)
    posC[const_indices] = torch.arange(len(const_indices[0]), device=device)

    # WARNING: We do NOT add epsilon for stability here, as we want it to match the nan output of the C++ kernel.
    # WARNING: We also do not use abs() here as we want to match the behavior of the C++ kernel.
    for k in range(M):
        # For each node in the expression tree
        batch_results: List[Tensor] = []
        for b in range(B):
            # For each expression in the batch
            op = int(ops_tensor[k, b].item())

            if op == NO_OP:
                # Append a column of NaNs for this expression in the batch
                batch_results.append(torch.full((N, 1), float('nan'), dtype=x_tensor.dtype, device=device))
                continue

            arity = get_arity(int(op))
            arg0: Optional[Tensor] = None
            arg1: Optional[Tensor] = None
            if arity >= 1:
                ch0_idx = ch_tensor[k, b, 0].item()
                if ch0_idx != -1:
                    arg0 = cache_list[int(ch0_idx)][:, b]
            if arity == 2:
                ch1_idx = ch_tensor[k, b, 1].item()
                if ch1_idx != -1:
                    arg1 = cache_list[int(ch1_idx)][:, b]

            res: Optional[Tensor] = None
            if op == LEARNABLE_CONSTANT:
                c_idx = posC[k, b].item()
                if c_idx != -1:
                    res = c_tensor[int(c_idx)].expand(N)
            elif op == CONST_1:
                res = torch.full((N,), 1.0, dtype=x_tensor.dtype, device=device)
            elif op >= VAR_START_ID:
                var_idx = op - VAR_START_ID
                res = x_tensor[:, int(var_idx)]
            elif arity > 0 and arg0 is not None:
                if op == SIN:
                    res = torch.sin(arg0)
                elif op == COS:
                    res = torch.cos(arg0)
                elif op == EXP:
                    res = torch.exp(arg0)
                elif op == LOG:
                    res = torch.log(arg0)
                elif op == SQUARE:
                    res = arg0 * arg0
                elif op == SQRT:
                    res = torch.sqrt(arg0)
                elif arity > 1 and arg1 is not None:
                    if op == ADD:
                        res = arg0 + arg1
                    elif op == SUB:
                        res = arg0 - arg1
                    elif op == MUL:
                        res = arg0 * arg1
                    elif op == DIV:
                        res = arg0 / arg1
            
            if res is not None:
                batch_results.append(res.view(N, 1))
            else:
                batch_results.append(torch.full((N, 1), float('nan'), dtype=x_tensor.dtype, device=device))

        # Concatenate results from all expressions in the batch for the current node
        current_output = torch.cat(batch_results, dim=1)
        cache_list.append(current_output)

    return torch.stack(cache_list, dim=0)

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_noop_nan(device : str) -> None:
    # If noop is used, the output should be NaN
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    ops_tensor = torch.tensor([[NO_OP]], dtype=torch.int64)
    ch_tensor = -torch.ones(1, 1, 2, dtype=torch.int64)
    c_tensor = symbolic_torch.create_constants(ops_tensor)
    custom_res = build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    assert torch.isnan(custom_res).all(), "Expected output to be NaN when using NO_OP"

    N = x_tensor.shape[0]
    B = ops_tensor.shape[-1]

    grad_output = torch.zeros(N, B)
    # Test gradients
    x_grad, c_grad = build_and_run_with_grad(x_tensor, ops_tensor, ch_tensor, c_tensor, grad_output, null_nan_output_grad=True, device=device, use_custom_kernel=True)
    assert x_grad is not None, "Expected x_grad to be a tensor"
    assert torch.all(x_grad == 0), "Expected x_grad to be zero"
    assert c_grad is not None, "Expected c_grad to be a tensor"
    assert torch.all(c_grad == 0), "Expected c_grad to be zero"


@pytest.fixture(params=['cpu', 'cuda'])
def device(request: pytest.FixtureRequest) -> str:
    return request.param

# 1. Test validate_inputs
def test_validate_inputs_arity(device : str) -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    for op in all_ops:
        arity = get_arity(op)
        
        # We need at least one node before the operator to act as a child
        ops_tensor = torch.zeros(1 + arity, 1, dtype=torch.int64)
        ch_tensor = -torch.ones(1 + arity, 1, 2, dtype=torch.int64)

        # Set up dummy child nodes
        for i in range(arity):
            ops_tensor[i, 0] = VAR_START_ID + i
        
        # Set the operator
        ops_tensor[arity, 0] = op

        # Set correct number of children for the operator
        for i in range(arity):
            ch_tensor[arity, 0, i] = i
            
        c_tensor = symbolic_torch.create_constants(ops_tensor)
        # This should not raise an error
        build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)

        # Set one too many children
        if arity < 2:
            ch_tensor[arity, 0, arity] = 0
            with pytest.raises(RuntimeError, match="Input validation failed"):
                build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
        
        if arity == 2:
            # For binary ops, test with one too few children
            ch_tensor_one_less = ch_tensor.clone()
            ch_tensor_one_less[arity, 0, arity - 1] = -1
            with pytest.raises(RuntimeError, match="Input validation failed"):
                build_and_run(x_tensor, ops_tensor, ch_tensor_one_less, c_tensor, device=device)
        elif arity == 1:
            # For unary ops, test with one too few children.
            # This is only an error if the result is used by another node.
            # M=0: VAR_START_ID
            # M=1: op (unary) with invalid child
            # M=2: ADD(M=0, M=1)
            ops_tensor_unary_fail = torch.tensor([
                [VAR_START_ID],
                [op],
                [ADD]
            ], dtype=torch.int64)
            ch_tensor_unary_fail = -torch.ones(3, 1, 2, dtype=torch.int64)
            ch_tensor_unary_fail[1, 0, 0] = -1 # Invalid child for the unary op
            ch_tensor_unary_fail[2, 0, 0] = 0
            ch_tensor_unary_fail[2, 0, 1] = 1
            
            c_tensor_unary_fail = symbolic_torch.create_constants(ops_tensor_unary_fail)
            with pytest.raises(RuntimeError, match="Input validation failed"):
                build_and_run(x_tensor, ops_tensor_unary_fail, ch_tensor_unary_fail, c_tensor_unary_fail, device=device)

def test_validate_inputs_child_index_too_large(device : str) -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    ops_tensor = torch.tensor([[VAR_START_ID], [ADD]], dtype=torch.int64)
    ch_tensor = -torch.ones(2, 1, 2, dtype=torch.int64)
    ch_tensor[1, 0, 0] = 0
    ch_tensor[1, 0, 1] = 2 # Invalid, should be < k=1
    c_tensor = symbolic_torch.create_constants(ops_tensor)
    with pytest.raises(RuntimeError, match="Input validation failed"):
        build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)

def test_validate_inputs_unary_op_no_children(device: str) -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    for op in unary_ops:
        ops_tensor = torch.tensor([[op]], dtype=torch.int64)
        ch_tensor = -torch.ones(1, 1, 2, dtype=torch.int64)
        c_tensor = symbolic_torch.create_constants(ops_tensor)
        with pytest.raises(RuntimeError, match="Input validation failed"):
            build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)

def test_validate_inputs_binary_op_no_children(device: str) -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    for op in binary_ops:
        ops_tensor = torch.tensor([[op]], dtype=torch.int64)
        ch_tensor = -torch.ones(1, 1, 2, dtype=torch.int64)
        c_tensor = symbolic_torch.create_constants(ops_tensor)
        with pytest.raises(RuntimeError, match="Input validation failed"):
            build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)

# 2. Test datatype checking
def test_wrong_datatypes(device : str) -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    ops_tensor = torch.tensor([[VAR_START_ID]], dtype=torch.int64)
    ch_tensor = -torch.ones(1, 1, 2, dtype=torch.int64)
    c_tensor = torch.randn(0, dtype=torch.float32)

    with pytest.raises(RuntimeError, match='X must be a floating-point tensor'):
        build_and_run(x_tensor.to(torch.int32), ops_tensor, ch_tensor, c_tensor, device=device)
    with pytest.raises(RuntimeError, match="must be a Long tensor"):
        build_and_run(x_tensor, ops_tensor.to(torch.float32), ch_tensor, c_tensor, device=device)
    with pytest.raises(RuntimeError, match="must be a Long tensor"):
        build_and_run(x_tensor, ops_tensor, ch_tensor.to(torch.float32), c_tensor, device=device)
    with pytest.raises(RuntimeError, match='C must be a floating-point tensor'):
        build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor.to(torch.int32), device=device)

# 3. Test individual operations
@pytest.mark.parametrize("op", all_ops)
def test_single_op(device : str, op : int) -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    
    arity = get_arity(op)
    
    ops_tensor = torch.zeros(1 + arity, 1, dtype=torch.int64)
    ch_tensor = -torch.ones(1 + arity, 1, 2, dtype=torch.int64)

    for i in range(arity):
        ops_tensor[i, 0] = VAR_START_ID + i
        ch_tensor[arity, 0, i] = i
    ops_tensor[arity, 0] = op

    c_tensor = symbolic_torch.create_constants(ops_tensor)
    custom_res = build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    manual_res = manual_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    assert torch.allclose(custom_res, manual_res, atol=1e-6, equal_nan=True)

# 4. Test batch processing
def test_batching(device : str) -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    # Tree 1: X0 + X1
    # Tree 2: sin(X0)
    ops_tensor = torch.tensor([
        [VAR_START_ID, VAR_START_ID],
        [VAR_START_ID+1, NO_OP],
        [ADD, SIN]
    ], dtype=torch.int64)
    ch_tensor = -torch.ones(3, 2, 2, dtype=torch.int64)
    ch_tensor[2, 0, 0] = 0
    ch_tensor[2, 0, 1] = 1
    ch_tensor[2, 1, 0] = 0

    c_tensor = symbolic_torch.create_constants(ops_tensor)
    custom_res = build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    manual_res = manual_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    assert custom_res.shape == (3, 10, 2)
    assert torch.allclose(custom_res, manual_res, atol=1e-6, equal_nan=True)

# 5. Test CPU vs GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_gpu_match() -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    ops_tensor = torch.tensor([[VAR_START_ID], [VAR_START_ID+1], [ADD]], dtype=torch.int64)
    ch_tensor = -torch.ones(3, 1, 2, dtype=torch.int64)
    ch_tensor[2, 0, 0] = 0
    ch_tensor[2, 0, 1] = 1
    c_tensor = symbolic_torch.create_constants(ops_tensor)
    
    cpu_res = build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device='cpu')
    gpu_res = build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device='cuda')
    
    assert torch.allclose(cpu_res, gpu_res.cpu(), atol=1e-6, equal_nan=True)


# 6. Test complex expression: sin(X1) + X0 * X1
def test_complex_expression(device : str) -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    # M=0: X0
    # M=1: X1
    # M=2: sin(X1) (sin(M=1))
    # M=3: X0 * X1 (M=0 * M=1)
    # M=4: sin(X1) + X0*X1 (M=2 + M=3)
    ops_tensor = torch.tensor([
        [VAR_START_ID],
        [VAR_START_ID + 1],
        [SIN],
        [MUL],
        [ADD]
    ], dtype=torch.int64)
    ch_tensor = -torch.ones(5, 1, 2, dtype=torch.int64)
    ch_tensor[2, 0, 0] = 1 # sin(X1)
    ch_tensor[3, 0, 0] = 0 # X0 * X1
    ch_tensor[3, 0, 1] = 1
    ch_tensor[4, 0, 0] = 2 # op + op
    ch_tensor[4, 0, 1] = 3

    c_tensor = symbolic_torch.create_constants(ops_tensor)
    custom_res = build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    manual_res = manual_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    assert torch.allclose(custom_res, manual_res, atol=1e-6, equal_nan=True)

# 7. Test constants
def test_constants(device : str) -> None:
    x_tensor = torch.randn(4, 1, dtype=torch.float32)
    # M=0: C0 (learnable)
    # M=1: 1 (const)
    # M=2: C0 + 1
    ops_tensor = torch.tensor([
        [LEARNABLE_CONSTANT],
        [CONST_1],
        [ADD]
    ], dtype=torch.int64)
    ch_tensor = -torch.ones(3, 1, 2, dtype=torch.int64)
    ch_tensor[2, 0, 0] = 0
    ch_tensor[2, 0, 1] = 1
    c_tensor = torch.tensor([3.14], dtype=torch.float32).to(device)

    custom_res = build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    manual_res = manual_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    assert torch.allclose(custom_res, manual_res, atol=1e-6)
    # The output shape is (M, N, B), which is (3, 4, 1) here.
    # We need to construct a tensor of the same shape to compare against.
    expected_res = torch.stack([
        c_tensor.expand(4, 1),
        torch.ones(4, 1, device=device) * 1.0,
        c_tensor.expand(4, 1) + 1.0
    ], dim=0).view(3, 4, 1)
    assert torch.allclose(custom_res.to(device), expected_res.to(device), atol=1e-6)

    # Test sin(C)
    ops_tensor = torch.tensor([[LEARNABLE_CONSTANT], [SIN]], dtype=torch.int64)
    ch_tensor = -torch.ones(2, 1, 2, dtype=torch.int64)
    ch_tensor[1, 0, 0] = 0
    c_tensor = torch.tensor([3.14], dtype=torch.float32).to(device)
    
    custom_res = build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    manual_res = manual_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    assert torch.allclose(custom_res, manual_res, atol=1e-6)
    expected_res_sin = torch.stack([
        c_tensor.expand(4, 1),
        torch.sin(c_tensor).expand(4, 1)
    ], dim=0).view(2, 4, 1)
    assert torch.allclose(custom_res.to(device), expected_res_sin.to(device), atol=1e-6)

# 8. Test Gradients
@pytest.mark.parametrize("op", all_ops)
def test_gradients(device: str, op: int) -> None:
    arity = get_arity(op)
    x_tensor = torch.randn(10, arity if arity > 0 else 1, dtype=torch.float32)
    
    ops_tensor = torch.zeros(1 + arity, 1, dtype=torch.int64)
    ch_tensor = -torch.ones(1 + arity, 1, 2, dtype=torch.int64)
    c_tensor = symbolic_torch.create_constants(ops_tensor).to(device)

    for i in range(arity):
        ops_tensor[i, 0] = VAR_START_ID + i
        ch_tensor[arity, 0, i] = i
    ops_tensor[arity, 0] = op

    # Special case for log to test negative inputs
    #if op == LOG:
    #    x_tensor = torch.abs(x_tensor) * -1 # Ensure negative inputs for log

    N = x_tensor.shape[0]
    B = ops_tensor.shape[-1]

    grad_output = torch.randn(N, B, dtype=torch.float32)
    custom_x_grad, custom_c_grad = build_and_run_with_grad(x_tensor, ops_tensor, ch_tensor, c_tensor, grad_output, null_nan_output_grad=True, device=device, use_custom_kernel=True)

    #print(f"Testing gradients for operator {op} with arity {arity}")
    #print(f"grad_output: {grad_output}")
    #print(f"x_tensor: {x_tensor}")
    #print(f"ops_tensor: {ops_tensor}")
    #print(f"ch_tensor: {ch_tensor}")
    #print(f"c_tensor: {c_tensor}")
    #print(f"custom_x_grad: {custom_x_grad}")

    manual_x_grad, manual_c_grad = build_and_run_with_grad(x_tensor, ops_tensor, ch_tensor, c_tensor, grad_output, null_nan_output_grad=True, device=device, use_custom_kernel=False)

    assert custom_x_grad is not None and manual_x_grad is not None, "Gradients should not be None"
    assert torch.allclose(custom_x_grad, manual_x_grad, atol=1e-5, equal_nan=True)
    if custom_c_grad is not None and manual_c_grad is not None:
        assert torch.allclose(custom_c_grad, manual_c_grad, atol=1e-5, equal_nan=True)

# 9. Benchmark
#def test_benchmark() -> None:
#    B, N, n_x = 4024, 1024, 4
#    x_tensor = torch.randn(N, n_x, dtype=torch.float32)
#    
#    # A moderately complex expression
#    # (X0 + X1) * sin(X2 - C)
#    ops_tensor = torch.tensor([
#        [VAR_START_ID],
#        [VAR_START_ID + 1],
#        [VAR_START_ID + 2],
#        [LEARNABLE_CONSTANT],
#        [ADD], # X0+X1
#        [SUB], # X2-C
#        [SIN], # sin(X2-C)
#        [MUL]  # (X0+X1)*sin(X2-C)
#    ], dtype=torch.int64).expand(-1, B)
#    
#    ch_tensor = -torch.ones(8, B, 2, dtype=torch.int64)
#    ch_tensor[4, :, 0] = 0
#    ch_tensor[4, :, 1] = 1
#    ch_tensor[5, :, 0] = 2
#    ch_tensor[5, :, 1] = 3
#    ch_tensor[6, :, 0] = 5
#    ch_tensor[7, :, 0] = 4
#    ch_tensor[7, :, 1] = 6
#
#    c_tensor = torch.randn(B, dtype=torch.float32)
#
#    # Manual PyTorch
#    start_time = time.time()
#    manual_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device='cpu')
#    manual_time = time.time() - start_time
#    print(f"\nManual PyTorch (CPU): {manual_time:.4f}s")
#
#    # Custom CPU
#    start_time = time.time()
#    build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device='cpu')
#    cpu_time = time.time() - start_time
#    print(f"Custom C++ (CPU): {cpu_time:.4f}s")
#
#    if torch.cuda.is_available():
#        # Custom GPU
#        start_time = time.time()
#        build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device='cuda')
#        torch.cuda.synchronize()
#        gpu_time = time.time() - start_time
#        print(f"Custom C++ (GPU): {gpu_time:.4f}s")
#
## 10. Larger Benchmark
#def test_large_benchmark() -> None:
#    B, N, n_x = 8192, 2048, 8
#    x_tensor = torch.randn(N, n_x, dtype=torch.float32)
#
#    # A more complex expression:
#    # log( (sin(X0) + cos(X1)) * (exp(X2) - sqrt(X3)) ) + (X4 * X5) - (X6 / X7)
#    ops_tensor = torch.tensor([
#        [VAR_START_ID + 0],      # M=0: X0
#        [VAR_START_ID + 1],      # M=1: X1
#        [VAR_START_ID + 2],      # M=2: X2
#        [VAR_START_ID + 3],      # M=3: X3
#        [VAR_START_ID + 4],      # M=4: X4
#        [VAR_START_ID + 5],      # M=5: X5
#        [VAR_START_ID + 6],      # M=6: X6
#        [VAR_START_ID + 7],      # M=7: X7
#        [SIN],                   # M=8: sin(X0)
#        [COS],                   # M=9: cos(X1)
#        [EXP],                   # M=10: exp(X2)
#        [SQRT],                  # M=11: sqrt(X3)
#        [ADD],                   # M=12: sin(X0) + cos(X1)
#        [SUB],                   # M=13: exp(X2) - sqrt(X3)
#        [MUL],                   # M=14: (sin(X0) + cos(X1)) * (exp(X2) - sqrt(X3))
#        [LOG],                   # M=15: log(...)
#        [MUL],                   # M=16: X4 * X5
#        [DIV],                   # M=17: X6 / X7
#        [ADD],                   # M=18: log(...) + (X4 * X5)
#        [SUB]                    # M=19: ... - (X6 / X7)
#    ], dtype=torch.int64).expand(-1, B)
#
#    ch_tensor = -torch.ones(20, B, 2, dtype=torch.int64)
#    ch_tensor[8, :, 0] = 0   # sin(X0)
#    ch_tensor[9, :, 0] = 1   # cos(X1)
#    ch_tensor[10, :, 0] = 2  # exp(X2)
#    ch_tensor[11, :, 0] = 3  # sqrt(X3)
#    ch_tensor[12, :, 0] = 8  # M=8 + M=9
#    ch_tensor[12, :, 1] = 9
#    ch_tensor[13, :, 0] = 10 # M=10 - M=11
#    ch_tensor[13, :, 1] = 11
#    ch_tensor[14, :, 0] = 12 # M=12 * M=13
#    ch_tensor[14, :, 1] = 13
#    ch_tensor[15, :, 0] = 14 # log(M=14)
#    ch_tensor[16, :, 0] = 4  # M=4 * M=5
#    ch_tensor[16, :, 1] = 5
#    ch_tensor[17, :, 0] = 6  # M=6 / M=7
#    ch_tensor[17, :, 1] = 7
#    ch_tensor[18, :, 0] = 15 # M=15 + M=16
#    ch_tensor[18, :, 1] = 16
#    ch_tensor[19, :, 0] = 18 # M=18 - M=17
#    ch_tensor[19, :, 1] = 17
#
#    # Manual PyTorch
#    start_time = time.time()
#    manual_run(x_tensor, ops_tensor, ch_tensor, device='cpu')
#    manual_time = time.time() - start_time
#    print(f"\nLarge Benchmark - Manual PyTorch (CPU): {manual_time:.4f}s")
#
#    # Custom CPU
#    start_time = time.time()
#    build_and_run(x_tensor, ops_tensor, ch_tensor, device='cpu')
#    cpu_time = time.time() - start_time
#    print(f"Large Benchmark - Custom C++ (CPU): {cpu_time:.4f}s")
#
#    if torch.cuda.is_available():
#        # Custom GPU
#        start_time = time.time()
#        build_and_run(x_tensor, ops_tensor, ch_tensor, device='cuda')
#        torch.cuda.synchronize()
#        gpu_time = time.time() - start_time
#        print(f"Large Benchmark - Custom C++ (GPU): {gpu_time:.4f}s")

# 10. Test error handling for gradients on invalid operations
@pytest.mark.parametrize("op", [
    NO_OP,
    LOG,
    SQRT,
    DIV,
])
def test_gradient_on_invalid_op(op: int, device: str):
    print(f"Testing gradient on invalid operation: {op}")
    B, N, n_x = 1, 1, 2
    x_tensor = torch.ones(N, n_x, dtype=torch.float32)
    c_tensor = torch.ones((0, ), dtype=torch.float32)  # No constants needed for this test
    grad_output = torch.ones(N, B, dtype=torch.float32)

    if op == LOG:
        # log(0)
        ops_tensor = torch.tensor([[VAR_START_ID], [LOG]], dtype=torch.int64).expand(-1, B)
        ch_tensor = -torch.ones(2, B, 2, dtype=torch.int64)
        ch_tensor[1, :, 0] = 0
        x_tensor[0, 0] = 0
    elif op == SQRT:
        # sqrt(-1)
        ops_tensor = torch.tensor([[VAR_START_ID], [SQRT]], dtype=torch.int64).expand(-1, B)
        ch_tensor = -torch.ones(2, B, 2, dtype=torch.int64)
        ch_tensor[1, :, 0] = 0
        x_tensor[0, 0] = -1
    elif op == DIV:
        # 1/0
        ops_tensor = torch.tensor([[CONST_1], [VAR_START_ID], [DIV]], dtype=torch.int64).expand(-1, B)
        ch_tensor = -torch.ones(3, B, 2, dtype=torch.int64)
        ch_tensor[2, :, 0] = 0
        ch_tensor[2, :, 1] = 1
        x_tensor[0, 0] = 0
    elif op == NO_OP:
        ops_tensor = torch.tensor([[NO_OP]], dtype=torch.int64).expand(-1, B)
        ch_tensor = -torch.ones(1, B, 2, dtype=torch.int64)
    else:
        raise ValueError(f"Unsupported operator for gradient test: {op}")
    
    with pytest.raises(RuntimeError) as excinfo:
        build_and_run_with_grad(x_tensor, ops_tensor, ch_tensor, c_tensor, grad_output, null_nan_output_grad=False, device=device)

    # Check if the error message contains one of the expected substrings
    error_str = str(excinfo.value)
    # TODO: Update this to match the actual error message from your C++ kernel
    assert "Backward error: Gradient propagated to a" in error_str 


if __name__ == "__main__":
    pytest.main()

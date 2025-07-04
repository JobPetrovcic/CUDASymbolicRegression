import torch
from torch import Tensor
import time
import pytest
from typing import List, Optional

# Import the compiled extension
import symbolic_torch

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

def build_and_run(x_tensor: Tensor, ops_tensor: Tensor, ch_tensor: Tensor, c_tensor: Optional[Tensor] = None, device: str = 'cpu') -> Tensor:
    x_tensor = x_tensor.to(device)
    ops_tensor = ops_tensor.to(device)
    ch_tensor = ch_tensor.to(device)
    if c_tensor is not None:
        c_tensor = c_tensor.to(device)
    
    # The custom operator is the second element of the tuple
    y, _ = symbolic_torch.evaluate(x_tensor, ops_tensor, ch_tensor, c_tensor)
    return y

def manual_run(x_tensor: Tensor, ops_tensor: Tensor, ch_tensor: Tensor, c_tensor: Optional[Tensor] = None, device: str = 'cpu') -> Tensor:
    M, B = ops_tensor.shape
    N, _ = x_tensor.shape
    
    x_tensor = x_tensor.to(device)
    ops_tensor = ops_tensor.to(device)
    ch_tensor = ch_tensor.to(device)
    if c_tensor is not None:
        c_tensor = c_tensor.to(device)

    cache: Tensor = torch.zeros(M, B, N, dtype=x_tensor.dtype, device=device)
    
    # Map learnable constant indices
    posC: Tensor = -torch.ones_like(ops_tensor)
    if c_tensor is not None:
        c_idx_counter = 0
        for m in range(M):
            for b in range(B):
                if ops_tensor[m, b] == LEARNABLE_CONSTANT:
                    posC[m,b] = c_idx_counter
                    c_idx_counter += 1

    for k in range(M):
        for b in range(B):
            op: int = ops_tensor[k, b].item()
            if op == NO_OP:
                continue

            args: List[Tensor] = []
            for i in range(get_arity(op)):
                child_k: int = ch_tensor[k, b, i].item()
                args.append(cache[child_k, b, :])

            result: Tensor
            if op >= VAR_START_ID:
                var_idx: int = op - VAR_START_ID
                result = x_tensor[:, var_idx]
            elif op == LEARNABLE_CONSTANT:
                c_idx: int = int(posC[k,b].item())
                if c_tensor is not None:
                    result = torch.full((N,), c_tensor[c_idx].item(), device=device)
                else:
                    result = torch.zeros(N, device=device)
            elif op == CONST_1:
                result = torch.ones(N, device=device)
            elif op == SIN:
                result = torch.sin(args[0])
            elif op == COS:
                result = torch.cos(args[0])
            elif op == EXP:
                result = torch.exp(args[0])
            elif op == LOG:
                result = torch.log(args[0])
            elif op == SQUARE:
                result = args[0]**2
            elif op == SQRT:
                result = torch.sqrt(args[0])
            elif op == ADD:
                result = args[0] + args[1]
            elif op == SUB:
                result = args[0] - args[1]
            elif op == MUL:
                result = args[0] * args[1]
            elif op == DIV:
                result = args[0] / args[1]
            else:
                result = torch.zeros(N, device=device)
            
            cache[k, b, :] = result

    return cache[M - 1, :, :].squeeze(0)


@pytest.fixture(params=['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'])
def device(request: pytest.FixtureRequest) -> str:
    return request.param

# 1. Test validate_inputs
def test_validate_inputs_arity(device : str) -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    for op in all_ops:
        arity = get_arity(op)
        # Test with too few children
        ops_tensor = torch.tensor([[op]], dtype=torch.int64)
        ch_tensor = -torch.ones(1, 1, 2, dtype=torch.int64)
        if arity > 0:
            with pytest.raises(RuntimeError, match="Input validation failed"):
                 build_and_run(x_tensor, ops_tensor, ch_tensor, device=device)
        
        # Test with too many children
        ch_tensor = torch.zeros(1, 1, 2, dtype=torch.int64)
        if arity < 2:
             with pytest.raises(RuntimeError, match="Input validation failed"):
                 build_and_run(x_tensor, ops_tensor, ch_tensor, device=device)

def test_validate_inputs_child_index_too_large(device : str) -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    ops_tensor = torch.tensor([[VAR_START_ID], [ADD]], dtype=torch.int64)
    ch_tensor = -torch.ones(2, 1, 2, dtype=torch.int64)
    ch_tensor[1, 0, 0] = 0
    ch_tensor[1, 0, 1] = 2 # Invalid, should be < k=1
    with pytest.raises(RuntimeError, match="Input validation failed"):
        build_and_run(x_tensor, ops_tensor, ch_tensor, device=device)

# 2. Test datatype checking
def test_wrong_datatypes(device : str) -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    ops_tensor = torch.tensor([[VAR_START_ID]], dtype=torch.int64)
    ch_tensor = -torch.ones(1, 1, 2, dtype=torch.int64)
    c_tensor = torch.randn(1, dtype=torch.float32)

    with pytest.raises(RuntimeError, match='X must be a floating-point tensor'):
        build_and_run(x_tensor.to(torch.int32), ops_tensor, ch_tensor, device=device)
    with pytest.raises(RuntimeError, match="must be a Long tensor"):
        build_and_run(x_tensor, ops_tensor.to(torch.float32), ch_tensor, device=device)
    with pytest.raises(RuntimeError, match="must be a Long tensor"):
        build_and_run(x_tensor, ops_tensor, ch_tensor.to(torch.float32), device=device)
    with pytest.raises(RuntimeError, match='C must be a floating-point tensor'):
        build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor.to(torch.int64), device=device)

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

    custom_res = build_and_run(x_tensor, ops_tensor, ch_tensor, device=device)
    manual_res = manual_run(x_tensor, ops_tensor, ch_tensor, device=device)
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

    custom_res = build_and_run(x_tensor, ops_tensor, ch_tensor, device=device)
    manual_res = manual_run(x_tensor, ops_tensor, ch_tensor, device=device)
    assert custom_res.shape == (2, 10)
    assert torch.allclose(custom_res, manual_res, atol=1e-6, equal_nan=True)

# 5. Test CPU vs GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_gpu_match() -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    ops_tensor = torch.tensor([[VAR_START_ID], [VAR_START_ID+1], [ADD]], dtype=torch.int64)
    ch_tensor = -torch.ones(3, 1, 2, dtype=torch.int64)
    ch_tensor[2, 0, 0] = 0
    ch_tensor[2, 0, 1] = 1
    
    cpu_res = build_and_run(x_tensor, ops_tensor, ch_tensor, device='cpu')
    gpu_res = build_and_run(x_tensor, ops_tensor, ch_tensor, device='cuda')
    assert torch.allclose(cpu_res, gpu_res.cpu(), atol=1e-6)

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

    custom_res = build_and_run(x_tensor, ops_tensor, ch_tensor, device=device)
    manual_res = manual_run(x_tensor, ops_tensor, ch_tensor, device=device)
    assert torch.allclose(custom_res, manual_res, atol=1e-6, equal_nan=True)

# 7. Test constants
def test_constants(device : str) -> None:
    x_tensor = torch.randn(10, 1, dtype=torch.float32)
    c_tensor = torch.tensor([0.5], dtype=torch.float32)
    
    # Test C * X0
    ops_tensor = torch.tensor([[VAR_START_ID], [LEARNABLE_CONSTANT], [MUL]], dtype=torch.int64)
    ch_tensor = -torch.ones(3, 1, 2, dtype=torch.int64)
    ch_tensor[2, 0, 0] = 0
    ch_tensor[2, 0, 1] = 1
    
    custom_res = build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    manual_res = manual_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    assert torch.allclose(custom_res, manual_res, atol=1e-6)
    assert torch.allclose(custom_res, x_tensor.to(device).squeeze() * 0.5, atol=1e-6)

    # Test sin(C)
    ops_tensor = torch.tensor([[LEARNABLE_CONSTANT], [SIN]], dtype=torch.int64)
    ch_tensor = -torch.ones(2, 1, 2, dtype=torch.int64)
    ch_tensor[1, 0, 0] = 0
    
    custom_res = build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    manual_res = manual_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    assert torch.allclose(custom_res, manual_res, atol=1e-6)
    assert torch.allclose(custom_res, torch.sin(c_tensor.to(device)).expand(1,10), atol=1e-6)

# 8. Benchmark
def test_benchmark() -> None:
    B, N, n_x = 1024, 512, 4
    x_tensor = torch.randn(N, n_x, dtype=torch.float32)
    
    # A moderately complex expression
    # (X0 + X1) * sin(X2 - C)
    ops_tensor = torch.tensor([
        [VAR_START_ID],
        [VAR_START_ID + 1],
        [VAR_START_ID + 2],
        [LEARNABLE_CONSTANT],
        [ADD], # X0+X1
        [SUB], # X2-C
        [SIN], # sin(X2-C)
        [MUL]  # (X0+X1)*sin(X2-C)
    ], dtype=torch.int64).expand(-1, B)
    
    ch_tensor = -torch.ones(8, B, 2, dtype=torch.int64)
    ch_tensor[4, :, 0] = 0
    ch_tensor[4, :, 1] = 1
    ch_tensor[5, :, 0] = 2
    ch_tensor[5, :, 1] = 3
    ch_tensor[6, :, 0] = 5
    ch_tensor[7, :, 0] = 4
    ch_tensor[7, :, 1] = 6

    c_tensor = torch.randn(B, dtype=torch.float32)

    # Manual PyTorch
    start_time = time.time()
    manual_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device='cpu')
    manual_time = time.time() - start_time
    print(f"\nManual PyTorch (CPU): {manual_time:.4f}s")

    # Custom CPU
    start_time = time.time()
    build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device='cpu')
    cpu_time = time.time() - start_time
    print(f"Custom C++ (CPU): {cpu_time:.4f}s")

    if torch.cuda.is_available():
        # Custom GPU
        start_time = time.time()
        build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device='cuda')
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"Custom C++ (GPU): {gpu_time:.4f}s")

if __name__ == "__main__":
    pytest.main()

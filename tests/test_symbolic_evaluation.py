import pytest
import torch
from torch import Tensor
import time
from typing import List, Optional

# Import the compiled extension
import symbolic_torch
#torch.autograd.set_detect_anomaly(True)

from symbolic_torch.evaluation import evaluate_backend
from symbolic_torch import Operator

NULL_CHILD = -1

unary_ops: List[int] = [int(Operator.SIN), int(Operator.COS), int(Operator.EXP), int(Operator.LOG), int(Operator.SQUARE), int(Operator.SQRT)]
binary_ops: List[int] = [int(Operator.ADD), int(Operator.SUB), int(Operator.MUL), int(Operator.DIV), int(Operator.POW)]

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

    # NOTE: The tests in this file use M-first tensors, so we call backend directly.
    y = evaluate_backend(x_tensor, ops_tensor, ch_tensor, c_tensor)
    return y

def build_and_run_with_grad(x_tensor: Tensor, ops_tensor: Tensor, ch_tensor: Tensor, c_tensor: Tensor, grad_output : Tensor, null_nan_output_grad : bool, device: str = 'cpu', use_custom_kernel: bool = True):
    x_tensor = x_tensor.clone().detach().to(device).requires_grad_(True)
    ops_tensor = ops_tensor.to(device)
    ch_tensor = ch_tensor.to(device)
    c_tensor = c_tensor.clone().detach().to(device).requires_grad_(True)

    if use_custom_kernel:
        y = evaluate_backend(x_tensor, ops_tensor, ch_tensor, c_tensor)
    else:
        y = manual_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    
    # Use the last node's output for grad calculation
    output = y[-1]
    # check that grad_output is 0 where output is NaN
    grad_output = grad_output.to(device)

    if null_nan_output_grad:
        grad_output[torch.isnan(output)] = 0
    
    output.backward(gradient=grad_output) # type: ignore[call-arg]
    
    return x_tensor.grad, c_tensor.grad

def manual_run(x_tensor: Tensor, ops_tensor: Tensor, ch_tensor: Tensor, c_tensor: Tensor, device: str = 'cpu') -> Tensor:
    M, B = ops_tensor.shape
    N, _ = x_tensor.shape
    
    x_tensor = x_tensor.to(device)
    ops_tensor = ops_tensor.to(device)
    ch_tensor = ch_tensor.to(device)
    c_tensor = c_tensor.to(device)

    cache_list: List[Tensor] = []
    
    # WARNING: We do NOT add epsilon for stability here, as we want it to match the nan output of the C++ kernel.
    # WARNING: We also do not use abs() here as we want to match the behavior of the C++ kernel.
    for k in range(M):
        # For each node in the expression tree
        batch_results: List[Tensor] = []
        for b in range(B):
            # For each expression in the batch
            op = int(ops_tensor[k, b].item())

            if op == int(Operator.NO_OP):
                # Append a column of NaNs for this expression in the batch
                batch_results.append(torch.full((N, 1), float('nan'), dtype=x_tensor.dtype, device=device))
                continue

            arity = get_arity(int(op))
            arg0: Optional[Tensor] = None
            arg1: Optional[Tensor] = None
            if arity >= 1:
                ch0_idx = ch_tensor[k, b, 0].item()
                if ch0_idx != NULL_CHILD:
                    arg0 = cache_list[int(ch0_idx)][:, b]
            if arity == 2:
                ch1_idx = ch_tensor[k, b, 1].item()
                if ch1_idx != NULL_CHILD:
                    arg1 = cache_list[int(ch1_idx)][:, b]

            res: Optional[Tensor] = None
            if op == int(Operator.LEARNABLE_CONSTANT):
                res = c_tensor[k, b].expand(N)
            elif op == int(Operator.CONST_1):
                res = torch.full((N,), 1.0, dtype=x_tensor.dtype, device=device)
            elif op == int(Operator.CONST_2):
                res = torch.full((N,), 2.0, dtype=x_tensor.dtype, device=device)
            elif op == int(Operator.CONST_3):
                res = torch.full((N,), 3.0, dtype=x_tensor.dtype, device=device)
            elif op == int(Operator.CONST_4):
                res = torch.full((N,), 4.0, dtype=x_tensor.dtype, device=device)
            elif op == int(Operator.CONST_5):
                res = torch.full((N,), 5.0, dtype=x_tensor.dtype, device=device)
            elif op >= int(Operator.VAR_START_ID):
                var_idx = op - int(Operator.VAR_START_ID)
                res = x_tensor[:, int(var_idx)]
            elif arity > 0 and arg0 is not None:
                if op == int(Operator.SIN):
                    res = torch.sin(arg0)
                elif op == int(Operator.COS):
                    res = torch.cos(arg0)
                elif op == int(Operator.EXP):
                    res = torch.exp(arg0)
                elif op == int(Operator.LOG):
                    res = torch.log(arg0)
                elif op == int(Operator.SQUARE):
                    res = arg0 * arg0
                elif op == int(Operator.SQRT):
                    res = torch.sqrt(arg0)
                elif arity > 1 and arg1 is not None:
                    if op == int(Operator.ADD):
                        res = arg0 + arg1
                    elif op == int(Operator.SUB):
                        res = arg0 - arg1
                    elif op == int(Operator.MUL):
                        res = arg0 * arg1
                    elif op == int(Operator.DIV):
                        res = arg0 / arg1
                    elif op == int(Operator.POW):
                        res = torch.pow(arg0, arg1)
            
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
    ops_tensor = torch.tensor([[int(Operator.NO_OP)]], dtype=torch.int64)
    ch_tensor = -torch.ones(1, 1, 2, dtype=torch.int64)
    # create_constants expects (B,M), test ops_tensor is (M,B)
    c_tensor = symbolic_torch.create_constants(ops_tensor.permute(1,0))
    c_tensor = c_tensor.permute(1,0)
    custom_res = build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    assert torch.isnan(custom_res).all(), "Expected output to be NaN when using Operator.NO_OP"

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
            ops_tensor[i, 0] = int(Operator.VAR_START_ID) + i
        
        # Set the operator
        ops_tensor[arity, 0] = op

        # Set correct number of children for the operator
        for i in range(arity):
            ch_tensor[arity, 0, i] = i
            
        # create_constants expects (B,M), test ops_tensor is (M,B)
        c_tensor = symbolic_torch.create_constants(ops_tensor.permute(1,0))
        c_tensor = c_tensor.permute(1,0)
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
            ch_tensor_one_less[arity, 0, arity - 1] = NULL_CHILD
            with pytest.raises(RuntimeError, match="Input validation failed"):
                build_and_run(x_tensor, ops_tensor, ch_tensor_one_less, c_tensor, device=device)
        elif arity == 1:
            # For unary ops, test with one too few children.
            # This is only an error if the result is used by another node.
            # M=0: Operator.VAR_START_ID
            # M=1: op (unary) with invalid child
            # M=2: Operator.ADD(M=0, M=1)
            ops_tensor_unary_fail = torch.tensor([
                [int(Operator.VAR_START_ID)],
                [op],
                [int(Operator.ADD)]
            ], dtype=torch.int64)
            ch_tensor_unary_fail = -torch.ones(3, 1, 2, dtype=torch.int64)
            ch_tensor_unary_fail[1, 0, 0] = NULL_CHILD # Invalid child for the unary op
            ch_tensor_unary_fail[2, 0, 0] = 0
            ch_tensor_unary_fail[2, 0, 1] = 1
            
            # create_constants expects (B,M), test ops_tensor is (M,B)
            c_tensor_unary_fail = symbolic_torch.create_constants(ops_tensor_unary_fail.permute(1,0))
            c_tensor_unary_fail = c_tensor_unary_fail.permute(1,0)
            with pytest.raises(RuntimeError, match="Input validation failed"):
                build_and_run(x_tensor, ops_tensor_unary_fail, ch_tensor_unary_fail, c_tensor_unary_fail, device=device)

def test_validate_inputs_child_index_too_large(device : str) -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    ops_tensor = torch.tensor([[int(Operator.VAR_START_ID)], [int(Operator.ADD)]], dtype=torch.int64)
    ch_tensor = -torch.ones(2, 1, 2, dtype=torch.int64)
    ch_tensor[1, 0, 0] = 0
    ch_tensor[1, 0, 1] = 2 # Invalid, should be < k=1
    # create_constants expects (B,M), test ops_tensor is (M,B)
    c_tensor = symbolic_torch.create_constants(ops_tensor.permute(1,0))
    c_tensor = c_tensor.permute(1,0)
    with pytest.raises(RuntimeError, match="Input validation failed"):
        build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)

def test_validate_inputs_unary_op_no_children(device: str) -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    for op in unary_ops:
        ops_tensor = torch.tensor([[op]], dtype=torch.int64)
        ch_tensor = -torch.ones(1, 1, 2, dtype=torch.int64)
        # create_constants expects (B,M), test ops_tensor is (M,B)
        c_tensor = symbolic_torch.create_constants(ops_tensor.permute(1,0))
        c_tensor = c_tensor.permute(1,0)
        with pytest.raises(RuntimeError, match="Input validation failed"):
            build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)

def test_validate_inputs_binary_op_no_children(device: str) -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    for op in binary_ops:
        ops_tensor = torch.tensor([[op]], dtype=torch.int64)
        ch_tensor = -torch.ones(1, 1, 2, dtype=torch.int64)
        # create_constants expects (B,M), test ops_tensor is (M,B)
        c_tensor = symbolic_torch.create_constants(ops_tensor.permute(1,0))
        c_tensor = c_tensor.permute(1,0)
        with pytest.raises(RuntimeError, match="Input validation failed"):
            build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)

# 2. Test datatype checking
def test_wrong_datatypes(device : str) -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    ops_tensor = torch.tensor([[int(Operator.VAR_START_ID)]], dtype=torch.int64)
    ch_tensor = -torch.ones(1, 1, 2, dtype=torch.int64)
    c_tensor = torch.zeros_like(ops_tensor, dtype=torch.float32)

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
        ops_tensor[i, 0] = int(Operator.VAR_START_ID) + i
        ch_tensor[arity, 0, i] = i
    ops_tensor[arity, 0] = op

    # create_constants expects (B,M), test ops_tensor is (M,B)
    c_tensor = symbolic_torch.create_constants(ops_tensor.permute(1,0))
    c_tensor = c_tensor.permute(1,0)
    custom_res = build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    manual_res = manual_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    assert torch.allclose(custom_res, manual_res, atol=1e-6, equal_nan=True)

# 4. Test batch processing
def test_batching(device : str) -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    # Tree 1: X0 + X1
    # Tree 2: sin(X0)
    ops_tensor = torch.tensor([
        [int(Operator.VAR_START_ID), int(Operator.VAR_START_ID)],
        [int(Operator.VAR_START_ID)+1, int(Operator.NO_OP)],
        [int(Operator.ADD), int(Operator.SIN)]
    ], dtype=torch.int64)
    ch_tensor = -torch.ones(3, 2, 2, dtype=torch.int64)
    ch_tensor[2, 0, 0] = 0
    ch_tensor[2, 0, 1] = 1
    ch_tensor[2, 1, 0] = 0

    # create_constants expects (B,M), test ops_tensor is (M,B)
    c_tensor = symbolic_torch.create_constants(ops_tensor.permute(1,0))
    c_tensor = c_tensor.permute(1,0)
    custom_res = build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    manual_res = manual_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    assert custom_res.shape == (3, 10, 2)
    assert torch.allclose(custom_res, manual_res, atol=1e-6, equal_nan=True)

# 5. Test CPU vs GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_gpu_match() -> None:
    x_tensor = torch.randn(10, 2, dtype=torch.float32)
    ops_tensor = torch.tensor([[int(Operator.VAR_START_ID)], [int(Operator.VAR_START_ID)+1], [int(Operator.ADD)]], dtype=torch.int64)
    ch_tensor = -torch.ones(3, 1, 2, dtype=torch.int64)
    ch_tensor[2, 0, 0] = 0
    ch_tensor[2, 0, 1] = 1
    # create_constants expects (B,M), test ops_tensor is (M,B)
    c_tensor = symbolic_torch.create_constants(ops_tensor.permute(1,0))
    c_tensor = c_tensor.permute(1,0)
    
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
        [int(Operator.VAR_START_ID)],
        [int(Operator.VAR_START_ID) + 1],
        [int(Operator.SIN)],
        [int(Operator.MUL)],
        [int(Operator.ADD)]
    ], dtype=torch.int64)
    ch_tensor = -torch.ones(5, 1, 2, dtype=torch.int64)
    ch_tensor[2, 0, 0] = 1 # sin(X1)
    ch_tensor[3, 0, 0] = 0 # X0 * X1
    ch_tensor[3, 0, 1] = 1
    ch_tensor[4, 0, 0] = 2 # op + op
    ch_tensor[4, 0, 1] = 3

    # create_constants expects (B,M), test ops_tensor is (M,B)
    c_tensor = symbolic_torch.create_constants(ops_tensor.permute(1,0))
    c_tensor = c_tensor.permute(1,0)
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
        [int(Operator.LEARNABLE_CONSTANT)],
        [int(Operator.CONST_1)],
        [int(Operator.ADD)]
    ], dtype=torch.int64)
    ch_tensor = -torch.ones(3, 1, 2, dtype=torch.int64)
    ch_tensor[2, 0, 0] = 0
    ch_tensor[2, 0, 1] = 1
    c_tensor = torch.zeros_like(ops_tensor, dtype=torch.float32).to(device)
    c_tensor[0, 0] = 3.14

    custom_res = build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    manual_res = manual_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    assert torch.allclose(custom_res, manual_res, atol=1e-6)
    # The output shape is (M, N, B), which is (3, 4, 1) here.
    # We need to construct a tensor of the same shape to compare against.
    expected_res = torch.stack([
        torch.full((4, 1), 3.14, dtype=torch.float32, device=device),
        torch.ones(4, 1, device=device) * 1.0,
        torch.full((4, 1), 3.14, dtype=torch.float32, device=device) + 1.0
    ], dim=0).view(3, 4, 1)
    assert torch.allclose(custom_res.to(device), expected_res.to(device), atol=1e-6)

    # Test sin(C)
    ops_tensor = torch.tensor([[int(Operator.LEARNABLE_CONSTANT)], [int(Operator.SIN)]], dtype=torch.int64)
    ch_tensor = -torch.ones(2, 1, 2, dtype=torch.int64)
    ch_tensor[1, 0, 0] = 0
    c_tensor = torch.zeros_like(ops_tensor, dtype=torch.float32).to(device)
    c_tensor[0, 0] = 3.14
    
    custom_res = build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    manual_res = manual_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device=device)
    assert torch.allclose(custom_res, manual_res, atol=1e-6)
    expected_res_sin = torch.stack([
        torch.full((4, 1), 3.14, dtype=torch.float32, device=device),
        torch.sin(torch.tensor(3.14, device=device)).expand(4, 1)
    ], dim=0).view(2, 4, 1)
    assert torch.allclose(custom_res.to(device), expected_res_sin.to(device), atol=1e-6)

# 8. Test Gradients
@pytest.mark.parametrize("op", all_ops)
def test_gradients(device: str, op: int) -> None:
    arity = get_arity(op)
    x_tensor = torch.randn(10, arity if arity > 0 else 1, dtype=torch.float32)
    
    ops_tensor = torch.zeros(1 + arity, 1, dtype=torch.int64)
    ch_tensor = -torch.ones(1 + arity, 1, 2, dtype=torch.int64)
    # create_constants expects (B,M), test ops_tensor is (M,B)
    c_tensor = symbolic_torch.create_constants(ops_tensor.permute(1,0))
    c_tensor = c_tensor.permute(1,0).to(device)

    for i in range(arity):
        ops_tensor[i, 0] = int(Operator.VAR_START_ID) + i
        ch_tensor[arity, 0, i] = i
    ops_tensor[arity, 0] = op

    # Special case for log to test negative inputs
    #if op == int(Operator.LOG:
    #    x_tensor = torch.abs(x_tensor) * -1 # Ensure negative inputs for log

    N = x_tensor.shape[0]
    B = ops_tensor.shape[-1]

    grad_output = torch.randn(N, B, dtype=torch.float32)
    custom_x_grad, custom_c_grad = build_and_run_with_grad(x_tensor, ops_tensor, ch_tensor, c_tensor, grad_output, null_nan_output_grad=True, device=device, use_custom_kernel=True)

    manual_x_grad, manual_c_grad = build_and_run_with_grad(x_tensor, ops_tensor, ch_tensor, c_tensor, grad_output, null_nan_output_grad=True, device=device, use_custom_kernel=False)

    assert custom_x_grad is not None and manual_x_grad is not None, "Gradients should not be None"
    assert torch.allclose(custom_x_grad, manual_x_grad, atol=1e-5, equal_nan=True)
    if custom_c_grad is not None and manual_c_grad is not None:
        assert torch.allclose(custom_c_grad, manual_c_grad, atol=1e-5, equal_nan=True)

# 9. Benchmark
def test_benchmark() -> None:
    B, N, n_x = 4024, 1024, 4
    x_tensor = torch.randn(N, n_x, dtype=torch.float32)
    
    # A moderately complex expression
    # (X0 + X1) * sin(X2 - C)
    ops_tensor = torch.tensor([
        [int(Operator.VAR_START_ID)],
        [int(Operator.VAR_START_ID) + 1],
        [int(Operator.VAR_START_ID) + 2],
        [int(Operator.LEARNABLE_CONSTANT)],
        [int(Operator.ADD)], # X0+X1
        [int(Operator.SUB)], # X2-C
        [int(Operator.SIN)], # sin(X2-C)
        [int(Operator.MUL)]  # (X0+X1)*sin(X2-C)
    ], dtype=torch.int64).expand(-1, B)
    
    ch_tensor = -torch.ones(8, B, 2, dtype=torch.int64)
    ch_tensor[4, :, 0] = 0
    ch_tensor[4, :, 1] = 1
    ch_tensor[5, :, 0] = 2
    ch_tensor[5, :, 1] = 3
    ch_tensor[6, :, 0] = 5
    ch_tensor[7, :, 0] = 4
    ch_tensor[7, :, 1] = 6

    c_tensor = torch.zeros_like(ops_tensor, dtype=torch.float32)
    const_mask = (ops_tensor == int(Operator.LEARNABLE_CONSTANT))
    c_tensor[const_mask] = torch.randn(int(const_mask.sum()))

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

# 10. Larger Benchmark
def test_large_benchmark() -> None:
    B, N, n_x = 8192, 2048, 8
    x_tensor = torch.randn(N, n_x, dtype=torch.float32)
    # No constants in this expression, so C is all zeros
    c_tensor = torch.zeros((20, B), dtype=torch.float32).to(x_tensor.device)

    # A more complex expression:
    # log( (sin(X0) + cos(X1)) * (exp(X2) - sqrt(X3)) ) + (X4 * X5) - (X6 / X7)
    ops_tensor = torch.tensor([
        [int(Operator.VAR_START_ID) + 0],      # M=0: X0
        [int(Operator.VAR_START_ID) + 1],      # M=1: X1
        [int(Operator.VAR_START_ID) + 2],      # M=2: X2
        [int(Operator.VAR_START_ID) + 3],      # M=3: X3
        [int(Operator.VAR_START_ID) + 4],      # M=4: X4
        [int(Operator.VAR_START_ID) + 5],      # M=5: X5
        [int(Operator.VAR_START_ID) + 6],      # M=6: X6
        [int(Operator.VAR_START_ID) + 7],      # M=7: X7
        [int(Operator.SIN)],                   # M=8: sin(X0)
        [int(Operator.COS)],                   # M=9: cos(X1)
        [int(Operator.EXP)],                   # M=10: exp(X2)
        [int(Operator.SQRT)],                  # M=11: sqrt(X3)
        [int(Operator.ADD)],                   # M=12: sin(X0) + cos(X1)
        [int(Operator.SUB)],                   # M=13: exp(X2) - sqrt(X3)
        [int(Operator.MUL)],                   # M=14: (sin(X0) + cos(X1)) * (exp(X2) - sqrt(X3))
        [int(Operator.LOG)],                   # M=15: log(...)
        [int(Operator.MUL)],                   # M=16: X4 * X5
        [int(Operator.DIV)],                   # M=17: X6 / X7
        [int(Operator.ADD)],                   # M=18: log(...) + (X4 * X5)
        [int(Operator.SUB)]                    # M=19: ... - (X6 / X7)
    ], dtype=torch.int64).expand(-1, B)

    ch_tensor = -torch.ones(20, B, 2, dtype=torch.int64)
    ch_tensor[8, :, 0] = 0   # sin(X0)
    ch_tensor[9, :, 0] = 1   # cos(X1)
    ch_tensor[10, :, 0] = 2  # exp(X2)
    ch_tensor[11, :, 0] = 3  # sqrt(X3)
    ch_tensor[12, :, 0] = 8  # M=8 + M=9
    ch_tensor[12, :, 1] = 9
    ch_tensor[13, :, 0] = 10 # M=10 - M=11
    ch_tensor[13, :, 1] = 11
    ch_tensor[14, :, 0] = 12 # M=12 * M=13
    ch_tensor[14, :, 1] = 13
    ch_tensor[15, :, 0] = 14 # log(M=14)
    ch_tensor[16, :, 0] = 4  # M=4 * M=5
    ch_tensor[16, :, 1] = 5
    ch_tensor[17, :, 0] = 6  # M=6 / M=7
    ch_tensor[17, :, 1] = 7
    ch_tensor[18, :, 0] = 15 # M=15 + M=16
    ch_tensor[18, :, 1] = 16
    ch_tensor[19, :, 0] = 18 # M=18 - M=17
    ch_tensor[19, :, 1] = 17

    # Manual PyTorch
    start_time = time.time()
    manual_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device='cpu')
    manual_time = time.time() - start_time
    print(f"\nLarge Benchmark - Manual PyTorch (CPU): {manual_time:.4f}s")

    # Custom CPU
    start_time = time.time()
    build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device='cpu')
    cpu_time = time.time() - start_time
    print(f"Large Benchmark - Custom C++ (CPU): {cpu_time:.4f}s")

    if torch.cuda.is_available():
        # Custom GPU
        start_time = time.time()
        build_and_run(x_tensor, ops_tensor, ch_tensor, c_tensor, device='cuda')
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"Large Benchmark - Custom C++ (GPU): {gpu_time:.4f}s")

# 10. Test error handling for gradients on invalid operations
@pytest.mark.parametrize("op", [
    Operator.NO_OP,
    Operator.LOG,
    Operator.DIV, 
])
def test_gradient_on_invalid_op(op: int, device: str):
    B, N, n_x = 1, 1, 2
    x_tensor = torch.ones(N, n_x, dtype=torch.float32)
    
    grad_output = torch.ones(N, B, dtype=torch.float32)

    if op == int(Operator.LOG):
        # log(0)
        ops_tensor = torch.tensor([[int(Operator.VAR_START_ID)], [int(Operator.LOG)]], dtype=torch.int64).expand(-1, B)
        ch_tensor = -torch.ones(2, B, 2, dtype=torch.int64)
        ch_tensor[1, :, 0] = 0
        x_tensor[0, 0] = 0
        c_tensor = torch.zeros_like(ops_tensor, dtype=torch.float32)
    elif op == int(Operator.DIV):
        # 1/0
        ops_tensor = torch.tensor([[int(Operator.CONST_1)], [int(Operator.VAR_START_ID)], [int(Operator.DIV)]], dtype=torch.int64).expand(-1, B)
        ch_tensor = -torch.ones(3, B, 2, dtype=torch.int64)
        ch_tensor[2, :, 0] = 0
        ch_tensor[2, :, 1] = 1
        x_tensor[0, 0] = 0
        c_tensor = torch.zeros_like(ops_tensor, dtype=torch.float32)
    elif op == int(Operator.NO_OP):
        ops_tensor = torch.tensor([[int(Operator.NO_OP)]], dtype=torch.int64).expand(-1, B)
        ch_tensor = -torch.ones(1, B, 2, dtype=torch.int64)
        c_tensor = torch.zeros_like(ops_tensor, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported operator for gradient test: {op}")
    
    with pytest.raises(RuntimeError, match="Backward error: Gradient propagated to a"):
        build_and_run_with_grad(x_tensor, ops_tensor, ch_tensor, c_tensor, grad_output, null_nan_output_grad=False, device=device, use_custom_kernel=True)


if __name__ == "__main__":
    pytest.main()
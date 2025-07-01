import torch
import numpy as np
import symbolic_cuda
from symbolic_cuda import Operator

def build_test_expression():
    """
    Builds tensors for the expression: y = sin(X[:,0] * C[0]) + X[:,1]
    Postfix: X_0 C_0 * sin X_1 +
    """
    op = Operator # Alias for convenience
    B = 1
    M = 6 # Max length
    
    # Ops: (M, B)
    ops_list = [
        op.VAR_START_ID.value + 0,    # k=0: X_0
        op.LEARNABLE_CONSTANT,  # k=1: C_0
        op.MUL,                 # k=2: k=0 * k=1
        op.SIN,                 # k=3: sin(k=2)
        op.VAR_START_ID.value + 1,    # k=4: X_1
        op.ADD,                 # k=5: k=3 + k=4
    ]
    Ops = torch.full((M, B), op.NO_OP.value, dtype=torch.int32)
    ops_list_tensor = torch.tensor([o.value if isinstance(o, Operator) else o for o in ops_list], dtype=torch.int32)
    Ops[:len(ops_list), 0] = ops_list_tensor

    # Ch: (M, B, MaxArity)
    Ch = torch.full((M, B, 2), -1, dtype=torch.int32)
    # k=2 (MUL) children are at k=0 and k=1
    Ch[2, 0, 0] = 0
    Ch[2, 0, 1] = 1
    # k=3 (SIN) child is at k=2
    Ch[3, 0, 0] = 2
    # k=5 (ADD) children are at k=3 and k=4
    Ch[5, 0, 0] = 3
    Ch[5, 0, 1] = 4

    return Ops, Ch

def test_evaluation():
    print("--- Running Test: symbolic_cuda.evaluate ---")
    # --- Setup ---
    device = torch.device("cuda")
    print(f"Using device: {device}")

    N = 100
    n_x = 2
    
    Ops, Ch = build_test_expression()
    Ops = Ops.to(device)
    Ch = Ch.to(device)
    
    X = torch.randn(N, n_x, dtype=torch.float32, device=device)
    
    # Let the custom function initialize C
    y_custom, C_custom = symbolic_cuda.evaluate(X, Ops, Ch, C=None)
    
    # For PyTorch comparison, use the same C
    C_pytorch = C_custom.clone().detach()

    # --- PyTorch Ground Truth ---
    X_pytorch = X.clone()
    X_pytorch.requires_grad_(True)
    C_pytorch.requires_grad_(True)
    
    y_pytorch = torch.sin(X_pytorch[:, 0] * C_pytorch[0]) + X_pytorch[:, 1]
    y_pytorch = y_pytorch.unsqueeze(0)

    # --- Forward Pass Comparison ---
    print(f"Comparing forward pass... ", end="")
    assert y_custom.shape == y_pytorch.shape, f"Shape mismatch! {y_custom.shape} vs {y_pytorch.shape}"
    assert torch.allclose(y_custom, y_pytorch, atol=1e-6), "Forward pass mismatch!"
    print("OK")
    
    # --- Backward Pass Comparison ---
    print(f"Comparing backward pass... ", end="")
    
    # Custom function backward
    # We need to re-run the forward pass with tensors that require gradients
    X.requires_grad_(True)
    C_custom.requires_grad_(True)
    y_custom_grad, _ = symbolic_cuda.evaluate(X, Ops, Ch, C=C_custom)
    loss_custom = y_custom_grad.sum()
    import code; code.interact(local=locals())  # Debugging line to inspect variables
    loss_custom.backward()

    # PyTorch backward
    loss_pytorch = y_pytorch.sum()
    loss_pytorch.backward()
    
    assert torch.allclose(X.grad, X_pytorch.grad, atol=1e-5), "Gradient dL/dX mismatch!"
    assert torch.allclose(C_custom.grad, C_pytorch.grad, atol=1e-5), "Gradient dL/dC mismatch!"
    print("OK")

    print("\nTest passed successfully!")


if __name__ == "__main__":
    # Print available operators
    print("Available Operators:")
    for name, member in Operator.__members__.items():
        print(f"{name}: {member.value}")
    print("-" * 20)

    test_evaluation()
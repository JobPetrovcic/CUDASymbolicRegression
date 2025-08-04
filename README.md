# symbolic-torch

A high-performance PyTorch library for symbolic regression using batched expression generation, evaluation, and optimization on CPU and GPU.

## Installation

Clone the repository and install the package using pip. This will compile the C++/CUDA extensions.

```bash
git clone https://github.com/your-username/symbolic-torch.git
cd symbolic-torch
pip install .
```
or, succinctly,
```bash
pip install git+https://github.com/your-username/symbolic-torch.git
```

## Quick Start Example

The following example demonstrates the end-to-end workflow: defining a grammar, sampling a population of candidate expressions, and using the batched optimizer to find the best expression and constants to fit a target dataset.

```python
# main_example.py
import torch
import symbolic_torch

# This is an extended example demonstrating a more complete workflow of the symbolic_torch library:
# 1. Grammar: Define a Probabilistic Context-Free Grammar (PCFG).
# 2. Generation (Sampling): Sample a batch of diverse candidate expressions from the grammar.
# 3. Evaluate & Optimize: Use the batched L-BFGS optimizer to find the best constants for *each*
#    sampled expression to fit a given target dataset.
# 4. Verify: Identify the best-fitting expression from the batch.

# --- 1. Grammar Definition ---
# A simple grammar that can produce a variety of expressions.
grammar_string = """
E -> E + T [0.4]
E -> T [0.6]
T -> F * T [0.4]
T -> F [0.6]
F -> R ( V ) [0.3]
F -> V [0.7]
R -> sin [0.5]
R -> cos [0.5]
V -> X_0 [0.5]
V -> C [0.5]
"""

# --- 2. Generation via Sampling ---
# We will now sample expressions directly from the PCFG.

# Setup PCFG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
    grammar=grammar_string,
    start_symbol="E",
    padded_maximum_length=32, # Max length of the *infix* expression string
    n_variables=1,
    device=device
)

# Sample a batch of B expressions.
# `pcfg.sample()` directly returns the `Ops` and `Ch` tensors in postfix format.
B = 10
print("--- 1. Sampling Candidate Expressions ---")
print(f"Device: {device}")
print(f"Sampling {B} expressions from the grammar...")
Ops, Ch = pcfg.sample(B)

# To see what we've sampled, we convert the postfix `Ops` to infix strings.
# This is for display purposes only; the optimizer uses the postfix `Ops` and `Ch`.
max_infix_len = 64
infix_ids = pcfg.postfix_to_infix(Ops, max_infix_len)
sampled_strings = pcfg.to_string(infix_ids)

print("\nSampled candidate expressions:")
for i, s in enumerate(sampled_strings[:5]): # Print the first 5
    print(f"  {i+1}: {s}")
print("  ...")


# --- 3. Evaluation & Optimization ---
# We'll create a synthetic dataset and fit all B candidate expressions to it.
N = 100  # Number of data points
n_x = 1  # Number of variables

# Let's define a target function the PCFG *could* discover.
# y = 2.5 * cos(X_0) - 1.5
X = torch.randn(N, n_x, device=device) * 2 # Data in [-2, 2]
Y_target = 2.5 * torch.cos(X[:, 0]) - 1.5
Y_target = Y_target.unsqueeze(0).expand(B, -1) # Tile for the batch -> (B, N)

# Create initial (random) constants for all sampled expressions.
C_init = symbolic_torch.create_constants(Ops, method='randn')

# Calculate the initial MSE before optimization
with torch.no_grad():
    y_cache_initial = symbolic_torch.evaluation.evaluate(X, Ops, Ch, C_init)
    root_indices = (Ops != int(symbolic_torch.Operator.NO_OP)).sum(dim=1) - 1
    indices = root_indices.view(B, 1, 1).expand(B, 1, N)
    y_pred_initial = y_cache_initial.gather(1, indices).squeeze(1)
    initial_mse_per_expr = torch.mean((y_pred_initial - Y_target)**2, dim=1)


print("\n--- 2. Optimization ---")
print(f"Fitting all {B} expressions simultaneously to the target data `y = 2.5*cos(X_0) - 1.5`...")

# Run the batched L-BFGS optimizer. It fits constants for all B expressions in parallel.
C_final, Y_final = symbolic_torch.optimize(
    X=X,
    Y=Y_target,
    Ops=Ops,
    Ch=Ch,
    C_init=C_init,
    max_iter=100,
    tolerance_grad=1e-7,
    line_search_fn='strong_wolfe'
)


# --- 4. Verification ---
print("\n--- 3. Verification ---")
# Calculate the final MSE for each expression
final_mse_per_expr = torch.mean((Y_final - Y_target)**2, dim=1)

# Find the best expression (the one with the lowest final MSE)
best_expr_idx = torch.argmin(final_mse_per_expr)
best_mse = final_mse_per_expr[best_expr_idx]

print("Optimization Results (Initial MSE -> Final MSE):")
for i in range(B):
    print(f"  Expr {i+1:2d}: {initial_mse_per_expr[i]:.4f} -> {final_mse_per_expr[i]:.4f} | {sampled_strings[i]}")

print(f"\nBest-fitting expression was #{best_expr_idx.item()+1} with a final MSE of {best_mse.item():.6f}")
print(f"Best expression found: {sampled_strings[best_expr_idx.item()]}")

# The constants for the winning expression
best_C = C_final[best_expr_idx]
best_Ops = Ops[best_expr_idx]
# Create a mask to find the locations of LEARNABLE_CONSTANT in the winning expression
const_mask = (best_Ops == int(symbolic_torch.Operator.LEARNABLE_CONSTANT))
found_consts = best_C[const_mask]
print(f"Its fitted constants are: {found_consts.cpu().numpy().round(4)}")

# We expect the best MSE to be very low, indicating one of the
# expressions was a good fit for the target data.
assert best_mse < 0.1, "The optimizer failed to find a good fit among the candidates."
print("\nExtended example finished successfully!")

```

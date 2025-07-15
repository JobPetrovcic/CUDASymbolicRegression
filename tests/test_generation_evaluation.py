import torch
import pytest

from symbolic_torch import ProbabilisticContextFreeGrammar, create_constants
from symbolic_torch.evaluation import evaluate

test_grammar = """E -> E + F [0.2]
E -> E - F [0.2]
E -> F [0.6]
F -> F * T [0.2]
F -> F / T [0.2]
F -> T [0.6]
T -> ( E ) P [0.2]
T -> R ( E ) [0.2]
T -> V [0.4000000000000001]
T -> ( E ) [0.20000000000000004]
V -> X_0 [1.0]
R -> sin [0.2]
R -> cos [0.2]
R -> exp [0.2]
R -> sqrt [0.2]
R -> log [0.2]
P -> ^2 [1.0]
"""

# Parametrize device to test both CPU and CUDA implementations
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_generated_evalvation(device : str):
    n_variables = 2
    M = 128
    pcfg = ProbabilisticContextFreeGrammar(
        grammar=test_grammar,
        start_symbol="E",
        padded_maximum_length=M,
        n_variables=n_variables,
        device=torch.device(device),
    )
    N = 100
    B = 10
    ops, ch = pcfg.sample(B)
    c = create_constants(ops)
    X = torch.randn(N, n_variables, device=torch.device(device))

    evaluated = evaluate(X, ops, ch, c)
    assert evaluated.shape == (B, M, N)
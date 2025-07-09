import torch
import pytest
import time
import random
from collections import defaultdict
from symbolic_torch import ProbababilisticContextFreeGrammar
from typing import Literal

# NOTE: All tests should be parameterized with device=["cpu", "cuda"] to ensure both backends are tested.

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

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_simple_grammar(device: Literal["cpu", "cuda"]):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")
    grammar = f"S -> 1 [1.0]"
    pcfg = ProbababilisticContextFreeGrammar(grammar, "S", 1, 0, torch.device(device))
    samples = pcfg.sample_string_expression(10)
    strings = pcfg.to_string(samples)
    # Assert that samples is a constant tensor of the with value CONST_1
    assert samples.shape == (10, 1) and (samples == CONST_1).all()

    # Assert that all strings are "1"
    for s in strings:
        assert s == "1"

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_constant_grammar(device: Literal["cpu", "cuda"]):
    grammar = f"S -> C [1.0]"
    pcfg = ProbababilisticContextFreeGrammar(grammar=grammar, start_symbol="S", padded_maximum_length=1, n_variables=0, device=torch.device(device))
    samples = pcfg.sample_string_expression(10)
    assert samples.shape == (10, 1)
    assert (samples == LEARNABLE_CONSTANT).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_variable_grammar(device: Literal["cpu", "cuda"]):
    for i in range(10):
        grammar = f"S -> X_{i} [1.0]"
        pcfg = ProbababilisticContextFreeGrammar(grammar, "S", 1, 10, torch.device(device))
        samples = pcfg.sample_string_expression(10)
        # Assert that samples is a constant tensor of the with value VAR_START_ID + i
        assert samples.shape == (10, 1)
        assert (samples == VAR_START_ID + i).all()

        # Assert that all strings are "X{i}"
        strings = pcfg.to_string(samples)
        for s in strings:
            assert s == f"X_{i}"

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("op", ["exp", "sin", "cos"])
def test_unary_operator_grammar(device: Literal["cpu", "cuda"], op: str):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")
    grammar = f"S -> {op} ( 1 ) [0.5]" + "\n" + "S -> 1 [0.5]" 
    pcfg = ProbababilisticContextFreeGrammar(grammar, "S", 5, 0, torch.device(device))
    samples = pcfg.sample_string_expression(1000)
    # Assert that samples contains both the operator and the constant with about 50% probability
    op_id = {"exp": EXP, "sin": SIN, "cos": COS}[op]
    assert samples.shape == (1000, 5)
    assert (samples == op_id).sum() > 400 
    assert ((samples == CONST_1)).sum() > 400

    strings = pcfg.to_string(samples)
    for s in strings:
        assert op in s or s == "1"

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("op, op_id", [("add", ADD), ("sub", SUB), ("mul", MUL), ("div", DIV)])
def test_binary_operator_grammar(device: Literal["cpu", "cuda"], op: str, op_id: int):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")
    
    op_symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
    grammar = f"S -> 1 {op_symbol} 1 [0.5]\nS -> 1 [0.5]"
    pcfg = ProbababilisticContextFreeGrammar(grammar, "S", 3, 0, torch.device(device))
    samples = pcfg.sample_string_expression(1000)
    
    assert samples.shape == (1000, 3)
    # Check that the operator appears in the samples
    assert torch.any(samples == op_id, dim=1).sum() > 400
    # Check that the constant '1' appears in the samples
    assert (samples == CONST_1).sum() > 1400

    strings = pcfg.to_string(samples)
    for s in strings:
        assert op_symbol in s or s == "1"

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_test_grammar(device: Literal["cpu", "cuda"]):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")
    pcfg = ProbababilisticContextFreeGrammar(test_grammar, "E", 128, 1, torch.device(device), verbose=True)
    samples = pcfg.sample_string_expression(100)
    assert samples.shape == (100, 128)
    strings = pcfg.to_string(samples)
    for s in strings:
        assert len(s) > 0

    # print 10 random samples
    for i in range(100):
        print(f"Sample {i}: {strings[i]}")

    print(f"--- Benchmarking on {device} ---")
    for batch_size in ([100, 1000, 10000, 100000, 1000000] + ([10000000] if device == "cuda" else [])):
        start = time.time()
        samples = pcfg.sample_string_expression(batch_size)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        print(f"Time for sampling {batch_size} expressions: {end - start:.4f}s")

        start = time.time()
        strings = pcfg.to_string(samples)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        print(f"Time for converting {batch_size} expressions to string: {end - start:.4f}s")

        # print how many strings are equal to X_0
        x0_count = sum(1 for s in strings if s == "X_0")
        print(f"Number of strings equal to 'X_0': {x0_count} / {len(strings)}")
    

def test_undefined_symbol_grammar():
    grammar = "S -> A [1.0]"
    with pytest.raises(ValueError, match="Symbol A with ID 20 has no rules and is not a terminal."):
        ProbababilisticContextFreeGrammar(grammar, "S", 10, 1, torch.device("cpu"))
    with pytest.raises(ValueError, match="Symbol A with ID 20 has no rules and is not a terminal."):
        ProbababilisticContextFreeGrammar(grammar, "S", 10, 1, torch.device("cuda"))

# --- Simple inefficient Python generator for the test_grammar ---
def parse_grammar(grammar_str : str) -> dict[str, list[tuple[list[str], float]]]:
    rules : dict[str, list[tuple[list[str], float]]] = defaultdict(list)
    for line in grammar_str.strip().split("\n"):
        if not line.strip():
            continue
        lhs, rhs_prob = line.split("->")
        rhs, prob = rhs_prob.rsplit("[", 1)
        rhs = rhs.strip()
        prob = float(prob.strip().rstrip("]"))
        rules[lhs.strip()].append((rhs.split(), prob))
    return rules

def sample_from_grammar(rules : dict[str, list[tuple[list[str], float]]], start_symbol: str, length_limit : int) -> list[str]:
    def expand(symbol : str, current_length: int) -> list[str] | None:
        if current_length > length_limit:
            return None
        if symbol not in rules:
            return [symbol]
        
        productions, probs = zip(*rules[symbol])
        chosen = random.choices(productions, weights=probs)[0]
        
        result : list[str] = []
        for sym in chosen:
            expansion = expand(sym, current_length + len(result))
            if expansion is None:
                return None
            result.extend(expansion)
        
        if current_length + len(result) > length_limit:
            return None
            
        return result

    while True:
        generated = expand(start_symbol, 0)
        if generated is not None:
            return generated

# --- Test comparing probabilities ---
def test_python_vs_cpp_cuda_probabilities():
    limit = 128
    grammar = test_grammar
    rules = parse_grammar(grammar)
    n_samples = 100000
    target_exprs = [
        "X_0",
        "X_0 * X_0",
        "sin ( X_0 )",
        "cos ( X_0 )",
        "exp ( X_0 )",
        "( X_0 ) ^2",
        "X_0 + X_0",
        "X_0 - X_0",
        "X_0 / X_0",
    ]
    # Python generator sampling
    py_counts : dict[str, int] = defaultdict(int)
    for _ in range(n_samples):
        expr = sample_from_grammar(rules, "E", limit)
        expr_str = " ".join(expr).strip()
        if expr_str in target_exprs:
            py_counts[expr_str] += 1

    # C++/CUDA implementation sampling
    pcfg_cpu = ProbababilisticContextFreeGrammar(test_grammar, "E", limit, 1, torch.device("cpu"))
    samples_cpu = pcfg_cpu.sample_string_expression(n_samples)
    strings_cpu = pcfg_cpu.to_string(samples_cpu)
    cpp_counts : dict[str, int] = defaultdict(int)
    for s in strings_cpu:
        if s in target_exprs:
            cpp_counts[s] += 1
    
    pcfg_cuda = ProbababilisticContextFreeGrammar(test_grammar, "E", limit, 1, torch.device("cuda"))
    samples_cuda = pcfg_cuda.sample_string_expression(n_samples)
    strings_cuda = pcfg_cuda.to_string(samples_cuda)
    cpp_counts_cuda : dict[str, int] = defaultdict(int)
    for s in strings_cuda:
        if s in target_exprs:
            cpp_counts_cuda[s] += 1

    # assert that the probabilities are similar with relative error < 0.1
    for expr in target_exprs:
        py_prob = py_counts[expr] / n_samples
        cpp_prob = cpp_counts[expr] / n_samples
        cuda_prob = cpp_counts_cuda[expr] / n_samples
        # Check CPU vs Python
        assert abs(py_prob - cpp_prob) / max(py_prob, cpp_prob) < 0.2, f"Probability mismatch for {expr}: {py_prob:.5f} vs {cpp_prob:.5f}"
        # Check CUDA vs Python
        assert abs(py_prob - cuda_prob) / max(py_prob, cuda_prob) < 0.2, f"Probability mismatch for {expr}: {py_prob:.5f} vs {cuda_prob:.5f}"

        # print
        print(f"Expression: {expr}, Python: {py_prob:.5f}, C++: {cpp_prob:.5f}, CUDA: {cuda_prob:.5f}")

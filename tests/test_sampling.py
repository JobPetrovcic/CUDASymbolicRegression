import torch
import pytest
import time
import random
from collections import defaultdict
from symbolic_torch import ProbabilisticContextFreeGrammar
from typing import Literal

from tests.utils import get_cuda_device_with_min_memory

# NOTE: All tests should be parameterized with device=["cpu", "cuda"] to ensure both backends are tested.


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """
    Fixture to provide a device for testing.
    If 'cuda' is requested but not available, it raises an error.
    """
    if request.param == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this system.")
    if request.param == "cpu":
        return torch.device("cpu")
    else:
        index = get_cuda_device_with_min_memory()
        return torch.device(f"cuda:{index}")

from symbolic_torch import Operator

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


def test_simple_grammar(device: Literal["cpu", "cuda"]):
    grammar = f"S -> 1 [1.0]"
    pcfg = ProbabilisticContextFreeGrammar(grammar, "S", 1, 0, torch.device(device))
    samples = pcfg.sample_string_expression(10)
    strings = pcfg.to_string(samples)
    # Assert that samples is a constant tensor of the with value Operator.CONST_1
    assert samples.shape == (10, 1) and (samples == int(Operator.CONST_1)).all()

    # Assert that all strings are "1"
    for s in strings:
        assert s == "1"


def test_constant_grammar(device: Literal["cpu", "cuda"]):
    grammar = f"S -> C [1.0]"
    pcfg = ProbabilisticContextFreeGrammar(grammar=grammar, start_symbol="S", padded_maximum_length=1, n_variables=0, device=torch.device(device))
    samples = pcfg.sample_string_expression(10)
    assert samples.shape == (10, 1)
    assert (samples == int(Operator.LEARNABLE_CONSTANT)).all()



def test_variable_grammar(device: Literal["cpu", "cuda"]):
    for i in range(10):
        grammar = f"S -> X_{i} [1.0]"
        pcfg = ProbabilisticContextFreeGrammar(grammar, "S", 1, 10, torch.device(device))
        samples = pcfg.sample_string_expression(10)
        # Assert that samples is a constant tensor of the with value Operator.VAR_START_ID + i
        assert samples.shape == (10, 1)
        assert (samples == int(Operator.VAR_START_ID) + i).all()

        # Assert that all strings are "X{i}"
        strings = pcfg.to_string(samples)
        for s in strings:
            assert s == f"X_{i}"


@pytest.mark.parametrize("op", ["exp", "sin", "cos"])
def test_unary_operator_grammar(device: Literal["cpu", "cuda"], op: str):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")
    grammar = f"S -> {op} ( 1 ) [0.5]" + "\n" + "S -> 1 [0.5]" 
    pcfg = ProbabilisticContextFreeGrammar(grammar, "S", 5, 0, torch.device(device))
    samples = pcfg.sample_string_expression(1000)
    # Assert that samples contains both the operator and the constant with about 50% probability
    op_id = {"exp": int(Operator.EXP), "sin": int(Operator.SIN), "cos": int(Operator.COS)}[op]
    assert samples.shape == (1000, 5)
    assert (samples == op_id).sum() > 400 
    assert ((samples == int(Operator.CONST_1))).sum() > 400

    strings = pcfg.to_string(samples)
    for s in strings:
        assert op in s or s == "1"


@pytest.mark.parametrize("op, op_id", [("add", int(Operator.ADD)), ("sub", int(Operator.SUB)), ("mul", int(Operator.MUL)), ("div", int(Operator.DIV)), ("pow", int(Operator.POW))])
def test_binary_operator_grammar(device: Literal["cpu", "cuda"], op: str, op_id: int):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")
    
    op_symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "^"}[op]
    grammar = f"S -> 1 {op_symbol} 1 [0.5]\nS -> 1 [0.5]"
    pcfg = ProbabilisticContextFreeGrammar(grammar, "S", 3, 0, torch.device(device))
    samples = pcfg.sample_string_expression(1000)
    
    assert samples.shape == (1000, 3)
    # Check that the operator appears in the samples
    assert torch.any(samples == op_id, dim=1).sum() > 400
    # Check that the constant '1' appears in the samples
    assert (samples == int(Operator.CONST_1)).sum() > 1400

    strings = pcfg.to_string(samples)
    for s in strings:
        assert op_symbol in s or s == "1"

@pytest.mark.large
def test_test_grammar(device: Literal["cpu", "cuda"]):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")
    pcfg = ProbabilisticContextFreeGrammar(test_grammar, "E", 128, 1, torch.device(device), verbose=True)
    samples = pcfg.sample_string_expression(100)
    assert samples.shape == (100, 128)
    strings = pcfg.to_string(samples)
    for s in strings:
        assert len(s) > 0

    # print 5 random samples
    for i in range(5):
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
    with pytest.raises(ValueError, match="Symbol A with ID 54 has no rules and is not a terminal."):
        ProbabilisticContextFreeGrammar(grammar, "S", 10, 1, torch.device("cpu"))
    with pytest.raises(ValueError, match="Symbol A with ID 54 has no rules and is not a terminal."):
        ProbabilisticContextFreeGrammar(grammar, "S", 10, 1, torch.device(f"cuda:{get_cuda_device_with_min_memory()}"))

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
        
def test_generation_max_tries_exceeded():
    """
    Tests that the PCFG raises an error when it cannot generate a valid
    expression within the `max_tries` limit.
    """
    # This grammar can only produce expressions of length 1, 2, 3, ...
    # It cannot terminate.
    grammar = "S -> 1 S [1.0]"
    
    # By setting max_length=5 and max_tries=10, the generator will
    # repeatedly try to generate `1 1 1 1 1 S`, overflow the length, and restart.
    # After 10 tries, it should give up.
    pcfg = ProbabilisticContextFreeGrammar(
        grammar=grammar,
        start_symbol="S",
        padded_maximum_length=5,
        n_variables=0,
        device=torch.device("cpu"),
        max_tries=10
    )

    with pytest.raises(RuntimeError, match="Could not produce a valid expression within the max_tries limit"):
        pcfg.sample_string_expression(1)

# --- Test comparing probabilities ---
@pytest.mark.large
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
    pcfg_cpu = ProbabilisticContextFreeGrammar(test_grammar, "E", limit, 1, torch.device("cpu"))
    samples_cpu = pcfg_cpu.sample_string_expression(n_samples)
    strings_cpu = pcfg_cpu.to_string(samples_cpu)
    cpp_counts : dict[str, int] = defaultdict(int)
    for s in strings_cpu:
        if s in target_exprs:
            cpp_counts[s] += 1

    index = get_cuda_device_with_min_memory()
    
    pcfg_cuda = ProbabilisticContextFreeGrammar(test_grammar, "E", limit, 1, torch.device(f"cuda:{index}"))
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

        print(f"Probabilities of expression {expr}, Python: {py_prob:.5f}, C++: {cpp_prob:.5f}, CUDA: {cuda_prob:.5f}")

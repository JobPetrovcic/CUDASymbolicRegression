from typing import Dict
import pytest
import torch
from symbolic_torch import ProbabilisticContextFreeGrammar
from tests.utils import get_cuda_device_with_min_memory


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


@pytest.fixture(params=["cpu", "cuda"])
def pcfg(request : Dict[str, str]):
    device = request.param
    if device == "cpu":
        device = torch.device("cpu")
    elif device == "cuda":
        index = get_cuda_device_with_min_memory()
        device = torch.device(f"cuda:{index}")
    else:
        raise ValueError("Device must be 'cpu' or 'cuda'.")

    
    return ProbabilisticContextFreeGrammar(
        grammar=test_grammar,
        start_symbol="E",
        padded_maximum_length=20,
        n_variables=1,
        device=device,
    )


def test_available_operators_returns_list(pcfg : ProbabilisticContextFreeGrammar):
    """Test that available_operators returns a list of strings."""
    operators = pcfg.available_operators()
    
    assert isinstance(operators, list)
    assert all(isinstance(symbol, str) for symbol in operators)
    assert len(operators) > 0


def test_available_operators_contains_expected_operators(pcfg : ProbabilisticContextFreeGrammar):
    """Test that available_operators contains expected mathematical operators and operators."""
    operators = pcfg.available_operators()
    
    # Expected operators from the grammar and built-in operators
    expected_operators = [
        '+', '-', '*', '/', '^',  # Binary operators
        'sin', 'cos', 'exp', 'log', 'sqrt',  # Functions from grammar
        '^2',  # Postfix operator from grammar
        'X_0',  # Variable
        'C',  # Learnable constant
        '1', '2', '3', '4', '5',  # Constants
        'pi', 'e',  # Mathematical constants
    ]
    
    symbol_set = set(operators)
    for expected in expected_operators:
        assert expected in symbol_set, f"Expected symbol '{expected}' not found in available operators"


def test_available_operators_consistency_with_get_symbol_id(pcfg : ProbabilisticContextFreeGrammar):
    """Test that all operators returned by available_operators can be looked up with get_symbol_id."""
    operators = pcfg.available_operators()
    
    for symbol in operators:
        # Each symbol should have a valid ID
        symbol_id = pcfg.get_symbol_id(symbol)
        assert isinstance(symbol_id, int)
        assert symbol_id >= 0


def test_available_operators_contains_operators(pcfg : ProbabilisticContextFreeGrammar):
    """Test that available_operators contains both terminal and non-terminal operators from the grammar."""
    operators = pcfg.available_operators()
    symbol_set = set(operators)

    assert 'E' not in symbol_set

    terminals_in_grammar = ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'sqrt', 'log', '^2', 'X_0']
    for term in terminals_in_grammar:
        assert term in symbol_set, f"Terminal '{term}' not found in available operators"


def test_available_operators_deterministic(pcfg : ProbabilisticContextFreeGrammar):
    """Test that available_operators returns the same result when called multiple times."""
    operators1 = pcfg.available_operators()
    operators2 = pcfg.available_operators()
    
    # Should return the same operators (though order might vary due to dict iteration)
    assert set(operators1) == set(operators2)
    assert len(operators1) == len(operators2)


def test_available_operators_includes_special_operators(pcfg : ProbabilisticContextFreeGrammar):
    """Test that available_operators includes special operators like NO_OP."""
    operators = pcfg.available_operators()
    symbol_set = set(operators)
    
    # NO_OP should be included as it's part of the symbol table
    assert 'NO_OP' in symbol_set

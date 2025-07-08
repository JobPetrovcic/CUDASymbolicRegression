import torch
import pytest
from typing import Literal
from symbolic_torch import ProbababilisticContextFreeGrammar

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

# For n_variables = 1
LPAREN = 17
RPAREN = 18

NULL_CHILD = -1

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
V -> A [1.0]
R -> sin [0.2]
R -> cos [0.2]
R -> exp [0.2]
R -> sqrt [0.2]
R -> log [0.2]
P -> square [1.0]
"""

@pytest.fixture
def pcfg(device: Literal["cpu", "cuda"]):
    return ProbababilisticContextFreeGrammar(test_grammar, "E", 20, 1, torch.device(device))

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_parse_to_postfix_constant(device: Literal["cpu", "cuda"], pcfg: ProbababilisticContextFreeGrammar):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")
    
    expression = torch.tensor([[CONST_1]], dtype=torch.int64, device=device)
    ops, children = pcfg.parse_to_postfix(expression)

    expected_ops = torch.tensor([[CONST_1, NO_OP, NO_OP, NO_OP, NO_OP]], dtype=torch.int64, device=device)
    expected_children = torch.tensor([[[NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD]]], dtype=torch.int64, device=device)
    
    assert torch.equal(ops[:, :1], expression)
    assert torch.equal(children[:, 0], torch.tensor([[NULL_CHILD, NULL_CHILD]], dtype=torch.int64, device=device))

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("op, op_id", [("sin", SIN), ("cos", COS), ("exp", EXP), ("log", LOG), ("sqrt", SQRT), ("square", SQUARE)])
def test_parse_to_postfix_unary(device: Literal["cpu", "cuda"], op: str, op_id: int, pcfg: ProbababilisticContextFreeGrammar):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")

    expression = torch.tensor([[op_id, LPAREN, CONST_1, RPAREN]], dtype=torch.int64, device=device)
    ops, children = pcfg.parse_to_postfix(expression)

    expected_ops = torch.tensor([[CONST_1, op_id]], dtype=torch.int64, device=device)
    expected_children = torch.tensor([[[NULL_CHILD, NULL_CHILD], [0, NULL_CHILD]]], dtype=torch.int64, device=device)

    assert torch.equal(ops[:, :2], expected_ops)
    assert torch.equal(children[:, :2], expected_children)

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("op, op_id", [("add", ADD), ("sub", SUB), ("mul", MUL), ("div", DIV)])
def test_parse_to_postfix_binary(device: Literal["cpu", "cuda"], op: str, op_id: int, pcfg: ProbababilisticContextFreeGrammar):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")

    expression = torch.tensor([[CONST_1, op_id, CONST_1]], dtype=torch.int64, device=device)
    ops, children = pcfg.parse_to_postfix(expression)

    expected_ops = torch.tensor([[CONST_1, CONST_1, op_id]], dtype=torch.int64, device=device)
    expected_children = torch.tensor([[[NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [0, 1]]], dtype=torch.int64, device=device)

    assert torch.equal(ops[:, :3], expected_ops)
    assert torch.equal(children[:, :3], expected_children)

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_parse_to_postfix_parenthesis(device: Literal["cpu", "cuda"], pcfg: ProbababilisticContextFreeGrammar):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")

    # (1 + 1) * 1
    expression = torch.tensor([[LPAREN, CONST_1, ADD, CONST_1, RPAREN, MUL, CONST_1]], dtype=torch.int64, device=device)
    ops, children = pcfg.parse_to_postfix(expression)

    expected_ops = torch.tensor([[CONST_1, CONST_1, ADD, CONST_1, MUL]], dtype=torch.int64, device=device)
    expected_children = torch.tensor([[[NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [0, 1], [NULL_CHILD, NULL_CHILD], [2, 3]]], dtype=torch.int64, device=device)

    assert torch.equal(ops[:, :5], expected_ops)
    assert torch.equal(children[:, :5], expected_children)

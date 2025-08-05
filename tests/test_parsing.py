import torch
import pytest
from symbolic_torch import ProbabilisticContextFreeGrammar, Operator
import time
from typing import List

# NOTE: All tests should be parameterized with device=["cpu", "cuda"] to ensure both backends are tested.

NULL_CHILD = -1
NULL_PARENT = -1

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
def pcfg(request: pytest.FixtureRequest):
    device = request.param
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")
    return ProbabilisticContextFreeGrammar(test_grammar, "E", 20, 1, torch.device(device))


def test_parse_to_postfix_constant(pcfg: ProbabilisticContextFreeGrammar):
    expression = torch.tensor([[int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_postfix(expression)

    assert torch.equal(ops[:, :1], expression)
    assert torch.equal(children[:, 0], torch.tensor([[NULL_CHILD, NULL_CHILD]], dtype=torch.int64, device=pcfg.device))


def test_parse_to_prefix_constant(pcfg: ProbabilisticContextFreeGrammar):
    expression = torch.tensor([[int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_prefix(expression)

    assert torch.equal(ops[:, :1], expression)
    assert torch.equal(children[:, 0], torch.tensor([[NULL_CHILD, NULL_CHILD]], dtype=torch.int64, device=pcfg.device))


@pytest.mark.parametrize("op_id", [int(Operator.SIN), int(Operator.COS), int(Operator.EXP), int(Operator.LOG), int(Operator.SQRT), int(Operator.SQUARE)])
def test_parse_to_postfix_unary(op_id: int, pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')
    expression = torch.tensor([[op_id, LPAREN, int(Operator.CONST_1), RPAREN]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_postfix(expression)

    expected_ops = torch.tensor([[int(Operator.CONST_1), op_id]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[[NULL_CHILD, NULL_CHILD], [0, NULL_CHILD]]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :2], expected_ops)
    assert torch.equal(children[:, :2], expected_children)


@pytest.mark.parametrize("op_id", [Operator.SIN, Operator.COS, Operator.EXP, Operator.LOG, Operator.SQRT, Operator.SQUARE])
def test_parse_to_prefix_unary(op_id: int, pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')
    expression = torch.tensor([[op_id, LPAREN, int(Operator.CONST_1), RPAREN]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_prefix(expression)

    expected_ops = torch.tensor([[op_id, int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[[1, NULL_CHILD], [NULL_CHILD, NULL_CHILD]]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :2], expected_ops)
    assert torch.equal(children[:, :2], expected_children)


@pytest.mark.parametrize("op_id", [Operator.ADD, Operator.SUB, Operator.MUL, Operator.DIV, Operator.POW])
def test_parse_to_postfix_binary(op_id: int, pcfg: ProbabilisticContextFreeGrammar):
    expression = torch.tensor([[int(Operator.CONST_1), op_id, int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_postfix(expression)

    expected_ops = torch.tensor([[int(Operator.CONST_1), int(Operator.CONST_1), op_id]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[[NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [0, 1]]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :3], expected_ops)
    assert torch.equal(children[:, :3], expected_children)


@pytest.mark.parametrize("op_id", [Operator.ADD, Operator.SUB, Operator.MUL, Operator.DIV, Operator.POW])
def test_parse_to_prefix_binary(op_id: int, pcfg: ProbabilisticContextFreeGrammar):
    expression = torch.tensor([[int(Operator.CONST_1), op_id, int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_prefix(expression)

    expected_ops = torch.tensor([[op_id, int(Operator.CONST_1), int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[[1, 2], [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD]]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :3], expected_ops)
    assert torch.equal(children[:, :3], expected_children)


def test_parse_to_postfix_parenthesis(pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')
    # (1 + 1) * 1
    expression = torch.tensor([[LPAREN, int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), RPAREN, int(Operator.MUL), int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_postfix(expression)

    expected_ops = torch.tensor([[int(Operator.CONST_1), int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), int(Operator.MUL)]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[[NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [0, 1], [NULL_CHILD, NULL_CHILD], [2, 3]]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :5], expected_ops)
    assert torch.equal(children[:, :5], expected_children)

def test_parse_to_prefix_parenthesis(pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')
    # (1 + 1) * 1
    expression = torch.tensor([[LPAREN, int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), RPAREN, int(Operator.MUL), int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_prefix(expression)

    expected_ops = torch.tensor([[int(Operator.MUL), int(Operator.ADD), int(Operator.CONST_1), int(Operator.CONST_1), int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[[1, 4], [2, 3], [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD]]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :5], expected_ops)
    assert torch.equal(children[:, :5], expected_children)

def test_parse_to_postfix_all_operators(pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')
    # (sin(X_0) + 1) * (log(X_0) - 1)
    X_0 = int(Operator.VAR_START_ID)
    expression = torch.tensor([[
        LPAREN, int(Operator.SIN), LPAREN, X_0, RPAREN, int(Operator.ADD), int(Operator.CONST_1), RPAREN,
        int(Operator.MUL),
        LPAREN, int(Operator.LOG), LPAREN, X_0, RPAREN, int(Operator.SUB), int(Operator.CONST_1), RPAREN
    ]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_postfix(expression)

    expected_ops = torch.tensor([[X_0, int(Operator.SIN), int(Operator.CONST_1), int(Operator.ADD), X_0, int(Operator.LOG), int(Operator.CONST_1), int(Operator.SUB), int(Operator.MUL)]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[
        [NULL_CHILD, NULL_CHILD], [0, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [1, 2],
        [NULL_CHILD, NULL_CHILD], [4, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [5, 6],
        [3, 7]
    ]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :9], expected_ops)
    assert torch.equal(children[:, :9], expected_children)


def test_parse_to_prefix_all_operators(pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')
    # (sin(X_0) + 1) * (log(X_0) - 1)
    X_0 = int(Operator.VAR_START_ID)
    expression = torch.tensor([[LPAREN, int(Operator.SIN), LPAREN, X_0, RPAREN, int(Operator.ADD), int(Operator.CONST_1), RPAREN, int(Operator.MUL), LPAREN, int(Operator.LOG), LPAREN, X_0, RPAREN, int(Operator.SUB), int(Operator.CONST_1), RPAREN]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_prefix(expression)

    # Prefix: * + sin X_0 1 - log X_0 1
    expected_ops = torch.tensor([[int(Operator.MUL), int(Operator.ADD), int(Operator.SIN), X_0, int(Operator.CONST_1), int(Operator.SUB), int(Operator.LOG), X_0, int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([
        [
            [1, 5],  # *
            [2, 4],  # +
            [3, NULL_CHILD],  # sin
            [NULL_CHILD, NULL_CHILD],  # X_0
            [NULL_CHILD, NULL_CHILD],  # 1
            [6, 8],  # -
            [7, NULL_CHILD],  # log
            [NULL_CHILD, NULL_CHILD],  # X_0
            [NULL_CHILD, NULL_CHILD]   # 1
        ]
    ], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :9], expected_ops)
    assert torch.equal(children[:, :9], expected_children)


def test_parse_to_postfix_precedence(pcfg: ProbabilisticContextFreeGrammar):
    # 1 + 1 * 1
    expression = torch.tensor([[int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), int(Operator.MUL), int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_postfix(expression)

    # Postfix: 1, 1, 1, *, +
    expected_ops = torch.tensor([[int(Operator.CONST_1), int(Operator.CONST_1), int(Operator.CONST_1), int(Operator.MUL), int(Operator.ADD)]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[
        [NULL_CHILD, NULL_CHILD], # 1
        [NULL_CHILD, NULL_CHILD], # 1
        [NULL_CHILD, NULL_CHILD], # 1
        [1, 2],                   # *
        [0, 3]                    # +
    ]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :5], expected_ops)
    assert torch.equal(children[:, :5], expected_children)


def test_parse_to_prefix_precedence(pcfg: ProbabilisticContextFreeGrammar):
    # 1 + 1 * 1
    expression = torch.tensor([[int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), int(Operator.MUL), int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_prefix(expression)

    # Prefix: + 1 * 1 1
    expected_ops = torch.tensor([[int(Operator.ADD), int(Operator.CONST_1), int(Operator.MUL), int(Operator.CONST_1), int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[
        [1, 2],                   # +
        [NULL_CHILD, NULL_CHILD], # 1
        [3, 4],                   # *
        [NULL_CHILD, NULL_CHILD], # 1
        [NULL_CHILD, NULL_CHILD]  # 1
    ]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :5], expected_ops)
    assert torch.equal(children[:, :5], expected_children)


def test_parse_to_postfix_associativity(pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')

    # Left associativity: 1 + 1 + 1 + 1 -> ((1+1)+1)+1
    expression_add = torch.tensor([[int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    ops_add, children_add = pcfg.parse_to_postfix(expression_add)
    expected_ops_add = torch.tensor([[int(Operator.CONST_1), int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), int(Operator.ADD)]], dtype=torch.int64, device=pcfg.device)
    expected_children_add = torch.tensor([[
        [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [0, 1],
        [NULL_CHILD, NULL_CHILD], [2, 3],
        [NULL_CHILD, NULL_CHILD], [4, 5]
    ]], dtype=torch.int64, device=pcfg.device)
    assert torch.equal(ops_add[:, :7], expected_ops_add)
    assert torch.equal(children_add[:, :7], expected_children_add)

    # Right associativity: sin(sin(sin(sin(1))))
    expression_sin = torch.tensor([[int(Operator.SIN), LPAREN, int(Operator.SIN), LPAREN, int(Operator.SIN), LPAREN, int(Operator.SIN), LPAREN, int(Operator.CONST_1), RPAREN, RPAREN, RPAREN, RPAREN]], dtype=torch.int64, device=pcfg.device)
    ops_sin, children_sin = pcfg.parse_to_postfix(expression_sin)
    expected_ops_sin = torch.tensor([[int(Operator.CONST_1), int(Operator.SIN), int(Operator.SIN), int(Operator.SIN), int(Operator.SIN)]], dtype=torch.int64, device=pcfg.device)
    expected_children_sin = torch.tensor([[
        [NULL_CHILD, NULL_CHILD], [0, NULL_CHILD], [1, NULL_CHILD], [2, NULL_CHILD], [3, NULL_CHILD]
    ]], dtype=torch.int64, device=pcfg.device)
    assert torch.equal(ops_sin[:, :5], expected_ops_sin)
    assert torch.equal(children_sin[:, :5], expected_children_sin)


def test_parse_to_prefix_associativity(pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')

    # Left associativity: 1 + 1 + 1 + 1 -> ((1+1)+1)+1
    expression_add = torch.tensor([[int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    ops_add, children_add = pcfg.parse_to_prefix(expression_add)
    # Prefix: + + + 1 1 1 1
    expected_ops_add = torch.tensor([[int(Operator.ADD), int(Operator.ADD), int(Operator.ADD), int(Operator.CONST_1), int(Operator.CONST_1), int(Operator.CONST_1), int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    expected_children_add = torch.tensor([[
        [1, 6], [2, 5], [3, 4],
        [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD],
        [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD]
    ]], dtype=torch.int64, device=pcfg.device)
    assert torch.equal(ops_add[:, :7], expected_ops_add)
    assert torch.equal(children_add[:, :7], expected_children_add)

    # Right associativity: sin(sin(sin(sin(1))))
    expression_sin = torch.tensor([[int(Operator.SIN), LPAREN, int(Operator.SIN), LPAREN, int(Operator.SIN), LPAREN, int(Operator.SIN), LPAREN, int(Operator.CONST_1), RPAREN, RPAREN, RPAREN, RPAREN]], dtype=torch.int64, device=pcfg.device)
    ops_sin, children_sin = pcfg.parse_to_prefix(expression_sin)
    # Prefix: sin sin sin sin 1
    expected_ops_sin = torch.tensor([[int(Operator.SIN), int(Operator.SIN), int(Operator.SIN), int(Operator.SIN), int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    expected_children_sin = torch.tensor([[
        [1, NULL_CHILD], [2, NULL_CHILD], [3, NULL_CHILD], [4, NULL_CHILD], [NULL_CHILD, NULL_CHILD]
    ]], dtype=torch.int64, device=pcfg.device)
    assert torch.equal(ops_sin[:, :5], expected_ops_sin)
    assert torch.equal(children_sin[:, :5], expected_children_sin)


def test_parse_to_postfix_mixed_operators(pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')

    # sin(1) + 1
    expression = torch.tensor([[int(Operator.SIN), LPAREN, int(Operator.CONST_1), RPAREN, int(Operator.ADD), int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_postfix(expression)

    # Postfix: 1, sin, 1, +
    expected_ops = torch.tensor([[int(Operator.CONST_1), int(Operator.SIN), int(Operator.CONST_1), int(Operator.ADD)]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[
        [NULL_CHILD, NULL_CHILD], # 1
        [0, NULL_CHILD],          # sin
        [NULL_CHILD, NULL_CHILD], # 1
        [1, 2]                    # +
    ]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :4], expected_ops)
    assert torch.equal(children[:, :4], expected_children)


def test_parse_to_prefix_mixed_operators(pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')

    # sin(1) + 1
    expression = torch.tensor([[int(Operator.SIN), LPAREN, int(Operator.CONST_1), RPAREN, int(Operator.ADD), int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_prefix(expression)

    # Prefix: + sin 1 1
    expected_ops = torch.tensor([[int(Operator.ADD), int(Operator.SIN), int(Operator.CONST_1), int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[
        [1, 3],                   # +
        [2, NULL_CHILD],          # sin
        [NULL_CHILD, NULL_CHILD], # 1
        [NULL_CHILD, NULL_CHILD]  # 1
    ]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :4], expected_ops)
    assert torch.equal(children[:, :4], expected_children)


def test_parse_to_postfix_complex_parenthesis(pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')

    # (1 + 1) + (1 + (1 + 1))
    expression = torch.tensor([[
        LPAREN, int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), RPAREN,
        int(Operator.ADD),
        LPAREN, int(Operator.CONST_1), int(Operator.ADD), LPAREN, int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), RPAREN, RPAREN
    ]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_postfix(expression)

    # Postfix: 1 1 + 1 1 1 + + +
    expected_ops = torch.tensor([[int(Operator.CONST_1), int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), int(Operator.CONST_1), int(Operator.CONST_1), int(Operator.ADD), int(Operator.ADD), int(Operator.ADD)]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[
        [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [0, 1],
        [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [4, 5],
        [3, 6],
        [2, 7]
    ]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :9], expected_ops)
    assert torch.equal(children[:, :9], expected_children)


def test_parse_to_prefix_complex_parenthesis(pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')

    # (1 + 1) + (1 + (1 + 1))
    expression = torch.tensor([[
        LPAREN, int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), RPAREN,
        int(Operator.ADD),
        LPAREN, int(Operator.CONST_1), int(Operator.ADD), LPAREN, int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), RPAREN, RPAREN
    ]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_prefix(expression)

    # Prefix: + + 1 1 + 1 + 1 1
    expected_ops = torch.tensor([[int(Operator.ADD), int(Operator.ADD), int(Operator.CONST_1), int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[
        [1, 4], [2, 3],
        [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD],
        [5, 6],
        [NULL_CHILD, NULL_CHILD],
        [7, 8],
        [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD]
    ]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :9], expected_ops)
    assert torch.equal(children[:, :9], expected_children)


def test_parse_to_postfix_batch(pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')

    expressions_list = [
        torch.tensor([[int(Operator.CONST_1)]], dtype=torch.int64),
        torch.tensor([[int(Operator.SIN), LPAREN, int(Operator.CONST_1), RPAREN]], dtype=torch.int64),
        torch.tensor([[int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1)]], dtype=torch.int64),
        torch.tensor([[LPAREN, int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), RPAREN, int(Operator.MUL), int(Operator.CONST_1)]], dtype=torch.int64),
    ]
    max_len = max(e.shape[1] for e in expressions_list)
    expressions = torch.full((len(expressions_list), max_len), int(Operator.NO_OP), dtype=torch.int64, device=pcfg.device)
    for i, e in enumerate(expressions_list):
        expressions[i, :e.shape[1]] = e

    batch_ops, batch_children = pcfg.parse_to_postfix(expressions)

    for i, expr in enumerate(expressions_list):
        single_ops, single_children = pcfg.parse_to_postfix(expr.to(pcfg.device))
        len_single_ops = single_ops.shape[1]
        # Check that the output is correct
        assert torch.equal(batch_ops[i, :len_single_ops], single_ops[0])
        assert torch.equal(batch_children[i, :len_single_ops], single_children[0])
        # Check padding
        assert torch.all(batch_ops[i, len_single_ops:] == int(Operator.NO_OP))


def test_parse_to_prefix_batch(pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')
    expressions_list = [
        torch.tensor([[int(Operator.CONST_1)]], dtype=torch.int64),
        torch.tensor([[int(Operator.SIN), LPAREN, int(Operator.CONST_1), RPAREN]], dtype=torch.int64),
        torch.tensor([[int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1)]], dtype=torch.int64),
        torch.tensor([[LPAREN, int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), RPAREN, int(Operator.MUL), int(Operator.CONST_1)]], dtype=torch.int64),
    ]
    max_len = max(e.shape[1] for e in expressions_list)
    expressions = torch.full((len(expressions_list), max_len), int(Operator.NO_OP), dtype=torch.int64, device=pcfg.device)
    for i, e in enumerate(expressions_list):
        expressions[i, :e.shape[1]] = e

    batch_ops, batch_children = pcfg.parse_to_prefix(expressions)

    for i, expr in enumerate(expressions_list):
        single_ops, single_children = pcfg.parse_to_prefix(expr.to(pcfg.device))
        len_single_ops = single_ops.shape[1]
        # Check that the output is correct
        assert torch.equal(batch_ops[i, :len_single_ops], single_ops[0])
        assert torch.equal(batch_children[i, :len_single_ops], single_children[0])
        # Check padding
        assert torch.all(batch_ops[i, len_single_ops:] == int(Operator.NO_OP))


def test_parse_to_postfix_max_length(pcfg: ProbabilisticContextFreeGrammar):
    # Create an expression of max length (128)
    # e.g. 1 + 1 + ... + 1 (64 ones, 63 pluses) -> 127 length
    num_ones = 64
    expr_list = [int(Operator.CONST_1)] * num_ones
    for i in range(num_ones - 1):
        expr_list.insert(2 * i + 1, int(Operator.ADD))

    expression = torch.tensor([expr_list], dtype=torch.int64, device=pcfg.device)
    assert expression.shape[1] == 2 * num_ones - 1

    ops, children = pcfg.parse_to_postfix(expression)

    # Check that it doesn't error out and the result is plausible
    # The postfix should be 1, 1, +, 1, +, ..., 1, +
    # Total length is also 127
    assert ops.shape[1] == 2 * num_ones -1
    assert children.shape[1] == 2 * num_ones -1
    assert ops[0, -1] == int(Operator.ADD) # last op is Operator.ADD


def test_parse_to_prefix_max_length(pcfg: ProbabilisticContextFreeGrammar):
    # Create an expression of max length (128)
    # e.g. 1 + 1 + ... + 1 (64 ones, 63 pluses) -> 127 length
    num_ones = 64
    expr_list = [Operator.CONST_1] * num_ones
    for i in range(num_ones - 1):
        expr_list.insert(2 * i + 1, Operator.ADD)

    expression = torch.tensor([expr_list], dtype=torch.int64, device=pcfg.device)
    assert expression.shape[1] == 2 * num_ones - 1

    ops, children = pcfg.parse_to_prefix(expression)

    # Check that it doesn't error out and the result is plausible
    # The prefix should be + + ... + 1 1 ... 1
    # Total length is also 127
    assert ops.shape[1] == 2 * num_ones - 1
    assert children.shape[1] == 2 * num_ones - 1
    assert ops[0, 0] == int(Operator.ADD) # first op is Operator.ADD

@pytest.mark.large
def test_benchmark_parsing_cpu_vs_cuda():
    import time

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    N_BENCH_LOOPS = 10

    
    pcfg_cpu = ProbabilisticContextFreeGrammar(test_grammar, "E", 20, 1, torch.device("cpu"))
    pcfg_cuda = ProbabilisticContextFreeGrammar(test_grammar, "E", 20, 1, torch.device("cuda"))
    LPAREN = pcfg_cpu.get_symbol_id('(')
    RPAREN = pcfg_cpu.get_symbol_id(')')
    
    # Batch of complex expressions
    expressions_list = [
        torch.tensor([[LPAREN, int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), RPAREN, int(Operator.MUL), int(Operator.CONST_1)]]),
        torch.tensor([[int(Operator.SIN), LPAREN, int(Operator.COS), LPAREN, int(Operator.CONST_1), RPAREN, RPAREN, int(Operator.ADD), int(Operator.LOG), LPAREN, int(Operator.CONST_1), RPAREN]]),
        torch.tensor([[LPAREN, LPAREN, int(Operator.CONST_1), int(Operator.MUL), int(Operator.CONST_1), RPAREN, int(Operator.ADD), int(Operator.CONST_1), RPAREN, int(Operator.DIV), int(Operator.CONST_1)]]),
    ] * 1000000 # Make a large batch

    max_len = max(e.shape[1] for e in expressions_list)
    expressions = torch.full((len(expressions_list), max_len), int(Operator.NO_OP), dtype=torch.int64)
    for i, e in enumerate(expressions_list):
        expressions[i, :e.shape[1]] = e

    expressions_cpu = expressions.to("cpu")
    expressions_cuda = expressions.to("cuda")

    # Warmup
    pcfg_cpu.parse_to_postfix(expressions_cpu)
    pcfg_cuda.parse_to_postfix(expressions_cuda)
    torch.cuda.synchronize()

    # CPU benchmark
    start_time = time.time()
    for _ in range(N_BENCH_LOOPS):
        pcfg_cpu.parse_to_postfix(expressions_cpu)
    cpu_time = (time.time() - start_time) / N_BENCH_LOOPS
    print(f"\nCPU parsing time: {cpu_time:.4f}s")

    # CUDA benchmark
    start_time = time.time()
    for _ in range(N_BENCH_LOOPS):
        pcfg_cuda.parse_to_postfix(expressions_cuda)
    torch.cuda.synchronize()
    cuda_time = (time.time() - start_time) / N_BENCH_LOOPS
    print(f"CUDA parsing time: {cuda_time:.4f}s")

    # NOTE: for lower values CPU is a bit faster, but for larger batches CUDA becomes significantly faster.
    
# ===============================================
# Tests for parent-pointer parsing functions
# ===============================================

def test_parse_to_postfix_parent_constant(pcfg: ProbabilisticContextFreeGrammar):
    expression = torch.tensor([[int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    ops, parents = pcfg.parse_to_postfix_parent(expression)
    
    assert torch.equal(ops[:, :1], expression)
    assert torch.equal(parents[:, 0], torch.tensor([NULL_PARENT], dtype=torch.int64, device=pcfg.device))


def test_parse_to_prefix_parent_constant(pcfg: ProbabilisticContextFreeGrammar):
    expression = torch.tensor([[int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    ops, parents = pcfg.parse_to_prefix_parent(expression)
    
    assert torch.equal(ops[:, :1], expression)
    assert torch.equal(parents[:, 0], torch.tensor([NULL_PARENT], dtype=torch.int64, device=pcfg.device))


@pytest.mark.parametrize("op_id", [int(Operator.SIN), int(Operator.COS), int(Operator.EXP), int(Operator.LOG), int(Operator.SQRT), int(Operator.SQUARE)])
def test_parse_to_postfix_parent_unary(op_id: int, pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')
    expression = torch.tensor([[op_id, LPAREN, int(Operator.CONST_1), RPAREN]], dtype=torch.int64, device=pcfg.device)
    ops, parents = pcfg.parse_to_postfix_parent(expression)

    expected_ops = torch.tensor([[int(Operator.CONST_1), op_id]], dtype=torch.int64, device=pcfg.device)
    expected_parents = torch.tensor([[1, NULL_PARENT]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :2], expected_ops)
    assert torch.equal(parents[:, :2], expected_parents)


@pytest.mark.parametrize("op_id", [Operator.SIN, Operator.COS, Operator.EXP, Operator.LOG, Operator.SQRT, Operator.SQUARE])
def test_parse_to_prefix_parent_unary(op_id: int, pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')
    expression = torch.tensor([[op_id, LPAREN, int(Operator.CONST_1), RPAREN]], dtype=torch.int64, device=pcfg.device)
    ops, parents = pcfg.parse_to_prefix_parent(expression)

    expected_ops = torch.tensor([[op_id, int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    expected_parents = torch.tensor([[NULL_PARENT, 0]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :2], expected_ops)
    assert torch.equal(parents[:, :2], expected_parents)


@pytest.mark.parametrize("op_id", [Operator.ADD, Operator.SUB, Operator.MUL, Operator.DIV, Operator.POW])
def test_parse_to_postfix_parent_binary(op_id: int, pcfg: ProbabilisticContextFreeGrammar):
    expression = torch.tensor([[int(Operator.CONST_1), op_id, int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    ops, parents = pcfg.parse_to_postfix_parent(expression)

    expected_ops = torch.tensor([[int(Operator.CONST_1), int(Operator.CONST_1), op_id]], dtype=torch.int64, device=pcfg.device)
    expected_parents = torch.tensor([[2, 2, NULL_PARENT]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :3], expected_ops)
    assert torch.equal(parents[:, :3], expected_parents)


@pytest.mark.parametrize("op_id", [Operator.ADD, Operator.SUB, Operator.MUL, Operator.DIV, Operator.POW])
def test_parse_to_prefix_parent_binary(op_id: int, pcfg: ProbabilisticContextFreeGrammar):
    expression = torch.tensor([[int(Operator.CONST_1), op_id, int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    ops, parents = pcfg.parse_to_prefix_parent(expression)

    expected_ops = torch.tensor([[op_id, int(Operator.CONST_1), int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    expected_parents = torch.tensor([[NULL_PARENT, 0, 0]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :3], expected_ops)
    assert torch.equal(parents[:, :3], expected_parents)


def test_parse_to_postfix_parent_all_operators(pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')
    # (sin(X_0) + 1) * (log(X_0) - 1)
    X_0 = int(Operator.VAR_START_ID)
    expression = torch.tensor([[
        LPAREN, int(Operator.SIN), LPAREN, X_0, RPAREN, int(Operator.ADD), int(Operator.CONST_1), RPAREN,
        int(Operator.MUL),
        LPAREN, int(Operator.LOG), LPAREN, X_0, RPAREN, int(Operator.SUB), int(Operator.CONST_1), RPAREN
    ]], dtype=torch.int64, device=pcfg.device)
    ops, parents = pcfg.parse_to_postfix_parent(expression)

    # Postfix ops: [X_0, sin, 1, +, X_0, log, 1, -, *]
    expected_ops = torch.tensor([[X_0, int(Operator.SIN), int(Operator.CONST_1), int(Operator.ADD), X_0, int(Operator.LOG), int(Operator.CONST_1), int(Operator.SUB), int(Operator.MUL)]], dtype=torch.int64, device=pcfg.device)
    # Postfix parents: [1, 3, 3, 8, 5, 7, 7, 8, NP]
    expected_parents = torch.tensor([[1, 3, 3, 8, 5, 7, 7, 8, NULL_PARENT]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :9], expected_ops)
    assert torch.equal(parents[:, :9], expected_parents)


def test_parse_to_prefix_parent_all_operators(pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')
    # (sin(X_0) + 1) * (log(X_0) - 1)
    X_0 = int(Operator.VAR_START_ID)
    expression = torch.tensor([[LPAREN, int(Operator.SIN), LPAREN, X_0, RPAREN, int(Operator.ADD), int(Operator.CONST_1), RPAREN, int(Operator.MUL), LPAREN, int(Operator.LOG), LPAREN, X_0, RPAREN, int(Operator.SUB), int(Operator.CONST_1), RPAREN]], dtype=torch.int64, device=pcfg.device)
    ops, parents = pcfg.parse_to_prefix_parent(expression)

    # Prefix ops: [*, +, sin, X_0, 1, -, log, X_0, 1]
    expected_ops = torch.tensor([[int(Operator.MUL), int(Operator.ADD), int(Operator.SIN), X_0, int(Operator.CONST_1), int(Operator.SUB), int(Operator.LOG), X_0, int(Operator.CONST_1)]], dtype=torch.int64, device=pcfg.device)
    # Prefix parents: [NP, 0, 1, 2, 1, 0, 5, 6, 5]
    expected_parents = torch.tensor([[NULL_PARENT, 0, 1, 2, 1, 0, 5, 6, 5]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :9], expected_ops)
    assert torch.equal(parents[:, :9], expected_parents)


def test_parse_to_postfix_parent_batch(pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')

    expressions_list = [
        torch.tensor([[int(Operator.CONST_1)]], dtype=torch.int64),
        torch.tensor([[int(Operator.SIN), LPAREN, int(Operator.CONST_1), RPAREN]], dtype=torch.int64),
        torch.tensor([[int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1)]], dtype=torch.int64),
        torch.tensor([[LPAREN, int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), RPAREN, int(Operator.MUL), int(Operator.CONST_1)]], dtype=torch.int64),
    ]
    max_len = max(e.shape[1] for e in expressions_list)
    expressions = torch.full((len(expressions_list), max_len), int(Operator.NO_OP), dtype=torch.int64, device=pcfg.device)
    for i, e in enumerate(expressions_list):
        expressions[i, :e.shape[1]] = e

    batch_ops, batch_parents = pcfg.parse_to_postfix_parent(expressions)

    for i, expr in enumerate(expressions_list):
        single_ops, single_parents = pcfg.parse_to_postfix_parent(expr.to(pcfg.device))
        len_single_ops = single_ops.shape[1]
        # Check that the output is correct
        assert torch.equal(batch_ops[i, :len_single_ops], single_ops[0])
        assert torch.equal(batch_parents[i, :len_single_ops], single_parents[0])
        # Check padding
        assert torch.all(batch_ops[i, len_single_ops:] == int(Operator.NO_OP))
        assert torch.all(batch_parents[i, len_single_ops:] == int(Operator.NO_OP))


def test_parse_to_prefix_parent_batch(pcfg: ProbabilisticContextFreeGrammar):
    LPAREN = pcfg.get_symbol_id('(')
    RPAREN = pcfg.get_symbol_id(')')
    expressions_list = [
        torch.tensor([[int(Operator.CONST_1)]], dtype=torch.int64),
        torch.tensor([[int(Operator.SIN), LPAREN, int(Operator.CONST_1), RPAREN]], dtype=torch.int64),
        torch.tensor([[int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1)]], dtype=torch.int64),
        torch.tensor([[LPAREN, int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), RPAREN, int(Operator.MUL), int(Operator.CONST_1)]], dtype=torch.int64),
    ]
    max_len = max(e.shape[1] for e in expressions_list)
    expressions = torch.full((len(expressions_list), max_len), int(Operator.NO_OP), dtype=torch.int64, device=pcfg.device)
    for i, e in enumerate(expressions_list):
        expressions[i, :e.shape[1]] = e

    batch_ops, batch_parents = pcfg.parse_to_prefix_parent(expressions)

    for i, expr in enumerate(expressions_list):
        single_ops, single_parents = pcfg.parse_to_prefix_parent(expr.to(pcfg.device))
        len_single_ops = single_ops.shape[1]
        # Check that the output is correct
        assert torch.equal(batch_ops[i, :len_single_ops], single_ops[0])
        assert torch.equal(batch_parents[i, :len_single_ops], single_parents[0])
        # Check padding
        assert torch.all(batch_ops[i, len_single_ops:] == int(Operator.NO_OP))
        assert torch.all(batch_parents[i, len_single_ops:] == int(Operator.NO_OP))

@pytest.mark.large
def test_benchmark_parsing_parent_cpu_vs_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    N_BENCH_LOOPS = 10
    
    pcfg_cpu = ProbabilisticContextFreeGrammar(test_grammar, "E", 20, 1, torch.device("cpu"))
    pcfg_cuda = ProbabilisticContextFreeGrammar(test_grammar, "E", 20, 1, torch.device("cuda"))
    LPAREN = pcfg_cpu.get_symbol_id('(')
    RPAREN = pcfg_cpu.get_symbol_id(')')
    
    expressions_list = [
        torch.tensor([[LPAREN, int(Operator.CONST_1), int(Operator.ADD), int(Operator.CONST_1), RPAREN, int(Operator.MUL), int(Operator.CONST_1)]]),
        torch.tensor([[int(Operator.SIN), LPAREN, int(Operator.COS), LPAREN, int(Operator.CONST_1), RPAREN, RPAREN, int(Operator.ADD), int(Operator.LOG), LPAREN, int(Operator.CONST_1), RPAREN]]),
    ] * 1000000

    max_len = max(e.shape[1] for e in expressions_list)
    expressions = torch.full((len(expressions_list), max_len), int(Operator.NO_OP), dtype=torch.int64)
    for i, e in enumerate(expressions_list):
        expressions[i, :e.shape[1]] = e

    expressions_cpu = expressions.to("cpu")
    expressions_cuda = expressions.to("cuda")

    # Warmup
    pcfg_cpu.parse_to_postfix_parent(expressions_cpu)
    pcfg_cuda.parse_to_postfix_parent(expressions_cuda)
    torch.cuda.synchronize()

    # CPU benchmark
    start_time = time.time()
    for _ in range(N_BENCH_LOOPS):
        pcfg_cpu.parse_to_postfix_parent(expressions_cpu)
    cpu_time = (time.time() - start_time) / N_BENCH_LOOPS
    print(f"\nCPU parent parsing time: {cpu_time:.4f}s")

    # CUDA benchmark
    start_time = time.time()
    for _ in range(N_BENCH_LOOPS):
        pcfg_cuda.parse_to_postfix_parent(expressions_cuda)
    torch.cuda.synchronize()
    cuda_time = (time.time() - start_time) / N_BENCH_LOOPS
    print(f"CUDA parent parsing time: {cuda_time:.4f}s")


# ===============================================
# Tests for Error Reporting and Verbosity
# ===============================================
def to_ids(pcfg_obj: ProbabilisticContextFreeGrammar, tokens: List[str]) -> torch.Tensor:
    return torch.tensor([pcfg_obj.get_symbol_id(s) for s in tokens], dtype=torch.int64, device=pcfg_obj.device)

def test_parsing_error_verbosity(pcfg: ProbabilisticContextFreeGrammar):
    """Tests the new error reporting with different verbosity levels."""
    
    # Batch of expressions with valid and invalid items
    expressions_str = [
        "X_0 + 5",         # Valid
        "( X_0 + 5",       # Invalid: Mismatched parenthesis
        "sin ( X_0 )",     # Valid
        "*",               # Invalid: Malformed (binary op without operands)
        "cos ( X_0"        # Invalid: Mismatched parenthesis
    ]
    
    max_len = max(len(s.split()) for s in expressions_str)
    
    # Create padded tensor of IDs
    expressions_list_padded = [s.split() + ['NO_OP'] * (max_len - len(s.split())) for s in expressions_str]
    expressions_tensor = torch.stack([to_ids(pcfg, tokens) for tokens in expressions_list_padded])

    # --- Test verbosity=0 (default, summary only) ---
    with pytest.raises(RuntimeError) as excinfo:
        pcfg.parse_to_postfix(expressions_tensor, verbosity=0)
    
    err_str_v0 = str(excinfo.value)
    assert "Error Summary:" in err_str_v0
    assert "Parsing failed: Mismatched or unbalanced parentheses.: 2 occurrences." in err_str_v0
    assert "Binary operator is missing one or both operands during tree construction.: 1 occurrences." in err_str_v0
    assert "Displaying" not in err_str_v0
    assert "Problem at Index" not in err_str_v0

    # --- Test verbosity=1 (summary + up to 5 examples) ---
    with pytest.raises(RuntimeError) as excinfo:
        pcfg.parse_to_postfix(expressions_tensor, verbosity=1)
        
    err_str_v1 = str(excinfo.value)
    assert "Error Summary:" in err_str_v1
    assert "Parsing failed: Mismatched or unbalanced parentheses.: 2 occurrences" in err_str_v1
    assert "Displaying 3 out of 3 problematic expressions" in err_str_v1
    assert "Problem at Index: 1" in err_str_v1
    assert "Input Expression: '( X_0 + 5'" in err_str_v1
    assert "Problem at Index: 3" in err_str_v1
    assert "Input Expression: '*'" in err_str_v1
    assert "Problem at Index: 4" in err_str_v1
    assert "Input Expression: 'cos ( X_0'" in err_str_v1
    
    # --- Test verbosity=2 (summary + all examples) ---
    with pytest.raises(RuntimeError) as excinfo:
        pcfg.parse_to_postfix(expressions_tensor, verbosity=2)

    err_str_v2 = str(excinfo.value)
    # Since we have < 5 errors, output should be identical to verbosity=1
    assert err_str_v1 == err_str_v2
import torch
import pytest
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
    return ProbababilisticContextFreeGrammar(test_grammar, "E", 20, 1, torch.device(device))


def test_parse_to_postfix_constant(pcfg: ProbababilisticContextFreeGrammar):
    expression = torch.tensor([[CONST_1]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_postfix(expression)

    assert torch.equal(ops[:, :1], expression)
    assert torch.equal(children[:, 0], torch.tensor([[NULL_CHILD, NULL_CHILD]], dtype=torch.int64, device=pcfg.device))


@pytest.mark.parametrize("op_id", [SIN, COS, EXP, LOG, SQRT, SQUARE])
def test_parse_to_postfix_unary(op_id: int, pcfg: ProbababilisticContextFreeGrammar):
    expression = torch.tensor([[op_id, LPAREN, CONST_1, RPAREN]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_postfix(expression)

    expected_ops = torch.tensor([[CONST_1, op_id]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[[NULL_CHILD, NULL_CHILD], [0, NULL_CHILD]]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :2], expected_ops)
    assert torch.equal(children[:, :2], expected_children)


@pytest.mark.parametrize("op_id", [ADD, SUB, MUL, DIV])
def test_parse_to_postfix_binary(op_id: int, pcfg: ProbababilisticContextFreeGrammar):
    expression = torch.tensor([[CONST_1, op_id, CONST_1]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_postfix(expression)

    expected_ops = torch.tensor([[CONST_1, CONST_1, op_id]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[[NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [0, 1]]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :3], expected_ops)
    assert torch.equal(children[:, :3], expected_children)


def test_parse_to_postfix_parenthesis(pcfg: ProbababilisticContextFreeGrammar):
    # (1 + 1) * 1
    expression = torch.tensor([[LPAREN, CONST_1, ADD, CONST_1, RPAREN, MUL, CONST_1]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_postfix(expression)

    expected_ops = torch.tensor([[CONST_1, CONST_1, ADD, CONST_1, MUL]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[[NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [0, 1], [NULL_CHILD, NULL_CHILD], [2, 3]]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :5], expected_ops)
    assert torch.equal(children[:, :5], expected_children)

def test_parse_to_postfix_all_operators(pcfg: ProbababilisticContextFreeGrammar):
    # (sin(X_0) + 1) * (log(X_0) - 1)
    X_0 = VAR_START_ID
    expression = torch.tensor([[
        LPAREN, SIN, LPAREN, X_0, RPAREN, ADD, CONST_1, RPAREN,
        MUL,
        LPAREN, LOG, LPAREN, X_0, RPAREN, SUB, CONST_1, RPAREN
    ]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_postfix(expression)

    expected_ops = torch.tensor([[X_0, SIN, CONST_1, ADD, X_0, LOG, CONST_1, SUB, MUL]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[
        [NULL_CHILD, NULL_CHILD], [0, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [1, 2],
        [NULL_CHILD, NULL_CHILD], [4, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [5, 6],
        [3, 7]
    ]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :9], expected_ops)
    assert torch.equal(children[:, :9], expected_children)


def test_parse_to_postfix_precedence(pcfg: ProbababilisticContextFreeGrammar):
    # 1 + 1 * 1
    expression = torch.tensor([[CONST_1, ADD, CONST_1, MUL, CONST_1]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_postfix(expression)

    # Postfix: 1, 1, 1, *, +
    expected_ops = torch.tensor([[CONST_1, CONST_1, CONST_1, MUL, ADD]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[
        [NULL_CHILD, NULL_CHILD], # 1
        [NULL_CHILD, NULL_CHILD], # 1
        [NULL_CHILD, NULL_CHILD], # 1
        [1, 2],                   # *
        [0, 3]                    # +
    ]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :5], expected_ops)
    assert torch.equal(children[:, :5], expected_children)


def test_parse_to_postfix_associativity(pcfg: ProbababilisticContextFreeGrammar):
    # Left associativity: 1 + 1 + 1 + 1 -> ((1+1)+1)+1
    expression_add = torch.tensor([[CONST_1, ADD, CONST_1, ADD, CONST_1, ADD, CONST_1]], dtype=torch.int64, device=pcfg.device)
    ops_add, children_add = pcfg.parse_to_postfix(expression_add)
    expected_ops_add = torch.tensor([[CONST_1, CONST_1, ADD, CONST_1, ADD, CONST_1, ADD]], dtype=torch.int64, device=pcfg.device)
    expected_children_add = torch.tensor([[
        [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [0, 1],
        [NULL_CHILD, NULL_CHILD], [2, 3],
        [NULL_CHILD, NULL_CHILD], [4, 5]
    ]], dtype=torch.int64, device=pcfg.device)
    assert torch.equal(ops_add[:, :7], expected_ops_add)
    assert torch.equal(children_add[:, :7], expected_children_add)

    # Right associativity: sin(sin(sin(sin(1))))
    expression_sin = torch.tensor([[SIN, LPAREN, SIN, LPAREN, SIN, LPAREN, SIN, LPAREN, CONST_1, RPAREN, RPAREN, RPAREN, RPAREN]], dtype=torch.int64, device=pcfg.device)
    ops_sin, children_sin = pcfg.parse_to_postfix(expression_sin)
    expected_ops_sin = torch.tensor([[CONST_1, SIN, SIN, SIN, SIN]], dtype=torch.int64, device=pcfg.device)
    expected_children_sin = torch.tensor([[
        [NULL_CHILD, NULL_CHILD], [0, NULL_CHILD], [1, NULL_CHILD], [2, NULL_CHILD], [3, NULL_CHILD]
    ]], dtype=torch.int64, device=pcfg.device)
    assert torch.equal(ops_sin[:, :5], expected_ops_sin)
    assert torch.equal(children_sin[:, :5], expected_children_sin)


def test_parse_to_postfix_mixed_operators(pcfg: ProbababilisticContextFreeGrammar):
    # sin(1) + 1
    expression = torch.tensor([[SIN, LPAREN, CONST_1, RPAREN, ADD, CONST_1]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_postfix(expression)

    # Postfix: 1, sin, 1, +
    expected_ops = torch.tensor([[CONST_1, SIN, CONST_1, ADD]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[
        [NULL_CHILD, NULL_CHILD], # 1
        [0, NULL_CHILD],          # sin
        [NULL_CHILD, NULL_CHILD], # 1
        [1, 2]                    # +
    ]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :4], expected_ops)
    assert torch.equal(children[:, :4], expected_children)


def test_parse_to_postfix_complex_parenthesis(pcfg: ProbababilisticContextFreeGrammar):
    # (1 + 1) + (1 + (1 + 1))
    expression = torch.tensor([[
        LPAREN, CONST_1, ADD, CONST_1, RPAREN,
        ADD,
        LPAREN, CONST_1, ADD, LPAREN, CONST_1, ADD, CONST_1, RPAREN, RPAREN
    ]], dtype=torch.int64, device=pcfg.device)
    ops, children = pcfg.parse_to_postfix(expression)

    # Postfix: 1 1 + 1 1 1 + + +
    expected_ops = torch.tensor([[CONST_1, CONST_1, ADD, CONST_1, CONST_1, CONST_1, ADD, ADD, ADD]], dtype=torch.int64, device=pcfg.device)
    expected_children = torch.tensor([[
        [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [0, 1],
        [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [NULL_CHILD, NULL_CHILD], [4, 5],
        [3, 6],
        [2, 7]
    ]], dtype=torch.int64, device=pcfg.device)

    assert torch.equal(ops[:, :9], expected_ops)
    assert torch.equal(children[:, :9], expected_children)


def test_parse_to_postfix_batch(pcfg: ProbababilisticContextFreeGrammar):
    expressions_list = [
        torch.tensor([[CONST_1]], dtype=torch.int64),
        torch.tensor([[SIN, LPAREN, CONST_1, RPAREN]], dtype=torch.int64),
        torch.tensor([[CONST_1, ADD, CONST_1]], dtype=torch.int64),
        torch.tensor([[LPAREN, CONST_1, ADD, CONST_1, RPAREN, MUL, CONST_1]], dtype=torch.int64),
    ]
    max_len = max(e.shape[1] for e in expressions_list)
    expressions = torch.full((len(expressions_list), max_len), NO_OP, dtype=torch.int64, device=pcfg.device)
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
        assert torch.all(batch_ops[i, len_single_ops:] == NO_OP)


def test_parse_to_postfix_max_length(pcfg: ProbababilisticContextFreeGrammar):
    # Create an expression of max length (128)
    # e.g. 1 + 1 + ... + 1 (64 ones, 63 pluses) -> 127 length
    num_ones = 64
    expr_list = [CONST_1] * num_ones
    for i in range(num_ones - 1):
        expr_list.insert(2 * i + 1, ADD)

    expression = torch.tensor([expr_list], dtype=torch.int64, device=pcfg.device)
    assert expression.shape[1] == 2 * num_ones - 1

    ops, children = pcfg.parse_to_postfix(expression)

    # Check that it doesn't error out and the result is plausible
    # The postfix should be 1, 1, +, 1, +, ..., 1, +
    # Total length is also 127
    assert ops.shape[1] == 2 * num_ones -1
    assert children.shape[1] == 2 * num_ones -1
    assert ops[0, -1] == ADD # last op is ADD


def test_benchmark_parsing_cpu_vs_cuda():
    import time

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Batch of complex expressions
    expressions_list = [
        torch.tensor([[LPAREN, CONST_1, ADD, CONST_1, RPAREN, MUL, CONST_1]]),
        torch.tensor([[SIN, LPAREN, COS, LPAREN, CONST_1, RPAREN, RPAREN, ADD, LOG, LPAREN, CONST_1, RPAREN]]),
        torch.tensor([[LPAREN, LPAREN, CONST_1, MUL, CONST_1, RPAREN, ADD, CONST_1, RPAREN, DIV, CONST_1]]),
    ] * 1000000 # Make a large batch

    max_len = max(e.shape[1] for e in expressions_list)
    expressions = torch.full((len(expressions_list), max_len), NO_OP, dtype=torch.int64)
    for i, e in enumerate(expressions_list):
        expressions[i, :e.shape[1]] = e

    pcfg_cpu = ProbababilisticContextFreeGrammar(test_grammar, "E", 20, 1, torch.device("cpu"))
    pcfg_cuda = ProbababilisticContextFreeGrammar(test_grammar, "E", 20, 1, torch.device("cuda"))

    expressions_cpu = expressions.to("cpu")
    expressions_cuda = expressions.to("cuda")

    # Warmup
    pcfg_cpu.parse_to_postfix(expressions_cpu)
    pcfg_cuda.parse_to_postfix(expressions_cuda)
    torch.cuda.synchronize()

    # CPU benchmark
    start_time = time.time()
    for _ in range(10):
        pcfg_cpu.parse_to_postfix(expressions_cpu)
    cpu_time = time.time() - start_time
    print(f"\nCPU parsing time: {cpu_time:.4f}s")

    # CUDA benchmark
    start_time = time.time()
    for _ in range(10):
        pcfg_cuda.parse_to_postfix(expressions_cuda)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time
    print(f"CUDA parsing time: {cuda_time:.4f}s")

    # NOTE: for lower values CPU is a bit faster, but for larger batches CUDA becomes significantly faster.

import torch
import pytest
from typing import List

from symbolic_torch import ProbabilisticContextFreeGrammar, Operator

# A standard grammar that includes all operator types for thorough testing
test_grammar: str = """
E -> E + F [0.2]
E -> E - F [0.2]
E -> F [0.6]
F -> F * T [0.2]
F -> F / T [0.2]
F -> T [0.6]
T -> ( E ) P [0.1]
T -> R ( E ) [0.3]
T -> V [0.4]
T -> ( E ) [0.2]
V -> X_0 [0.3]
V -> X_1 [0.3]
V -> C   [0.2]
V -> 5   [0.2]
R -> sin [0.2]
R -> cos [0.2]
R -> exp [0.2]
R -> sqrt [0.2]
R -> log [0.2]
P -> ^2 [1.0]
"""

# Fixture to provide a configured PCFG object for CPU and CUDA
@pytest.fixture(scope="module")
def pcfg_cpu() -> ProbabilisticContextFreeGrammar:
    """Provides a PCFG instance configured for CPU."""
    return ProbabilisticContextFreeGrammar(
        grammar=test_grammar,
        start_symbol="E",
        padded_maximum_length=64,
        n_variables=2,
        device=torch.device("cpu"),
    )

@pytest.fixture(scope="module")
def pcfg_cuda() -> ProbabilisticContextFreeGrammar:
    """Provides a PCFG instance configured for CUDA, skipping if not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return ProbabilisticContextFreeGrammar(
        grammar=test_grammar,
        start_symbol="E",
        padded_maximum_length=64,
        n_variables=2,
        device=torch.device("cuda"),
    )

@pytest.fixture(params=["cpu", "cuda"])
def pcfg(request: pytest.FixtureRequest, pcfg_cpu: ProbabilisticContextFreeGrammar, pcfg_cuda: ProbabilisticContextFreeGrammar) -> ProbabilisticContextFreeGrammar:
    """Parametrized fixture to provide either the CPU or CUDA PCFG instance."""
    if request.param == "cpu":
        return pcfg_cpu
    return pcfg_cuda

# Helper to convert a list of strings to a tensor of IDs
def to_ids(pcfg_obj: ProbabilisticContextFreeGrammar, tokens: List[str]) -> torch.Tensor:
    """Converts a list of string tokens to a tensor of their corresponding integer IDs."""
    return torch.tensor([pcfg_obj.get_symbol_id(s) for s in tokens], dtype=torch.int64, device=pcfg_obj.device)

# Helper to trim padding from the output tensor for comparison
def trim_padding(tensor: torch.Tensor) -> torch.Tensor:
    """Removes NO_OP padding from the end of a 1D tensor."""
    is_padding: torch.Tensor = (tensor == int(Operator.NO_OP))
    if not torch.any(is_padding):
        return tensor # No padding found
    
    # Find the index of the first padding token
    first_padding_idx: torch.Tensor = torch.where(is_padding, 1, 0).argmax()
    return tensor[:first_padding_idx]

def run_conversion_test(pcfg_obj: ProbabilisticContextFreeGrammar, conversion_type: str, input_str_list: List[str], expected_str_list: List[str]) -> None:
    """General-purpose test runner for conversions."""
    # Convert string lists to tensors
    input_ids: torch.Tensor = to_ids(pcfg_obj, input_str_list).unsqueeze(0)
    expected_ids: torch.Tensor = to_ids(pcfg_obj, expected_str_list)

    if conversion_type == "postfix_to_infix":
        postfix_ops, _ = pcfg_obj.parse_to_postfix(input_ids)
        result: torch.Tensor = pcfg_obj.postfix_to_infix(postfix_ops, 100)
    elif conversion_type == "prefix_to_infix":
        prefix_ops, _ = pcfg_obj.parse_to_prefix(input_ids)
        result = pcfg_obj.prefix_to_infix(prefix_ops, 100)
    else:
        raise ValueError(f"Unknown conversion type: {conversion_type}")

    result_string = pcfg_obj.to_string(result)

    trimmed_result: torch.Tensor = trim_padding(result[0])
    assert torch.equal(trimmed_result, expected_ids), f"Failed on {" ".join(input_str_list)}, expected {" ".join(expected_str_list)} but got {result_string[0]}"

@pytest.mark.parametrize("conversion_type", ["postfix_to_infix", "prefix_to_infix"])
class TestInfixConversions:

    # --- Group 1: Constants and Variables ---
    @pytest.mark.parametrize("token", ["X_0", "X_1", "C", "5"])
    def test_operands(self, pcfg: ProbabilisticContextFreeGrammar, conversion_type: str, token: str) -> None:
        run_conversion_test(pcfg, conversion_type, [token], [token])

    # --- Group 2: Functional Unary Operators ---
    @pytest.mark.parametrize("op", ["sin", "cos", "exp", "log", "sqrt"])
    def test_functional_unary_ops(self, pcfg: ProbabilisticContextFreeGrammar, conversion_type: str, op: str) -> None:
        input_str = [op, '(', 'X_0', ')']
        expected_str = [op, '(', 'X_0', ')']
        run_conversion_test(pcfg, conversion_type, input_str, expected_str)

    # --- Group 3: Postfix Unary Operators ---
    def test_postfix_unary_op(self, pcfg: ProbabilisticContextFreeGrammar, conversion_type: str) -> None:
        input_str = ['(', 'X_0', ')', '^2']
        expected_str = ['(', 'X_0', ')', '^2']
        run_conversion_test(pcfg, conversion_type, input_str, expected_str)
        
    # --- Group 4: Binary Operators ---
    @pytest.mark.parametrize("op", ["+", "-", "*", "/", "^"])
    def test_binary_ops(self, pcfg: ProbabilisticContextFreeGrammar, conversion_type: str, op: str) -> None:
        input_str = ['(', 'X_0', op, 'X_1', ')']
        expected_str = ['(', 'X_0', op, 'X_1', ')']
        run_conversion_test(pcfg, conversion_type, input_str, expected_str)

    # --- Group 5: Longer, Nested Test ---
    def test_complex_expression(self, pcfg: ProbabilisticContextFreeGrammar, conversion_type: str) -> None:
        input_str = ['(', 'sin', '(', 'X_0', ')', '+', '(', '5', '*', 'X_1', ')', ')']
        expected_str = ['(', 'sin', '(', 'X_0', ')', '+', '(', '5', '*', 'X_1', ')', ')']
        run_conversion_test(pcfg, conversion_type, input_str, expected_str)

    def test_complex_expression2(self, pcfg: ProbabilisticContextFreeGrammar, conversion_type: str) -> None:
        input_str = ['(', 'X_0', '+', '(', 'X_1', '*', '5', ')', '^2', ')', '+', '(', 'cos', '(', 'C', ')', ')']
        expected_str = ['(', '(', 'X_0', '+', '(', '(', 'X_1', '*', '5', ')', ')', '^2', ')', '+', 'cos', '(', 'C', ')', ')']
        run_conversion_test(pcfg, conversion_type, input_str, expected_str)

    # --- Group 6: Batch Processing Test ---
    def test_batch_processing(self, pcfg: ProbabilisticContextFreeGrammar, conversion_type: str) -> None:
        expr1_in = ['X_0', '+', 'X_1']
        expr1_exp = ['(', 'X_0', '+', 'X_1', ')']
        
        expr2_in = ['cos', '(', '5', ')']
        expr2_exp = ['cos', '(', '5', ')']
        
        expr3_in = ['X_0', '^2']
        expr3_exp = ['(', 'X_0', ')', '^2']

        max_len = max(len(expr1_in), len(expr2_in), len(expr3_in))
        
        def pad(tokens: List[str], length: int) -> List[str]:
            return tokens + ['NO_OP'] * (length - len(tokens))

        ids1 = to_ids(pcfg, pad(expr1_in, max_len))
        ids2 = to_ids(pcfg, pad(expr2_in, max_len))
        ids3 = to_ids(pcfg, pad(expr3_in, max_len))
        
        input_batch = torch.stack([ids1, ids2, ids3], dim=0)

        expected_ids1 = to_ids(pcfg, expr1_exp)
        expected_ids2 = to_ids(pcfg, expr2_exp)
        expected_ids3 = to_ids(pcfg, expr3_exp)

        if conversion_type == "postfix_to_infix":
            postfix_ops, _ = pcfg.parse_to_postfix(input_batch)
            result_batch = pcfg.postfix_to_infix(postfix_ops, 100)
        else:
            prefix_ops, _ = pcfg.parse_to_prefix(input_batch)
            result_batch = pcfg.prefix_to_infix(prefix_ops, 100)

        assert torch.equal(trim_padding(result_batch[0]), expected_ids1)
        assert torch.equal(trim_padding(result_batch[1]), expected_ids2)
        assert torch.equal(trim_padding(result_batch[2]), expected_ids3)


@pytest.mark.parametrize("pcfg", ["pcfg_cpu", "pcfg_cuda"], indirect=True)
class TestConversionErrors:
    
    def test_postfix_to_infix_output_too_short(self, pcfg: ProbabilisticContextFreeGrammar) -> None:
        string_expr = to_ids(pcfg, ['X_0', '+', '5']).unsqueeze(0)
        postfix_ops, _ = pcfg.parse_to_postfix(string_expr)
        
        with pytest.raises(RuntimeError, match="postfix to infix expression too long"):
            pcfg.postfix_to_infix(postfix_ops, 4)

    def test_prefix_to_infix_output_too_short(self, pcfg: ProbabilisticContextFreeGrammar) -> None:
        string_expr = to_ids(pcfg, ['X_0', '+', '5']).unsqueeze(0)
        prefix_ops, _ = pcfg.parse_to_prefix(string_expr)
        
        with pytest.raises(RuntimeError, match="prefix to infix expression too long"):
            pcfg.prefix_to_infix(prefix_ops, 4)

    def test_postfix_malformed_input(self, pcfg: ProbabilisticContextFreeGrammar) -> None:
        malformed_postfix = to_ids(pcfg, ['X_0', '+']).unsqueeze(0)
        
        with pytest.raises(RuntimeError, match="Binary operator without enough operands"):
            pcfg.postfix_to_infix(malformed_postfix, 10)

    def test_prefix_malformed_input(self, pcfg: ProbabilisticContextFreeGrammar) -> None:
        malformed_prefix = to_ids(pcfg, ['+', 'X_0', 'NO_OP']).unsqueeze(0)
        
        with pytest.raises(RuntimeError, match="Binary operator without enough operands"):
            pcfg.prefix_to_infix(malformed_prefix, 10)
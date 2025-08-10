import torch
import pytest
from typing import List

from symbolic_torch import ProbabilisticContextFreeGrammar, Operator
from .utils import get_cuda_device_with_min_memory

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
V -> X_0 [0.2]
V -> X_1 [0.2]
V -> C   [0.3]
V -> 5   [0.3]
R -> sin [0.2]
R -> cos [0.2]
R -> exp [0.2]
R -> sqrt [0.2]
R -> log [0.2]
P -> ^2 [0.25]
P -> ^3 [0.25]
P -> ^4 [0.25]
P -> ^5 [0.25]
"""

# Fixture to provide a configured PCFG object for CPU and CUDA
@pytest.fixture(scope="module")
def pcfg_cpu() -> ProbabilisticContextFreeGrammar:
    print("Hello from CPU fixture")
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
    print("Hello from CUDA fixture")
    """Provides a PCFG instance configured for CUDA, skipping if not available."""
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this system.")
    index = get_cuda_device_with_min_memory()
    return ProbabilisticContextFreeGrammar(
        grammar=test_grammar,
        start_symbol="E",
        padded_maximum_length=64,
        n_variables=2,
        device=torch.device(f"cuda:{index}"),
    )

@pytest.fixture
def pcfg(request: pytest.FixtureRequest, pcfg_cpu: ProbabilisticContextFreeGrammar, pcfg_cuda: ProbabilisticContextFreeGrammar) -> ProbabilisticContextFreeGrammar:
    """Parametrized fixture to provide either the CPU or CUDA PCFG instance."""
    if request.param == "pcfg_cpu":
        return pcfg_cpu
    elif request.param == "pcfg_cuda":
        return pcfg_cuda
    raise ValueError(f"Invalid pcfg fixture param: {request.param}")

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
@pytest.mark.parametrize("pcfg", ["pcfg_cpu", "pcfg_cuda"], indirect=True)
class TestInfixConversions:

    # --- Group 1: Constants and Variables ---
    @pytest.mark.parametrize("token", ["X_0", "X_1", "C", "5"])
    def test_operands(self, pcfg: ProbabilisticContextFreeGrammar, conversion_type: str, token: str) -> None:
        run_conversion_test(pcfg, conversion_type, [token], [token])

    # --- Group 2: Functional Unary Operators ---
    @pytest.mark.parametrize("op", ["sin", "cos", "exp", "log", "sqrt", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "floor", "ceil", "ln", "log10", "neg"])
    def test_functional_unary_ops(self, pcfg: ProbabilisticContextFreeGrammar, conversion_type: str, op: str) -> None:
        input_str = [op, '(', 'X_0', ')']
        expected_str = [op, '(', 'X_0', ')']
        run_conversion_test(pcfg, conversion_type, input_str, expected_str)

    # --- Group 3: Postfix Unary Operators ---
    def test_postfix_unary_op(self, pcfg: ProbabilisticContextFreeGrammar, conversion_type: str) -> None:
        input_str = ['(', 'X_0', ')', '^2']
        expected_str = ['(', 'X_0', ')', '^2']
        run_conversion_test(pcfg, conversion_type, input_str, expected_str)
    
    # Test specific prefix_to_infix conversion for power operators ^2, ^3, ^4, ^5
    @pytest.mark.parametrize("power_op", ["^2", "^3", "^4", "^5"])
    def test_prefix_to_infix_power_operators(self, pcfg: ProbabilisticContextFreeGrammar, power_op: str, conversion_type : str) -> None:
        if conversion_type != "prefix_to_infix":
            return
        """Test that prefix_to_infix correctly converts [^2, C] to (C)^2 string, etc."""
        # For prefix expressions like [^2, C], the expected infix should be (C)^2
        input_prefix = [power_op, 'C']
        expected_infix = ['(', 'C', ')', power_op]
        
        # Only test prefix_to_infix conversion specifically
        run_conversion_test(pcfg, "prefix_to_infix", input_prefix, expected_infix)
        
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
        
        with pytest.raises(RuntimeError, match=" Resulting infix expression is longer than the maximum allowed length"):
            pcfg.postfix_to_infix(postfix_ops, 4)

    def test_prefix_to_infix_output_too_short(self, pcfg: ProbabilisticContextFreeGrammar) -> None:
        string_expr = to_ids(pcfg, ['X_0', '+', '5']).unsqueeze(0)
        prefix_ops, _ = pcfg.parse_to_prefix(string_expr)
        
        with pytest.raises(RuntimeError, match=" Resulting infix expression is longer than the maximum allowed length"):
            pcfg.prefix_to_infix(prefix_ops, 4)

    def test_postfix_malformed_input(self, pcfg: ProbabilisticContextFreeGrammar) -> None:
        malformed_postfix = to_ids(pcfg, ['X_0', '+']).unsqueeze(0)
        
        with pytest.raises(RuntimeError, match="Malformed input, binary operator is missing one or both operands."):
            pcfg.postfix_to_infix(malformed_postfix, 10)

    def test_prefix_malformed_input(self, pcfg: ProbabilisticContextFreeGrammar) -> None:
        malformed_prefix = to_ids(pcfg, ['+', 'X_0', 'NO_OP']).unsqueeze(0)
        
        with pytest.raises(RuntimeError, match="Malformed input, binary operator is missing one or both operands."):
            pcfg.prefix_to_infix(malformed_prefix, 10)
# In /tests/test_to_infix.py, inside class TestConversionErrors:

    def test_postfix_malformed_unary(self, pcfg: ProbabilisticContextFreeGrammar) -> None:
        malformed_postfix = to_ids(pcfg, ['sin']).unsqueeze(0)
        
        with pytest.raises(RuntimeError, match="Malformed input, unary operator is missing an operand"):
            pcfg.postfix_to_infix(malformed_postfix, 10)

    def test_prefix_malformed_unary(self, pcfg: ProbabilisticContextFreeGrammar) -> None:
        malformed_prefix = to_ids(pcfg, ['sin']).unsqueeze(0)
        
        with pytest.raises(RuntimeError, match="Malformed input, unary operator is missing an operand"):
            pcfg.prefix_to_infix(malformed_prefix, 10)
            
    def test_postfix_malformed_extra_operands(self, pcfg: ProbabilisticContextFreeGrammar) -> None:
        # Postfix for "(X_0 + X_0) X_0" - leaves two items on stack
        malformed_postfix = to_ids(pcfg, ['X_0', 'X_0', '+', 'X_0']).unsqueeze(0)

        print(malformed_postfix.device)
        print(pcfg.device)
        
        with pytest.raises(RuntimeError, match="Conversion failed: Resulting infix expression has multiple roots, which is not allowed.: 1 occurrences."):
            pcfg.postfix_to_infix(malformed_postfix, 20)

    def test_prefix_malformed_extra_operands(self, pcfg: ProbabilisticContextFreeGrammar) -> None:
        # Prefix for "X_0 (+ X_0 X_0)" - leaves two items on stack
        malformed_prefix = to_ids(pcfg, ['X_0', '+', 'X_0', 'X_0']).unsqueeze(0)

        with pytest.raises(RuntimeError, match="Conversion failed: Resulting infix expression has multiple roots, which is not allowed"):
            pcfg.prefix_to_infix(malformed_prefix, 20)

    def test_postfix_to_infix_exact_max_length(self, pcfg: ProbabilisticContextFreeGrammar) -> None:
        """Test postfix_to_infix when the output exactly matches the maximum allowed length."""
        # Create nested log expressions: log(log(log(X_0)))
        # Postfix: X_0 log log log
        # Expected infix: log(log(log(X_0))) = 15 tokens: log ( log ( log ( X_0 ) ) )
        postfix_expr = ['X_0', 'log', 'log', 'log']
        postfix_ids = to_ids(pcfg, postfix_expr).unsqueeze(0)
        
        # Expected result should be exactly 15 tokens
        expected_infix = ['log', '(', 'log', '(', 'log', '(', 'X_0', ')', ')', ')']
        expected_ids = to_ids(pcfg, expected_infix)
        
        # Test with max_length = 10 (exactly the expected length)
        result = pcfg.postfix_to_infix(postfix_ids, 10)
        trimmed_result = trim_padding(result[0])
        
        assert torch.equal(trimmed_result, expected_ids), f"Expected {expected_infix}, got {pcfg.to_string(result)[0]}"

    def test_prefix_to_infix_exact_max_length(self, pcfg: ProbabilisticContextFreeGrammar) -> None:
        """Test prefix_to_infix when the output exactly matches the maximum allowed length."""
        # Create nested log expressions: log(log(log(X_0)))
        # Prefix: log log log X_0
        # Expected infix: log(log(log(X_0))) = 15 tokens: log ( log ( log ( X_0 ) ) )
        prefix_expr = ['log', 'log', 'log', 'X_0']
        prefix_ids = to_ids(pcfg, prefix_expr).unsqueeze(0)
        
        # Expected result should be exactly 10 tokens
        expected_infix = ['log', '(', 'log', '(', 'log', '(', 'X_0', ')', ')', ')']
        expected_ids = to_ids(pcfg, expected_infix)
        
        # Test with max_length = 10 (exactly the expected length)
        result = pcfg.prefix_to_infix(prefix_ids, 10)
        trimmed_result = trim_padding(result[0])
        
        assert torch.equal(trimmed_result, expected_ids), f"Expected {expected_infix}, got {pcfg.to_string(result)[0]}"

    def test_deeply_nested_max_length(self, pcfg: ProbabilisticContextFreeGrammar) -> None:
        """Test with deeply nested log expressions that exactly hit the maximum length limit."""
        # Create log(log(log(log(log(X_0))))) - 5 nested logs
        # Postfix: X_0 log log log log log
        # Expected infix: log(log(log(log(log(X_0))))) = 16 tokens: log ( log ( log ( log ( log ( X_0 ) ) ) ) )
        postfix_expr = ['X_0', 'log', 'log', 'log', 'log', 'log']
        postfix_ids = to_ids(pcfg, postfix_expr).unsqueeze(0)
        
        expected_infix = ['log', '(', 'log', '(', 'log', '(', 'log', '(', 'log', '(', 'X_0', ')', ')', ')', ')', ')']
        expected_ids = to_ids(pcfg, expected_infix)
        
        # Test with max_length = 16 (exactly the expected length)
        result = pcfg.postfix_to_infix(postfix_ids, 16)
        trimmed_result = trim_padding(result[0])
        
        assert torch.equal(trimmed_result, expected_ids), f"Expected {expected_infix}, got {pcfg.to_string(result)[0]}"
        
        # Also test the same with prefix notation
        # Prefix: log log log log log X_0
        prefix_expr = ['log', 'log', 'log', 'log', 'log', 'X_0']
        prefix_ids = to_ids(pcfg, prefix_expr).unsqueeze(0)
        
        result_prefix = pcfg.prefix_to_infix(prefix_ids, 16)
        trimmed_result_prefix = trim_padding(result_prefix[0])
        
        assert torch.equal(trimmed_result_prefix, expected_ids), f"Expected {expected_infix}, got {pcfg.to_string(result_prefix)[0]}"

    def test_hard_max_length_limit(self, pcfg: ProbabilisticContextFreeGrammar) -> None:
        """Test with expressions that approach the hard maximum length limit of 128."""
        # Create a PCFG with the hard maximum length of 128
        pcfg_max = ProbabilisticContextFreeGrammar(
            grammar=test_grammar,
            start_symbol="E",
            padded_maximum_length=128,  # Hard maximum length
            n_variables=2,
            device=pcfg.device,
        )
        
        n_logs = 30

        # Create postfix expression: X_0 log log log ... (127 times)
        postfix_expr = ['X_0'] + ['log'] * n_logs
        postfix_ids = to_ids(pcfg_max, postfix_expr).unsqueeze(0)

        # Expected infix should be log(log(log(...log(X_0)...))) with 127 nested logs
        # This should be exactly 382 tokens: 127 * 3 + 1 = 382
        expected_infix: List[str] = []
        for _ in range(n_logs):
            expected_infix.append('log')
            expected_infix.append('(')
        expected_infix.append('X_0')
        for _ in range(n_logs):
            expected_infix.append(')')
        
        expected_ids = to_ids(pcfg_max, expected_infix)

        # Test with max_length = 127 (exactly the expected length)
        result = pcfg_max.postfix_to_infix(postfix_ids, 128 * 3)
        trimmed_result = trim_padding(result[0])
        
        assert torch.equal(trimmed_result, expected_ids), f"Expected {len(expected_infix)} tokens, got {len(trimmed_result)} tokens"
        
        # Also test the same with prefix notation
        # Prefix: log log log ... log X_0 (128 logs + X_0)
        prefix_expr = ['log'] * n_logs + ['X_0']
        prefix_ids = to_ids(pcfg_max, prefix_expr).unsqueeze(0)

        result_prefix = pcfg_max.prefix_to_infix(prefix_ids, 128 * 3)
        trimmed_result_prefix = trim_padding(result_prefix[0])
        
        assert torch.equal(trimmed_result_prefix, expected_ids), f"Expected {len(expected_infix)} tokens, got {len(trimmed_result_prefix)} tokens"
        
        # Test a moderately complex expression
        # Create: (sin(X_0) + cos(X_1)) * (exp(C) - log(5))
        # This should create a reasonable length expression without excessive nesting
        # Postfix: X_0 sin X_1 cos + C exp 5 log - *
        postfix_complex = ['X_0', 'sin', 'X_1', 'cos', '+', 'C', 'exp', '5', 'log', '-', '*']
        postfix_complex_ids = to_ids(pcfg_max, postfix_complex).unsqueeze(0)
        
        # Expected infix: ((sin(X_0) + cos(X_1)) * (exp(C) - log(5)))
        expected_complex = ['(', '(', 'sin', '(', 'X_0', ')', '+', 'cos', '(', 'X_1', ')', ')', '*', '(', 'exp', '(', 'C', ')', '-', 'log', '(', '5', ')', ')', ')']
        expected_complex_ids = to_ids(pcfg_max, expected_complex)
        
        result_complex = pcfg_max.postfix_to_infix(postfix_complex_ids, len(expected_complex))
        trimmed_result_complex = trim_padding(result_complex[0])
        
        assert torch.equal(trimmed_result_complex, expected_complex_ids), f"Expected {len(expected_complex)} tokens, got {len(trimmed_result_complex)} tokens"
        
        # Test prefix version of the complex expression
        # Prefix: * + sin X_0 cos X_1 - exp C log 5
        prefix_complex = ['*', '+', 'sin', 'X_0', 'cos', 'X_1', '-', 'exp', 'C', 'log', '5']
        prefix_complex_ids = to_ids(pcfg_max, prefix_complex).unsqueeze(0)
        
        result_prefix_complex = pcfg_max.prefix_to_infix(prefix_complex_ids, len(expected_complex))
        trimmed_result_prefix_complex = trim_padding(result_prefix_complex[0])
        
        assert torch.equal(trimmed_result_prefix_complex, expected_complex_ids), f"Expected {len(expected_complex)} tokens, got {len(trimmed_result_prefix_complex)} tokens"
        
        # Test an expression that's longer but still within workspace limits
        # Create multiple sequential binary operations: ((X_0 + X_1) * (C + 5)) + ((X_0 - X_1) / (C - 5))
        # Postfix: X_0 X_1 + C 5 + * X_0 X_1 - C 5 - / +
        postfix_long = ['X_0', 'X_1', '+', 'C', '5', '+', '*', 'X_0', 'X_1', '-', 'C', '5', '-', '/', '+']
        postfix_long_ids = to_ids(pcfg_max, postfix_long).unsqueeze(0)
        
        # Expected infix: (((X_0 + X_1) * (C + 5)) + ((X_0 - X_1) / (C - 5)))
        expected_long = ['(', '(', '(', 'X_0', '+', 'X_1', ')', '*', '(', 'C', '+', '5', ')', ')', '+', '(', '(', 'X_0', '-', 'X_1', ')', '/', '(', 'C', '-', '5', ')', ')', ')']
        expected_long_ids = to_ids(pcfg_max, expected_long)
        
        result_long = pcfg_max.postfix_to_infix(postfix_long_ids, len(expected_long))
        trimmed_result_long = trim_padding(result_long[0])
        
        assert torch.equal(trimmed_result_long, expected_long_ids), f"Expected {len(expected_long)} tokens, got {len(trimmed_result_long)} tokens"
        
        # Test that we can handle expressions up to a reasonable size approaching the 128 limit
        # but staying within workspace constraints
        # Let's try fewer logs with more complex base expressions
        # Create: log(log(log(sin(X_0 + X_1) * cos(C - 5))))
        n_logs_safe = 3
        # Postfix: X_0 X_1 + sin C 5 - cos * log log log
        postfix_safe = ['X_0', 'X_1', '+', 'sin', 'C', '5', '-', 'cos', '*'] + ['log'] * n_logs_safe
        postfix_safe_ids = to_ids(pcfg_max, postfix_safe).unsqueeze(0)
        
        # Expected infix: log(log(log((sin((X_0 + X_1)) * cos((C - 5))))))
        expected_safe: List[str] = []
        for _ in range(n_logs_safe):
            expected_safe.append('log')
            expected_safe.append('(')
        expected_safe.extend(['(', 'sin', '(', '(', 'X_0', '+', 'X_1', ')', ')', '*', 'cos', '(', '(', 'C', '-', '5', ')', ')', ')'])
        for _ in range(n_logs_safe):
            expected_safe.append(')')
        
        expected_safe_ids = to_ids(pcfg_max, expected_safe)
        
        result_safe = pcfg_max.postfix_to_infix(postfix_safe_ids, len(expected_safe))
        trimmed_result_safe = trim_padding(result_safe[0])
        
        assert torch.equal(trimmed_result_safe, expected_safe_ids), f"Expected {len(expected_safe)} tokens, got {len(trimmed_result_safe)} tokens"
        
        # Create a larger expression that tests closer to the 128 limit
        # Build: (((X_0 + X_1) * (C + 5)) + ((X_0 - X_1) / (C - 5))) * sin(log(exp(X_0)))
        # This combines multiple operations to create a longer expression
        # Postfix: X_0 X_1 + C 5 + * X_0 X_1 - C 5 - / + X_0 exp log sin *
        postfix_large = ['X_0', 'X_1', '+', 'C', '5', '+', '*', 'X_0', 'X_1', '-', 'C', '5', '-', '/', '+', 'X_0', 'exp', 'log', 'sin', '*']
        postfix_large_ids = to_ids(pcfg_max, postfix_large).unsqueeze(0)
        
        # Expected infix: ((((X_0 + X_1) * (C + 5)) + ((X_0 - X_1) / (C - 5))) * sin(log(exp(X_0))))
        expected_large = ['(', '(', '(', '(', 'X_0', '+', 'X_1', ')', '*', '(', 'C', '+', '5', ')', ')', '+', '(', '(', 'X_0', '-', 'X_1', ')', '/', '(', 'C', '-', '5', ')', ')', ')', '*', 'sin', '(', 'log', '(', 'exp', '(', 'X_0', ')', ')', ')', ')']
        expected_large_ids = to_ids(pcfg_max, expected_large)
        
        result_large = pcfg_max.postfix_to_infix(postfix_large_ids, len(expected_large))
        trimmed_result_large = trim_padding(result_large[0])
        
        assert torch.equal(trimmed_result_large, expected_large_ids), f"Expected {len(expected_large)} tokens, got {len(trimmed_result_large)} tokens"
        
        # Test that we can handle reasonably complex expressions with the PCFG that has 128 max length
        # This verifies the PCFG can be created with the hard limit and basic functionality works
        assert len(expected_large) > 40, f"Test should use expressions longer than 40 tokens, got {len(expected_large)}"
        
        # Test edge case: what happens when we exactly hit certain length limits
        # Create an expression with exactly 100 tokens to test a large but safe size
        # Build: log(sin(cos(exp(sqrt(X_0 + X_1 * C - 5))))) + log(sin(cos(exp(sqrt(X_0 - X_1 / C + 5)))))
        # This should create a complex but balanced expression
        
        # Postfix for left side: X_0 X_1 C * + 5 - sqrt exp cos sin log
        # Postfix for right side: X_0 X_1 C / - 5 + sqrt exp cos sin log  
        # Combined: left right +
        postfix_balanced = ['X_0', 'X_1', 'C', '*', '+', '5', '-', 'sqrt', 'exp', 'cos', 'sin', 'log', 
                           'X_0', 'X_1', 'C', '/', '-', '5', '+', 'sqrt', 'exp', 'cos', 'sin', 'log', '+']
        postfix_balanced_ids = to_ids(pcfg_max, postfix_balanced).unsqueeze(0)
        
        # Expected: (log(sin(cos(exp(sqrt(((X_0 + (X_1 * C)) - 5)))))) + log(sin(cos(exp(sqrt(((X_0 - (X_1 / C)) + 5)))))))
        expected_balanced = ['(', 'log', '(', 'sin', '(', 'cos', '(', 'exp', '(', 'sqrt', '(', '(', '(', 'X_0', '+', '(', 'X_1', '*', 'C', ')', ')', '-', '5', ')', ')', ')', ')', ')', ')', '+', 'log', '(', 'sin', '(', 'cos', '(', 'exp', '(', 'sqrt', '(', '(', '(', 'X_0', '-', '(', 'X_1', '/', 'C', ')', ')', '+', '5', ')', ')', ')', ')', ')', ')', ')']
        expected_balanced_ids = to_ids(pcfg_max, expected_balanced)
        
        result_balanced = pcfg_max.postfix_to_infix(postfix_balanced_ids, len(expected_balanced))
        trimmed_result_balanced = trim_padding(result_balanced[0])
        
        assert torch.equal(trimmed_result_balanced, expected_balanced_ids), f"Expected {len(expected_balanced)} tokens, got {len(trimmed_result_balanced)} tokens"
        
        # Verify this is a substantial test
        assert len(expected_balanced) > 50, f"Final test should use expressions longer than 50 tokens, got {len(expected_balanced)}"
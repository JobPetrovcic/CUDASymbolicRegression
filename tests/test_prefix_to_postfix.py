#!/usr/bin/env python3
"""
Comprehensive test suite for the prefix_to_post        # Test: op 2 3 -> 2 3 op
        prefix_expr = [op_id, two_id, three_id, no_op, no_op]
        expected_postfix = [two_id, three_id, op_id, no_op, no_op]
        
        prefix_tensor = torch.tensor([prefix_expr], dtype=torch.long, device=device)
        expected_tensor = torch.tensor([expected_postfix], dtype=torch.long, device=device)
        
        postfix_result, parents_result = pcfg.prefix_to_postfix_parent(prefix_tensor)
        
        assert torch.equal(postfix_result, expected_tensor), \
            f"Failed for {operator_symbol}: expected {expected_postfix}, got {postfix_result.tolist()[0]}"
        
        # Check that parents tensor has the correct shape
        assert parents_result.shape == postfix_result.shape, \
            f"Parents tensor shape {parents_result.shape} doesn't match postfix shape {postfix_result.shape}"conversion functionality.
Tests individual operators, batching, and both CPU and CUDA implementations.
"""

import pytest
import torch
from typing import List, Tuple
import symbolic_torch
from tests.utils import get_cuda_device_with_min_memory


# Test grammar that includes most common operators
test_grammar = """
S -> E [1.0]
E -> E + E [0.2]
E -> E - E [0.2]
E -> E * E [0.2]
E -> E / E [0.2]
E -> sin E [0.05]
E -> cos E [0.05]
E -> X_0 [0.05]
E -> 2 [0.05]
"""


@pytest.fixture(params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """Parametrized fixture for CPU and CUDA devices."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this system.")
    if request.param == "cpu":
        return torch.device("cpu")
    else:
        index = get_cuda_device_with_min_memory()
        return torch.device(f"cuda:{index}")


@pytest.fixture
def pcfg(device: torch.device) -> symbolic_torch.ProbabilisticContextFreeGrammar:
    """Create a PCFG instance for the given device."""
    return symbolic_torch.ProbabilisticContextFreeGrammar(
        test_grammar, "S", 30, 2, device
    )


def get_no_op() -> int:
    """Get NO_OP symbol ID."""
    return int(symbolic_torch.Operator.NO_OP)


class TestBinaryOperators:
    """Test each binary operator individually."""
    
    @pytest.mark.parametrize("operator_symbol", ["+", "-", "*", "/", "^"])
    def test_binary_operator_simple(self, device: torch.device, operator_symbol: str):
        """Test each binary operator with simple operands."""
        simple_grammar = f"""
        S -> E [1.0]
        E -> E {operator_symbol} E [0.25]
        E -> 2 [0.375]
        E -> 3 [0.375]
        """
        
        pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            simple_grammar, "S", 10, 0, device
        )
        
        op_id = pcfg.get_symbol_id(operator_symbol)
        two_id = pcfg.get_symbol_id("2")
        three_id = pcfg.get_symbol_id("3")
        no_op = get_no_op()
        
        # Test: op 2 3 -> 2 3 op
        prefix_expr = [op_id, two_id, three_id, no_op, no_op]
        expected_postfix = [two_id, three_id, op_id, no_op, no_op]
        
        prefix_tensor = torch.tensor([prefix_expr], dtype=torch.long, device=device)
        expected_tensor = torch.tensor([expected_postfix], dtype=torch.long, device=device)
        
        postfix_result, parents_result = pcfg.prefix_to_postfix_parent(prefix_tensor)
        
        assert torch.equal(postfix_result, expected_tensor), \
            f"Failed for {operator_symbol}: expected {expected_postfix}, got {postfix_result.tolist()[0]}"
    
    def test_binary_operators_batch(self, device: torch.device):
        """Test all binary operators together in a batch."""
        binary_grammar = """
        S -> E [1.0]
        E -> E + E [0.15]
        E -> E - E [0.15]
        E -> E * E [0.15]
        E -> E / E [0.15]
        E -> E ^ E [0.15]
        E -> 2 [0.125]
        E -> 3 [0.125]
        """
        
        pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            binary_grammar, "S", 10, 0, device
        )
        
        operators = ["+", "-", "*", "/", "^"]
        two_id = pcfg.get_symbol_id("2")
        three_id = pcfg.get_symbol_id("3")
        no_op = get_no_op()
        
        expressions = []
        expected_results = []
        
        for op_symbol in operators:
            op_id = pcfg.get_symbol_id(op_symbol)
            expressions.append([op_id, two_id, three_id, no_op, no_op])
            expected_results.append([two_id, three_id, op_id, no_op, no_op])
        
        batch_tensor = torch.tensor(expressions, dtype=torch.long, device=device)
        expected_tensor = torch.tensor(expected_results, dtype=torch.long, device=device)
        
        postfix_result, parents_result = pcfg.prefix_to_postfix_parent(batch_tensor)
        
        assert torch.equal(postfix_result, expected_tensor), \
            f"Batch binary operators test failed"
        
        # Check that parents tensor has the correct shape
        assert parents_result.shape == postfix_result.shape, \
            f"Parents tensor shape {parents_result.shape} doesn't match postfix shape {postfix_result.shape}"


class TestUnaryOperators:
    """Test each unary operator individually."""
    
    @pytest.mark.parametrize("operator_symbol", [
        "sin", "cos", "exp", "log", "sqrt", "tan", "arcsin", "arccos", "arctan",
        "sinh", "cosh", "tanh", "floor", "ceil", "ln", "log10", "neg", "inv", 
        "cube", "fourth", "fifth", "square"
    ])
    def test_unary_operator_simple(self, device: torch.device, operator_symbol: str):
        """Test each unary operator with a simple operand."""
        unary_grammar = f"""
        S -> E [1.0]
        E -> {operator_symbol} E [0.5]
        E -> 2 [0.5]
        """
        
        pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            unary_grammar, "S", 10, 0, device
        )
        
        op_id = pcfg.get_symbol_id(operator_symbol)
        two_id = pcfg.get_symbol_id("2")
        no_op = get_no_op()
        
        # Test: op 2 -> 2 op
        prefix_expr = [op_id, two_id, no_op, no_op]
        expected_postfix = [two_id, op_id, no_op, no_op]
        
        prefix_tensor = torch.tensor([prefix_expr], dtype=torch.long, device=device)
        expected_tensor = torch.tensor([expected_postfix], dtype=torch.long, device=device)
        
        postfix_result, parents_result = pcfg.prefix_to_postfix_parent(prefix_tensor)
        
        assert torch.equal(postfix_result, expected_tensor), \
            f"Failed for {operator_symbol}: expected {expected_postfix}, got {postfix_result.tolist()[0]}"
        
        # Check that parents tensor has the correct shape
        assert parents_result.shape == postfix_result.shape, \
            f"Parents tensor shape {parents_result.shape} doesn't match postfix shape {postfix_result.shape}"
    
    def test_unary_operators_batch(self, device: torch.device):
        """Test multiple unary operators together in a batch."""
        pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            test_grammar, "S", 30, 2, device
        )
        
        # Test a subset of unary operators
        operators = ["sin", "cos", "exp", "log", "sqrt", "square", "neg", "inv"]
        two_id = pcfg.get_symbol_id("2")
        no_op = get_no_op()
        
        expressions = []
        expected_results = []
        
        for op_symbol in operators:
            op_id = pcfg.get_symbol_id(op_symbol)
            expressions.append([op_id, two_id, no_op, no_op])
            expected_results.append([two_id, op_id, no_op, no_op])
        
        batch_tensor = torch.tensor(expressions, dtype=torch.long, device=device)
        expected_tensor = torch.tensor(expected_results, dtype=torch.long, device=device)
        
        postfix_result, parents_result = pcfg.prefix_to_postfix_parent(batch_tensor)
        
        assert torch.equal(postfix_result, expected_tensor), \
            f"Batch unary operators test failed"


class TestVariablesAndConstants:
    """Test conversion with variables and constants."""
    
    def test_variables(self, device: torch.device):
        """Test expressions with variables."""
        var_grammar = """
        S -> E [1.0]
        E -> E + E [0.25]
        E -> E * E [0.25]
        E -> X_0 [0.25]
        E -> X_1 [0.25]
        """
        
        pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            var_grammar, "S", 10, 2, device
        )
        
        plus_id = pcfg.get_symbol_id("+")
        mult_id = pcfg.get_symbol_id("*")
        x0_id = pcfg.get_symbol_id("X_0")
        x1_id = pcfg.get_symbol_id("X_1")
        no_op = get_no_op()
        
        test_cases = [
            ([plus_id, x0_id, x1_id, no_op, no_op], [x0_id, x1_id, plus_id, no_op, no_op], "X_0 + X_1"),
            ([mult_id, x0_id, x1_id, no_op, no_op], [x0_id, x1_id, mult_id, no_op, no_op], "X_0 * X_1"),
        ]
        
        for prefix_expr, expected_postfix, description in test_cases:
            prefix_tensor = torch.tensor([prefix_expr], dtype=torch.long, device=device)
            expected_tensor = torch.tensor([expected_postfix], dtype=torch.long, device=device)
            
            postfix_result, parents_result = pcfg.prefix_to_postfix_parent(prefix_tensor)
            
            assert torch.equal(postfix_result, expected_tensor), \
                f"Failed for {description}: expected {expected_postfix}, got {postfix_result.tolist()[0]}"
    
    def test_constants(self, device: torch.device):
        """Test expressions with different constants."""
        const_grammar = """
        S -> E [1.0]
        E -> E + E [0.25]
        E -> 2 [0.25]
        E -> 3 [0.25]
        E -> 5 [0.25]
        """
        
        pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            const_grammar, "S", 10, 0, device
        )
        
        plus_id = pcfg.get_symbol_id("+")
        two_id = pcfg.get_symbol_id("2")
        three_id = pcfg.get_symbol_id("3")
        five_id = pcfg.get_symbol_id("5")
        no_op = get_no_op()
        
        test_cases = [
            ([plus_id, two_id, three_id, no_op, no_op], [two_id, three_id, plus_id, no_op, no_op], "2 + 3"),
            ([plus_id, five_id, two_id, no_op, no_op], [five_id, two_id, plus_id, no_op, no_op], "5 + 2"),
        ]
        
        for prefix_expr, expected_postfix, description in test_cases:
            prefix_tensor = torch.tensor([prefix_expr], dtype=torch.long, device=device)
            expected_tensor = torch.tensor([expected_postfix], dtype=torch.long, device=device)
            
            postfix_result, parents_result = pcfg.prefix_to_postfix_parent(prefix_tensor)
            
            assert torch.equal(postfix_result, expected_tensor), \
                f"Failed for {description}: expected {expected_postfix}, got {postfix_result.tolist()[0]}"


class TestComplexExpressions:
    """Test complex nested expressions."""
    
    def test_nested_binary_operations(self, device: torch.device):
        """Test nested binary operations."""
        pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            test_grammar, "S", 30, 2, device
        )
        
        plus_id = pcfg.get_symbol_id("+")
        mult_id = pcfg.get_symbol_id("*")
        two_id = pcfg.get_symbol_id("2")
        three_id = pcfg.get_symbol_id("3")
        five_id = pcfg.get_symbol_id("5")
        no_op = get_no_op()
        
        # Test: + (* 2 3) 5 -> 2 3 * 5 +
        prefix_expr = [plus_id, mult_id, two_id, three_id, five_id, no_op, no_op, no_op]
        expected_postfix = [two_id, three_id, mult_id, five_id, plus_id, no_op, no_op, no_op]
        
        prefix_tensor = torch.tensor([prefix_expr], dtype=torch.long, device=device)
        expected_tensor = torch.tensor([expected_postfix], dtype=torch.long, device=device)
        
        postfix_result, parents_result = pcfg.prefix_to_postfix_parent(prefix_tensor)
        
        assert torch.equal(postfix_result, expected_tensor), \
            f"Failed for nested binary: expected {expected_postfix}, got {postfix_result.tolist()[0]}"
    
    def test_nested_unary_operations(self, device: torch.device):
        """Test nested unary operations."""
        pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            test_grammar, "S", 30, 2, device
        )
        
        sin_id = pcfg.get_symbol_id("sin")
        cos_id = pcfg.get_symbol_id("cos")
        x0_id = pcfg.get_symbol_id("X_0")
        no_op = get_no_op()
        
        # Test: sin cos X_0 -> X_0 cos sin
        prefix_expr = [sin_id, cos_id, x0_id, no_op, no_op, no_op]
        expected_postfix = [x0_id, cos_id, sin_id, no_op, no_op, no_op]
        
        prefix_tensor = torch.tensor([prefix_expr], dtype=torch.long, device=device)
        expected_tensor = torch.tensor([expected_postfix], dtype=torch.long, device=device)
        
        postfix_result, parents_result = pcfg.prefix_to_postfix_parent(prefix_tensor)
        
        assert torch.equal(postfix_result, expected_tensor), \
            f"Failed for nested unary: expected {expected_postfix}, got {postfix_result.tolist()[0]}"
    
    def test_mixed_operations(self, device: torch.device):
        """Test mixed unary and binary operations."""
        pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            test_grammar, "S", 30, 2, device
        )
        
        plus_id = pcfg.get_symbol_id("+")
        sin_id = pcfg.get_symbol_id("sin")
        mult_id = pcfg.get_symbol_id("*")
        x0_id = pcfg.get_symbol_id("X_0")
        two_id = pcfg.get_symbol_id("2")
        three_id = pcfg.get_symbol_id("3")
        no_op = get_no_op()
        
        # Test: + (sin X_0) (* 2 3) -> X_0 sin 2 3 * +
        prefix_expr = [plus_id, sin_id, x0_id, mult_id, two_id, three_id, no_op, no_op, no_op, no_op]
        expected_postfix = [x0_id, sin_id, two_id, three_id, mult_id, plus_id, no_op, no_op, no_op, no_op]
        
        prefix_tensor = torch.tensor([prefix_expr], dtype=torch.long, device=device)
        expected_tensor = torch.tensor([expected_postfix], dtype=torch.long, device=device)
        
        postfix_result, parents_result = pcfg.prefix_to_postfix_parent(prefix_tensor)
        
        assert torch.equal(postfix_result, expected_tensor), \
            f"Failed for mixed operations: expected {expected_postfix}, got {postfix_result.tolist()[0]}"


class TestBatchProcessing:
    """Test batch processing functionality."""
    
    def test_batch_consistency(self, device: torch.device):
        """Test that batched conversion gives same results as individual conversions."""
        pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            test_grammar, "S", 30, 2, device
        )
        
        plus_id = pcfg.get_symbol_id("+")
        mult_id = pcfg.get_symbol_id("*")
        sin_id = pcfg.get_symbol_id("sin")
        x0_id = pcfg.get_symbol_id("X_0")
        x1_id = pcfg.get_symbol_id("X_1")
        two_id = pcfg.get_symbol_id("2")
        three_id = pcfg.get_symbol_id("3")
        no_op = get_no_op()
        
        # Create multiple test expressions of different complexities
        test_expressions = [
            [plus_id, two_id, three_id, no_op, no_op, no_op, no_op, no_op],  # + 2 3
            [mult_id, x0_id, two_id, no_op, no_op, no_op, no_op, no_op],     # * X_0 2
            [sin_id, x1_id, no_op, no_op, no_op, no_op, no_op, no_op],       # sin X_1
            [plus_id, sin_id, x0_id, mult_id, two_id, three_id, no_op, no_op], # + (sin X_0) (* 2 3)
        ]
        
        max_len = 8
        
        # Convert individually
        individual_results : List[torch.Tensor] = []
        for expr in test_expressions:
            single_tensor = torch.tensor([expr], dtype=torch.long, device=device)
            postfix_result, parents_result = pcfg.prefix_to_postfix_parent(single_tensor, max_len)
            individual_results.append(postfix_result)
        
        # Convert as batch
        batch_tensor = torch.tensor(test_expressions, dtype=torch.long, device=device)
        postfix_batch_result, parents_batch_postfix_result = pcfg.prefix_to_postfix_parent(batch_tensor, max_len)
        
        # Compare results
        for i, individual_result in enumerate(individual_results):
            assert torch.equal(postfix_batch_result[i], individual_result[0]), \
                f"Batch result differs from individual result for expression {i}: " \
                f"batch={postfix_batch_result[i].tolist()}, individual={individual_result[0].tolist()}"
    
    def test_large_batch(self, device: torch.device):
        """Test processing of large batches."""
        pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            test_grammar, "S", 30, 2, device
        )
        
        plus_id = pcfg.get_symbol_id("+")
        mult_id = pcfg.get_symbol_id("*")
        sin_id = pcfg.get_symbol_id("sin")
        x0_id = pcfg.get_symbol_id("X_0")
        two_id = pcfg.get_symbol_id("2")
        three_id = pcfg.get_symbol_id("3")
        no_op = get_no_op()
        
        # Create a large batch of expressions
        batch_size = 64
        expressions = []
        expected_results = []
        
        for i in range(batch_size):
            if i % 3 == 0:  # Binary
                expr = [plus_id, two_id, three_id, no_op, no_op, no_op]
                expected = [two_id, three_id, plus_id, no_op, no_op, no_op]
            elif i % 3 == 1:  # Unary
                expr = [sin_id, x0_id, no_op, no_op, no_op, no_op]
                expected = [x0_id, sin_id, no_op, no_op, no_op, no_op]
            else:  # Complex
                expr = [mult_id, plus_id, two_id, three_id, x0_id, no_op]
                expected = [two_id, three_id, plus_id, x0_id, mult_id, no_op]
            
            expressions.append(expr)
            expected_results.append(expected)
        
        batch_tensor = torch.tensor(expressions, dtype=torch.long, device=device)
        expected_tensor = torch.tensor(expected_results, dtype=torch.long, device=device)
        
        postfix_result, parents_result = pcfg.prefix_to_postfix_parent(batch_tensor)
        
        assert torch.equal(postfix_result, expected_tensor), \
            f"Large batch processing failed"
    
    def test_mixed_length_batch(self, device: torch.device):
        """Test batch with expressions of different lengths."""
        pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            test_grammar, "S", 30, 2, device
        )
        
        plus_id = pcfg.get_symbol_id("+")
        sin_id = pcfg.get_symbol_id("sin")
        x0_id = pcfg.get_symbol_id("X_0")
        two_id = pcfg.get_symbol_id("2")
        three_id = pcfg.get_symbol_id("3")
        no_op = get_no_op()
        
        # Different length expressions (padded to same length)
        max_len = 8
        test_cases = [
            # (prefix, expected_postfix, actual_length)
            ([two_id] + [no_op] * 7, [two_id] + [no_op] * 7, 1),
            ([sin_id, x0_id] + [no_op] * 6, [x0_id, sin_id] + [no_op] * 6, 2),
            ([plus_id, two_id, three_id] + [no_op] * 5, [two_id, three_id, plus_id] + [no_op] * 5, 3),
        ]
        
        expressions = [case[0] for case in test_cases]
        expected_results = [case[1] for case in test_cases]
        
        batch_tensor = torch.tensor(expressions, dtype=torch.long, device=device)
        expected_tensor = torch.tensor(expected_results, dtype=torch.long, device=device)
        
        postfix_result, parents_result = pcfg.prefix_to_postfix_parent(batch_tensor, max_len)
        
        assert torch.equal(postfix_result, expected_tensor), \
            f"Mixed length batch processing failed"


class TestDeviceConsistency:
    """Test consistency between CPU and CUDA implementations."""
    
    def test_cpu_cuda_simple_expressions(self):
        """Test that CPU and CUDA produce identical results for simple expressions."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available on this system.")
        
        # Create PCFGs for both devices
        cpu_pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            test_grammar, "S", 30, 2, torch.device("cpu")
        )
        cuda_index = get_cuda_device_with_min_memory()
        cuda_pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            test_grammar, "S", 30, 2, torch.device(f"cuda:{cuda_index}")
        )
        
        # Test simple expressions
        plus_id = cpu_pcfg.get_symbol_id("+")
        sin_id = cpu_pcfg.get_symbol_id("sin")
        x0_id = cpu_pcfg.get_symbol_id("X_0")
        two_id = cpu_pcfg.get_symbol_id("2")
        three_id = cpu_pcfg.get_symbol_id("3")
        no_op = get_no_op()
        
        test_expressions = [
            [plus_id, two_id, three_id, no_op, no_op],
            [sin_id, x0_id, no_op, no_op, no_op],
            [plus_id, sin_id, x0_id, two_id, no_op],
        ]
        
        max_len = 5
        
        for expr in test_expressions:
            # CPU conversion
            cpu_tensor = torch.tensor([expr], dtype=torch.long, device="cpu")
            cpu_postfix_result, cpu_parents_result = cpu_pcfg.prefix_to_postfix_parent(cpu_tensor, max_len)
            
            # CUDA conversion
            cuda_tensor = torch.tensor([expr], dtype=torch.long, device=f"cuda:{cuda_index}")
            cuda_postfix_result, cuda_parents_result = cuda_pcfg.prefix_to_postfix_parent(cuda_tensor, max_len)
            
            # Compare results
            assert torch.equal(cpu_postfix_result, cuda_postfix_result.cpu()), \
                f"CPU and CUDA results differ for expression {expr}: " \
                f"CPU={cpu_postfix_result.tolist()}, CUDA={cuda_postfix_result.cpu().tolist()}"
            assert torch.equal(cpu_parents_result, cuda_parents_result.cpu()), \
                f"CPU and CUDA parents results differ for expression {expr}: " \
                f"CPU={cpu_parents_result.tolist()}, CUDA={cuda_parents_result.cpu().tolist()}"

    def test_cpu_cuda_batch_expressions(self):
        """Test that CPU and CUDA produce identical results for batch expressions."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available on this system.")
        
        # Create PCFGs for both devices
        cpu_pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            test_grammar, "S", 30, 2, torch.device("cpu")
        )
        cuda_index = get_cuda_device_with_min_memory()
        cuda_pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            test_grammar, "S", 30, 2, torch.device(f"cuda:{cuda_index}")
        )
        
        # Create a batch of different expressions
        plus_id = cpu_pcfg.get_symbol_id("+")
        mult_id = cpu_pcfg.get_symbol_id("*")
        sin_id = cpu_pcfg.get_symbol_id("sin")
        cos_id = cpu_pcfg.get_symbol_id("cos")
        x0_id = cpu_pcfg.get_symbol_id("X_0")
        x1_id = cpu_pcfg.get_symbol_id("X_1")
        two_id = cpu_pcfg.get_symbol_id("2")
        three_id = cpu_pcfg.get_symbol_id("3")
        no_op = get_no_op()
        
        test_expressions = [
            [plus_id, two_id, three_id, no_op, no_op, no_op, no_op, no_op],
            [mult_id, x0_id, x1_id, no_op, no_op, no_op, no_op, no_op],
            [sin_id, cos_id, x0_id, no_op, no_op, no_op, no_op, no_op],
            [plus_id, mult_id, two_id, three_id, sin_id, x0_id, no_op, no_op],
        ]
        
        max_len = 8
        
        # CPU conversion
        cpu_tensor = torch.tensor(test_expressions, dtype=torch.long, device="cpu")
        cpu_postfix_result, cpu_parents_result = cpu_pcfg.prefix_to_postfix_parent(cpu_tensor, max_len)
        
        # CUDA conversion
        cuda_tensor = torch.tensor(test_expressions, dtype=torch.long, device=f"cuda:{cuda_index}")
        cuda_postfix_result, cuda_parents_result = cuda_pcfg.prefix_to_postfix_parent(cuda_tensor, max_len)
        
        # Compare results
        assert torch.equal(cpu_postfix_result, cuda_postfix_result.cpu()), \
            f"CPU and CUDA batch results differ: " \
            f"CPU={cpu_postfix_result.tolist()}, CUDA={cuda_postfix_result.cpu().tolist()}"
        assert torch.equal(cpu_parents_result, cuda_parents_result.cpu()), \
            f"CPU and CUDA parents results differ: " \
            f"CPU={cpu_parents_result.tolist()}, CUDA={cuda_parents_result.cpu().tolist()}"


class TestEdgeCases:
    """Test edge cases and special conditions."""
    
    def test_single_operand(self, device: torch.device):
        """Test single operand expressions."""
        pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            test_grammar, "S", 30, 2, device
        )
        
        two_id = pcfg.get_symbol_id("2")
        x0_id = pcfg.get_symbol_id("X_0")
        no_op = get_no_op()
        
        test_cases = [
            ([two_id, no_op, no_op], [two_id, no_op, no_op], "constant"),
            ([x0_id, no_op, no_op], [x0_id, no_op, no_op], "variable"),
        ]
        
        for prefix_expr, expected_postfix, description in test_cases:
            prefix_tensor = torch.tensor([prefix_expr], dtype=torch.long, device=device)
            expected_tensor = torch.tensor([expected_postfix], dtype=torch.long, device=device)
            
            postfix_result, parents_result = pcfg.prefix_to_postfix_parent(prefix_tensor)
            
            assert torch.equal(postfix_result, expected_tensor), \
                f"Single operand test failed for {description}: expected {expected_postfix}, got {postfix_result.tolist()[0]}"
    
    def test_extensive_padding(self, device: torch.device):
        """Test expressions with extensive padding."""
        pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            test_grammar, "S", 30, 2, device
        )
        
        plus_id = pcfg.get_symbol_id("+")
        two_id = pcfg.get_symbol_id("2")
        three_id = pcfg.get_symbol_id("3")
        no_op = get_no_op()
        
        # Test with lots of padding
        max_len = 20
        prefix_expr = [plus_id, two_id, three_id] + [no_op] * 17
        expected_postfix = [two_id, three_id, plus_id] + [no_op] * 17
        
        prefix_tensor = torch.tensor([prefix_expr], dtype=torch.long, device=device)
        expected_tensor = torch.tensor([expected_postfix], dtype=torch.long, device=device)
        
        postfix_result, parents_result = pcfg.prefix_to_postfix_parent(prefix_tensor, max_len)
        
        assert torch.equal(postfix_result, expected_tensor), \
            f"Extensive padding test failed: expected {expected_postfix}, got {postfix_result.tolist()[0]}"
    
    def test_all_operators_comprehensive(self, device: torch.device):
        """Test that all operators work correctly."""
        pcfg = symbolic_torch.ProbabilisticContextFreeGrammar(
            test_grammar, "S", 30, 2, device
        )
        
        # Test all binary operators
        binary_operators = ["+", "-", "*", "/", "^"]
        two_id = pcfg.get_symbol_id("2")
        three_id = pcfg.get_symbol_id("3")
        no_op = get_no_op()
        
        for op_symbol in binary_operators:
            op_id = pcfg.get_symbol_id(op_symbol)
            prefix_expr = [op_id, two_id, three_id, no_op, no_op]
            expected_postfix = [two_id, three_id, op_id, no_op, no_op]
            
            prefix_tensor = torch.tensor([prefix_expr], dtype=torch.long, device=device)
            expected_tensor = torch.tensor([expected_postfix], dtype=torch.long, device=device)
            
            postfix_result, parents_result = pcfg.prefix_to_postfix_parent(prefix_tensor)
            
            assert torch.equal(postfix_result, expected_tensor), \
                f"Comprehensive test failed for binary operator {op_symbol}"
        
        # Test all unary operators that are in the grammar
        unary_operators = [
            "sin", "cos", "exp", "log", "sqrt", "tan", "arcsin", "arccos", "arctan",
            "sinh", "cosh", "tanh", "floor", "ceil", "ln", "log10", "neg", "inv", 
            "cube", "fourth", "fifth", "square"
        ]
        
        for op_symbol in unary_operators:
            op_id = pcfg.get_symbol_id(op_symbol)
            prefix_expr = [op_id, two_id, no_op, no_op]
            expected_postfix = [two_id, op_id, no_op, no_op]
            
            prefix_tensor = torch.tensor([prefix_expr], dtype=torch.long, device=device)
            expected_tensor = torch.tensor([expected_postfix], dtype=torch.long, device=device)
            
            postfix_result, parents_result = pcfg.prefix_to_postfix_parent(prefix_tensor)
            
            assert torch.equal(postfix_result, expected_tensor), \
                f"Comprehensive test failed for unary operator {op_symbol}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

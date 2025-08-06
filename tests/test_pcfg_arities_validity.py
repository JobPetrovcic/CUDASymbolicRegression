import pytest
import torch
from symbolic_torch import ProbabilisticContextFreeGrammar, Operator


@pytest.fixture(params=["cpu", "cuda:1"])
def device_str(request):
    """Parametrized fixture for device testing."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param


@pytest.fixture
def pcfg(device_str):
    """Create a PCFG instance for testing."""
    # Use a minimal grammar for testing
    grammar = "S -> 1 [1.0]"
    return ProbabilisticContextFreeGrammar(
        grammar=grammar,
        start_symbol="S",
        padded_maximum_length=32,
        n_variables=5,  # Test with 5 variables (X_0 to X_4)
        device=torch.device(device_str),
    )


def test_get_arities_hardcoded_values(pcfg):
    """
    Test that get_arities() returns the correct arities for all hardcoded operator IDs.
    This test verifies the arity mapping for all operators including gaps in the numbering.
    """
    # Get the arities tensor from the PCFG
    arities = pcfg.get_arities()
    
    # Hardcoded expected arities for each operator ID
    # Note: This assumes n_operators covers all operators including variables
    expected_arities = {
        # Meta operators
        0: 0,   # NO_OP
        1: 0,   # LEARNABLE_CONSTANT
        
        # Hard constants
        2: 0,   # CONST_1
        3: 0,   # CONST_2
        4: 0,   # CONST_3
        5: 0,   # CONST_4
        6: 0,   # CONST_5
        7: 0,   # PI
        8: 0,   # E
        
        # Gap from 9 (no operator defined)
        9: 0,   # Undefined operator - should have arity 0
        
        # Unary functions
        10: 1,  # SIN
        11: 1,  # COS
        12: 1,  # EXP
        13: 1,  # LOG
        14: 1,  # SQUARE
        15: 1,  # SQRT
        
        # Gaps from 16-19 (no operators defined)
        16: 0,  # Undefined operator
        17: 0,  # Undefined operator
        18: 0,  # Undefined operator
        19: 0,  # Undefined operator
        
        # Binary functions
        20: 2,  # ADD
        21: 2,  # SUB
        22: 2,  # MUL
        23: 2,  # DIV
        24: 2,  # POW
        
        # Gaps from 25-29 (no operators defined)
        25: 0,  # Undefined operator
        26: 0,  # Undefined operator
        27: 0,  # Undefined operator
        28: 0,  # Undefined operator
        29: 0,  # Undefined operator
        
        # Variables (VAR_START_ID = 30, with n_variables = 5)
        30: 0,  # X_0
        31: 0,  # X_1
        32: 0,  # X_2
        33: 0,  # X_3
        34: 0,  # X_4
    }
    
    # Verify that we have the expected number of operators
    expected_n_operators = max(expected_arities.keys()) + 1
    assert pcfg.n_operators == expected_n_operators, f"Expected {expected_n_operators} operators, got {pcfg.n_operators}"
    
    # Check each operator's arity
    for op_id, expected_arity in expected_arities.items():
        actual_arity = arities[op_id].item()
        assert actual_arity == expected_arity, \
            f"Operator {op_id} has arity {actual_arity}, expected {expected_arity}"


def test_valid_ops_hardcoded_values(pcfg):
    """
    Test that valid_ops() returns the correct validity for all hardcoded operator IDs.
    This test verifies which operators are considered valid, including gaps and NO_OP.
    """
    # Get the validity tensor from the PCFG
    valid_ops = pcfg.valid_ops()
    
    # Hardcoded expected validity for each operator ID
    # Based on the is_valid_op function in operators.h
    expected_validity = {
        # Meta operators
        0: False,  # NO_OP - explicitly invalid
        1: True,   # LEARNABLE_CONSTANT
        
        # Hard constants
        2: True,   # CONST_1
        3: True,   # CONST_2
        4: True,   # CONST_3
        5: True,   # CONST_4
        6: True,   # CONST_5
        7: True,   # PI
        8: True,   # E
        
        # Gap (undefined operator)
        9: False,  # Undefined operator - invalid
        
        # Unary functions
        10: True,  # SIN
        11: True,  # COS
        12: True,  # EXP
        13: True,  # LOG
        14: True,  # SQUARE
        15: True,  # SQRT
        
        # Gaps (undefined operators)
        16: False, # Undefined operator - invalid
        17: False, # Undefined operator - invalid
        18: False, # Undefined operator - invalid
        19: False, # Undefined operator - invalid
        
        # Binary functions
        20: True,  # ADD
        21: True,  # SUB
        22: True,  # MUL
        23: True,  # DIV
        24: True,  # POW
        
        # Gaps (undefined operators)
        25: False, # Undefined operator - invalid
        26: False, # Undefined operator - invalid
        27: False, # Undefined operator - invalid
        28: False, # Undefined operator - invalid
        29: False, # Undefined operator - invalid
        
        # Variables (VAR_START_ID = 30, with n_variables = 5)
        30: True,  # X_0
        31: True,  # X_1
        32: True,  # X_2
        33: True,  # X_3
        34: True,  # X_4
    }
    
    # Verify that we have the expected number of operators
    expected_n_operators = max(expected_validity.keys()) + 1
    assert pcfg.n_operators == expected_n_operators, f"Expected {expected_n_operators} operators, got {pcfg.n_operators}"
    
    # Check each operator's validity
    for op_id, expected_valid in expected_validity.items():
        actual_valid = valid_ops[op_id].item()
        assert actual_valid == expected_valid, \
            f"Operator {op_id} has validity {actual_valid}, expected {expected_valid}"


def test_specific_operator_enum_values(pcfg):
    """
    Test specific operator enum values to ensure they match the expected IDs.
    This serves as a double-check that the enum values haven't changed.
    """
    # Test a few key operators to ensure enum values are as expected
    assert int(Operator.NO_OP) == 0
    assert int(Operator.LEARNABLE_CONSTANT) == 1
    assert int(Operator.CONST_1) == 2
    assert int(Operator.PI) == 7
    assert int(Operator.E) == 8
    assert int(Operator.SIN) == 10
    assert int(Operator.COS) == 11
    assert int(Operator.SQRT) == 15
    assert int(Operator.ADD) == 20
    assert int(Operator.SUB) == 21
    assert int(Operator.POW) == 24
    assert int(Operator.VAR_START_ID) == 30


def test_arities_and_validity_consistency(pcfg):
    """
    Test that the arities and validity are consistent with each other.
    Invalid operators should generally have arity 0, but arity 0 doesn't necessarily mean invalid.
    """
    arities = pcfg.get_arities()
    valid_ops = pcfg.valid_ops()
    
    for i in range(pcfg.n_operators):
        arity = arities[i].item()
        is_valid = valid_ops[i].item()
        
        # All valid operators should have arity >= 0
        if is_valid:
            assert arity >= 0, f"Valid operator {i} has negative arity {arity}"
        
        # NO_OP (operator 0) should be invalid and have arity 0
        if i == int(Operator.NO_OP):
            assert not is_valid, f"NO_OP (operator {i}) should be invalid"
            assert arity == 0, f"NO_OP (operator {i}) should have arity 0, got {arity}"


def test_variable_operators_boundary(pcfg):
    """
    Test that variable operators are correctly identified within the valid range.
    Variables should be valid for X_0 through X_{n_variables-1} and invalid beyond.
    """
    valid_ops = pcfg.valid_ops()
    arities = pcfg.get_arities()
    
    var_start = int(Operator.VAR_START_ID)
    n_variables = pcfg.n_variables
    
    # Variables within range should be valid with arity 0
    for i in range(n_variables):
        var_id = var_start + i
        if var_id < pcfg.n_operators:  # Make sure we don't go out of bounds
            assert valid_ops[var_id].item(), f"Variable X_{i} (operator {var_id}) should be valid"
            assert arities[var_id].item() == 0, f"Variable X_{i} (operator {var_id}) should have arity 0"


def test_tensor_properties(pcfg):
    """
    Test that the returned tensors have the correct properties (device, dtype, shape).
    """
    arities = pcfg.get_arities()
    valid_ops = pcfg.valid_ops()
    
    # Check device
    assert arities.device == pcfg.device, f"Arities tensor on wrong device: {arities.device} vs {pcfg.device}"
    assert valid_ops.device == pcfg.device, f"Valid ops tensor on wrong device: {valid_ops.device} vs {pcfg.device}"
    
    # Check dtype
    assert arities.dtype == torch.int64, f"Arities tensor has wrong dtype: {arities.dtype}"
    assert valid_ops.dtype == torch.bool, f"Valid ops tensor has wrong dtype: {valid_ops.dtype}"
    
    # Check shape
    expected_shape = (pcfg.n_operators,)
    assert arities.shape == expected_shape, f"Arities tensor has wrong shape: {arities.shape} vs {expected_shape}"
    assert valid_ops.shape == expected_shape, f"Valid ops tensor has wrong shape: {valid_ops.shape} vs {expected_shape}"


@pytest.mark.parametrize("n_variables", [0, 1, 3, 10, 30])
def test_different_variable_counts(device_str, n_variables):
    """
    Test get_arities() and valid_ops() with different numbers of variables.
    """
    grammar = "S -> 1 [1.0]"
    pcfg = ProbabilisticContextFreeGrammar(
        grammar=grammar,
        start_symbol="S",
        padded_maximum_length=32,
        n_variables=n_variables,
        device=torch.device(device_str),
    )
    
    valid_ops = pcfg.valid_ops()
    var_start = int(Operator.VAR_START_ID)
    
    # Check that exactly n_variables variables are valid
    for i in range(min(n_variables, pcfg.n_operators - var_start)):
        var_id = var_start + i
        assert valid_ops[var_id].item(), f"Variable X_{i} should be valid with n_variables={n_variables}"
    
    # Check that variables beyond n_variables are invalid (if they exist in the operator range)
    for i in range(n_variables, min(pcfg.n_operators - var_start, 10)):  # Check up to 10 variables beyond
        var_id = var_start + i
        if var_id < pcfg.n_operators:
            assert not valid_ops[var_id].item(), f"Variable X_{i} should be invalid with n_variables={n_variables}"

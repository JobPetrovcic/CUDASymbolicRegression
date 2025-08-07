import pytest
import torch
from symbolic_torch import ProbabilisticContextFreeGrammar, Operator
from tests.utils import get_cuda_device_with_min_memory


@pytest.fixture(params=["cpu", "cuda"])
def device_str(request):
    """Parametrized fixture for device testing."""
    if request.param == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this system.")
    if request.param == "cpu":
        return "cpu"
    else:
        index = get_cuda_device_with_min_memory()
        return f"cuda:{index}"  # Use the device with minimum memory available

@pytest.fixture
def pcfg(device_str : str) -> ProbabilisticContextFreeGrammar:
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
        16: 1,  # TAN
        17: 1,  # ARCSIN
        18: 1,  # ARCCOS
        19: 1,  # ARCTAN
        20: 1,  # SINH
        21: 1,  # COSH
        22: 1,  # TANH
        23: 1,  # FLOOR
        24: 1,  # CEIL
        25: 1,  # LN
        26: 1,  # LOG10
        27: 1,  # NEG
        28: 1,  # INV
        29: 1,  # CUBE
        30: 1,  # FOURTH
        31: 1,  # FIFTH
        
        # Gaps from 32-39 (no operators defined)
        32: 0,  # Undefined operator
        33: 0,  # Undefined operator
        34: 0,  # Undefined operator
        35: 0,  # Undefined operator
        36: 0,  # Undefined operator
        37: 0,  # Undefined operator
        38: 0,  # Undefined operator
        39: 0,  # Undefined operator
        
        # Binary functions
        40: 2,  # ADD
        41: 2,  # SUB
        42: 2,  # MUL
        43: 2,  # DIV
        44: 2,  # POW
        
        # Gaps from 45-49 (no operators defined)
        45: 0,  # Undefined operator
        46: 0,  # Undefined operator
        47: 0,  # Undefined operator
        48: 0,  # Undefined operator
        49: 0,  # Undefined operator
        
        # Variables (VAR_START_ID = 50, with n_variables = 5)
        50: 0,  # X_0
        51: 0,  # X_1
        52: 0,  # X_2
        53: 0,  # X_3
        54: 0,  # X_4
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
        16: True,  # TAN
        17: True,  # ARCSIN
        18: True,  # ARCCOS
        19: True,  # ARCTAN
        20: True,  # SINH
        21: True,  # COSH
        22: True,  # TANH
        23: True,  # FLOOR
        24: True,  # CEIL
        25: True,  # LN
        26: True,  # LOG10
        27: True,  # NEG
        28: True,  # INV
        29: True,  # CUBE
        30: True,  # FOURTH
        31: True,  # FIFTH
        
        # Gaps (undefined operators)
        32: False, # Undefined operator - invalid
        33: False, # Undefined operator - invalid
        34: False, # Undefined operator - invalid
        35: False, # Undefined operator - invalid
        36: False, # Undefined operator - invalid
        37: False, # Undefined operator - invalid
        38: False, # Undefined operator - invalid
        39: False, # Undefined operator - invalid
        
        # Binary functions
        40: True,  # ADD
        41: True,  # SUB
        42: True,  # MUL
        43: True,  # DIV
        44: True,  # POW
        
        # Gaps (undefined operators)
        45: False, # Undefined operator - invalid
        46: False, # Undefined operator - invalid
        47: False, # Undefined operator - invalid
        48: False, # Undefined operator - invalid
        49: False, # Undefined operator - invalid
        
        # Variables (VAR_START_ID = 50, with n_variables = 5)
        50: True,  # X_0
        51: True,  # X_1
        52: True,  # X_2
        53: True,  # X_3
        54: True,  # X_4
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
    assert int(Operator.ADD) == 40
    assert int(Operator.SUB) == 41
    assert int(Operator.POW) == 44
    assert int(Operator.VAR_START_ID) == 50


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

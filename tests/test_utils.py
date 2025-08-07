import pytest
from symbolic_torch import get_arity, Operator



test_cases = [
    # --- Nullary Operators (Arity 0) ---
    (Operator.NO_OP, 0, "NO_OP"),
    (Operator.LEARNABLE_CONSTANT, 0, "LEARNABLE_CONSTANT"),
    (Operator.PI, 0, "PI"),
    (Operator.CONST_1, 0, "CONST_1"),
    (Operator.VAR_START_ID, 0, "VAR_START_ID"),
    (int(Operator.VAR_START_ID) + 10, 0, "VAR_10"), # Test a specific variable

    # --- Unary Operators (Arity 1) ---
    (Operator.SIN, 1, "SIN"),
    (Operator.COS, 1, "COS"),
    (Operator.EXP, 1, "EXP"),
    (Operator.LOG, 1, "LOG"),
    (Operator.SQUARE, 1, "SQUARE"),
    (Operator.SQRT, 1, "SQRT"),
    (Operator.TAN, 1, "TAN"),
    (Operator.ARCSIN, 1, "ARCSIN"),
    (Operator.ARCCOS, 1, "ARCCOS"),
    (Operator.ARCTAN, 1, "ARCTAN"),
    (Operator.SINH, 1, "SINH"),
    (Operator.COSH, 1, "COSH"),
    (Operator.TANH, 1, "TANH"),
    (Operator.FLOOR, 1, "FLOOR"),
    (Operator.CEIL, 1, "CEIL"),
    (Operator.LN, 1, "LN"),
    (Operator.LOG10, 1, "LOG10"),
    (Operator.NEG, 1, "NEG"),
    (Operator.INV, 1, "INV"),
    (Operator.CUBE, 1, "CUBE"),
    (Operator.FOURTH, 1, "FOURTH"),
    (Operator.FIFTH, 1, "FIFTH"),

    # --- Binary Operators (Arity 2) ---
    (Operator.ADD, 2, "ADD"),
    (Operator.SUB, 2, "SUB"),
    (Operator.MUL, 2, "MUL"),
    (Operator.DIV, 2, "DIV"),
    (Operator.POW, 2, "POW"),
]

@pytest.mark.parametrize("operator, expected_arity, test_id", test_cases, ids=[case[2] for case in test_cases])
def test_get_arity(operator : Operator, expected_arity : int, test_id : str):
    """
    Tests the get_arity function from the C++ extension module.

    This test is parameterized to cover nullary, unary, and binary operators,
    ensuring that the C++ implementation correctly identifies the arity for
    each operator type defined in the Operator enum.
    """
    
    # Call the C++ function
    calculated_arity = get_arity(operator)
    
    # Assert that the result is correct
    assert calculated_arity == expected_arity, f"Failed on test ID: {test_id}"
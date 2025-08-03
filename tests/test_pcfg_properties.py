import pytest
import torch
from symbolic_torch import ProbabilisticContextFreeGrammar

# A minimal grammar for testing initialization
test_grammar = "S -> 1 [1.0]"

@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_pcfg_properties_are_exposed_and_correct(device_str: str):
    """
    Tests that the PCFG properties are exposed to Python, are read-only,
    and have the correct values they were initialized with.
    """
    if not torch.cuda.is_available() and device_str == "cuda":
        pytest.skip("CUDA not available")
    
    device = torch.device(device_str)
    
    # Define test parameters
    grammar = test_grammar
    start_symbol = "S"
    padded_maximum_length = 32
    n_variables = 5
    max_tries = 99
    tolerance = 1e-5
    verbose = True

    # Initialize the PCFG
    pcfg = ProbabilisticContextFreeGrammar(
        grammar=grammar,
        start_symbol=start_symbol,
        padded_maximum_length=padded_maximum_length,
        n_variables=n_variables,
        device=device,
        max_tries=max_tries,
        tolerance=tolerance,
        verbose=verbose
    )

    # 1. Assert that the properties exist and have the correct values
    assert pcfg.start_symbol == start_symbol
    assert pcfg.padded_maximum_length == padded_maximum_length
    assert pcfg.n_variables == n_variables
    assert pcfg.max_tries == max_tries
    assert abs(pcfg.tolerance - tolerance) < 1e-9 # Compare floats with tolerance
    assert pcfg.verbose == verbose
    assert pcfg.device == device

    # 2. Assert that the properties are read-only
    with pytest.raises(AttributeError):
        pcfg.start_symbol = "E"
    with pytest.raises(AttributeError):
        pcfg.padded_maximum_length = 64
    with pytest.raises(AttributeError):
        pcfg.n_variables = 10
    with pytest.raises(AttributeError):
        pcfg.max_tries = 200
    with pytest.raises(AttributeError):
        pcfg.tolerance = 1e-7
    with pytest.raises(AttributeError):
        pcfg.verbose = False
    with pytest.raises(AttributeError):
        pcfg.device = torch.device("cpu") if device_str == "cuda" else torch.device("cuda")

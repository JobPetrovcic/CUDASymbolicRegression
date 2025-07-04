# Import the compiled C++ module. The '.so' or '.pyd' file is named _C.
import torch
from . import _C
from ._C import Operator, evaluate

# This makes `from symbolic_extension import evaluate` work
__all__ = [
    "evaluate",
    "Operator",
]
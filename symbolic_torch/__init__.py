# symbolic_torch/__init__.py
# Import the compiled C++ module. The '.so' or '.pyd' file is named _C.
import torch # type: ignore[import]
from ._C import Operator, ProbabilisticContextFreeGrammar, get_arity
from .utils import create_constants
from . import evaluation
from .optimization import optimize

# This makes `from symbolic_extension import evaluate` work
__all__ = [
    "evaluation",
    "Operator",
    "ProbabilisticContextFreeGrammar",
    "create_constants",
    "optimize",
    "get_arity",
]
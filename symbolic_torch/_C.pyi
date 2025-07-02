from typing import Optional, Tuple
from enum import Enum
import torch

# 1. Re-declare the Enum for type hinting purposes
class Operator(Enum):
    NO_OP: int
    LEARNABLE_CONSTANT: int
    CONST_1: int
    SIN: int
    COS: int
    EXP: int
    LOG: int
    SQUARE: int
    SQRT: int
    ADD: int
    SUB: int
    MUL: int
    DIV: int
    VAR_START_ID: int

# 2. Define the signature of the 'evaluate' function
def evaluate(
    X: torch.Tensor,
    Ops: torch.Tensor,
    Ch: torch.Tensor,
    C: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Let M be the maximum number of operations in an expression, B the batch size of the expressions, N the number of datapoints, and n_x the number of variables in the expressions. 

    For an expression we unroll it using the postfix order into separate tensors Ops and Ch.

    X: Input tensor for the variables of shape (
    Ops: Tensor of operations specified by integers (see Operator Enum) of shape (M, B) for the expressions in postfix order. If some operations are not used, they can be set to NO_OP.
    Ch: Tensor of postfix order indices for the children of nodes in th expressions of shape (M, B). If the operation has less than MAX_ARITY = 2 children, the unused children must be set to -1.
    C: Optional tensor of learnable constants of shape (M, B). If not provided, it is initialized using the default normal distribution with mean 0 and standard deviation 1.
    """
    ...
from typing import Tuple
from enum import Enum
import torch

# 1. Re-declare the Enum for type hinting purposes
class Operator(Enum):
    NO_OP: int
    LEARNABLE_CONSTANT: int
    CONST_1: int
    CONST_2: int
    CONST_3: int
    CONST_4: int
    CONST_5: int
    PI: int
    E: int
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
    POW: int
    VAR_START_ID: int

# 2. Define the signature of the 'evaluate' function
def evaluate_backend(
    X: torch.Tensor, # (N, n_x)
    Ops: torch.Tensor, # (M, B)
    Ch: torch.Tensor, # (M, B, MAX_ARITY)
    C: torch.Tensor # (M, B)
) -> torch.Tensor:
    """
    Let M be the maximum number of operations in an expression, B the batch size of the expressions, N the number of datapoints, and n_x the number of variables in the expressions. 

    For an expression we unroll it using the postfix order into separate tensors Ops and Ch.

    X: Input tensor for the variables of shape (M, N, B).
    Ops: Tensor of operations specified by integers (see Operator Enum) of shape (M, B) for the expressions in postfix order. If some operations are not used, they can be set to NO_OP.
    Ch: Tensor of postfix order indices for the children of nodes in the expressions of shape (M, B, 2). If the operation has less than MAX_ARITY = 2 children, the unused children must be set to -1.
    C: Tensor of learnable constants of shape (M, B). For any (m, b) where Ops[m,b] is not a LEARNABLE_CONSTANT, the value of C[m,b] is ignored.
    
    """
    ...

class ProbabilisticContextFreeGrammar:
    device: torch.device

    def __init__(
        self,
        grammar: str,
        start_symbol: str,
        padded_maximum_length: int,
        n_variables: int,
        device: torch.device,
        max_tries: int = 64,
        tolerance: float = 1e-6,
        verbose: bool = False
    ) -> None:
        ...
    def sample(self, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
    def sample_string_expression(self, B: int) -> torch.Tensor:
        ...
    def to_string(self, expressions: torch.Tensor) -> list[str]:
        ...
    def parse_to_prefix(self, expressions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
    def parse_to_prefix_parent(self, expressions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
    def parse_to_postfix(self, expressions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
    def parse_to_postfix_parent(self, expressions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
    def postfix_to_infix(self, expressions: torch.Tensor, max_infix_len : int) -> torch.Tensor:
        ...
    def prefix_to_infix(self, expressions: torch.Tensor, max_infix_len : int) -> torch.Tensor:
        ...
    def get_symbol_id(self, symbol: str) -> int:
        ...

def get_arity(op: Operator) -> int:
    ...
from typing import Tuple
import torch

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
    def parse_to_postfix(self, expressions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
    def get_arities(self) -> torch.Tensor:
        ...

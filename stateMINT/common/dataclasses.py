from typing import Literal, Protocol

from flax import nnx
from omegaconf import DictConfig

Predictor = Literal["prevalence", "cases"]


class ModelFactory(Protocol):
    @classmethod
    def from_cfg(cls, cfg: DictConfig, input_size: int) -> nnx.Module: ...

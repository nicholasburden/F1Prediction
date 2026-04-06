from abc import ABC, abstractmethod
from typing import Literal

import torch
from pydantic import BaseModel


class Metric(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None: ...

    @abstractmethod
    def compute(self) -> float: ...

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]


class MAE(Metric):
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def name(self) -> str:
        return "mae"

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        self.total += torch.abs(pred - target).sum().item()
        self.count += pred.numel()

    def compute(self) -> float:
        return self.total / self.count


class MSE(Metric):
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def name(self) -> str:
        return "mse"

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        self.total += ((pred - target) ** 2).sum().item()
        self.count += pred.numel()

    def compute(self) -> float:
        return self.total / self.count


class WithinK(Metric):
    def __init__(self, k: float) -> None:
        self.k = k
        self.within = 0
        self.count = 0

    def reset(self) -> None:
        self.within = 0
        self.count = 0

    def name(self) -> str:
        return f"within_{self.k}"

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        self.within += (torch.abs(pred - target) <= self.k).sum().item()
        self.count += pred.numel()

    def compute(self) -> float:
        return self.within / self.count


MetricType = Literal["mae", "mse", "within_k"]

METRIC_REGISTRY: dict[MetricType, type[Metric]] = {
    "mae": MAE,
    "mse": MSE,
    "within_k": WithinK,
}


class MetricConfig(BaseModel):
    type: MetricType
    k: float | None = None

    def build(self) -> Metric:
        if self.type == "within_k":
            assert self.k
            return WithinK(k=self.k)
        return METRIC_REGISTRY[self.type]()

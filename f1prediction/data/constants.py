from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

type DataTable = Literal["laps", "weather", "results"]

DATA_DIR = Path("/Users/nick/code/F1Prediction/Data")


class Session(Enum):
    FP1 = "FP1"
    FP2 = "FP2"
    FP3 = "FP3"
    SQ = "SQ"
    Sprint = "Sprint"
    Q = "Q"
    R = "R"


@dataclass(frozen=True)
class EventSample:
    year: int
    event_id: int


@dataclass(frozen=True)
class EmbeddingConfig:
    dim: int
    vocab_size: int

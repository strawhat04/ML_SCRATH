from dataclasses import dataclass


@dataclass
class Node:
    id: int

    feature_index: int | None
    split_threshold: float | None

    sse: float
    n_samples: int

    prediction: float | None

    left: Node | None
    right: Node | None
    
@dataclass
class bestsplitnode:
    value : float
    sse : float



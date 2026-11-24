from dataclasses import dataclass


@dataclass
class Node:
    id: int

    feature_index: int | None
    split_threshold: float | None

    gini: float
    n_samples: int

    prediction: int | str | None

    left: Node | None
    right: Node | None
    
@dataclass
class bestsplitnode:
    value : float
    gini : float



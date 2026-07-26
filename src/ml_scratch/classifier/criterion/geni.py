import numpy as np


def gini(labels: np.ndarray) -> float:
    """
    This function takes numpy array of labels and returns impurity score of the given distribution
    Args:
        label : numpy array of labels
    """
    if labels is None:
        raise ValueError('Empty array recieved')
    if not isinstance(labels, np.ndarray):
        raise TypeError(f'Expected numpy array but received {type(labels)}')
    if labels.ndim!=1:
        raise TypeError(f'Expected 1D numpy array for label but recieved {labels.ndim}D array')

    _, count = np.unique(ar=labels, return_counts=True)
    p = count/labels.size

    gini_score = 1 - np.sum(p**2)

    return gini_score
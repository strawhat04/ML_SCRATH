import numpy as np


def squared_err(labels: np.ndarray) -> float:
    """
    This function takes numpy array of labels and returns sum of squared of the given distribution
    Args:
        label : numpy array of labels
    """
    if labels is None:
        raise ValueError('Empty array recieved')
    if not isinstance(labels, np.ndarray):
        raise TypeError(f'Expected numpy array but received {type(labels)}')
    if labels.ndim!=1:
        raise TypeError(f'Expected 1D numpy array for label but recieved {labels.ndim}D array')

    return np.square(labels - np.mean(labels)).sum()
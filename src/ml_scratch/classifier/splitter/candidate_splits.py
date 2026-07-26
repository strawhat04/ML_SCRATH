import numpy as np


def candidate_splits(feature : np.ndarray) -> np.ndarray:
    """
    Create all the posible split from the feature using the midpoint two adjacent values
    Args:
        feature : numpy array
    Return:
        Array of all the possible split criteria for the node
    
    Example:
        >>>> np.array([18,21,24,28,33])
            [19.5,
            22.5,
            26,
            30.5]
    """

    unique = np.unique(feature, sorted=True)

    return (unique[:-1] + unique[1:])/2
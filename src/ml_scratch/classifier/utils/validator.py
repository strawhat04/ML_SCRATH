import numpy as np


def validate_array(name: str, array:np.ndarray) -> None:

    if not isinstance(array, np.ndarray):
        raise TypeError(f'{name} : Expected numpy array but received {type(array)}')
    if name == 'label' and array.ndim!=1:
        raise ValueError(f'{name} : Expected 1D numpy array but recieved {array.ndim}D array')
    if array.size ==0:
        raise ValueError(f'{name} : Can not pass an empty numpy array')
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f'{name} :  Expected integer or float but received {array.dtype}')

    
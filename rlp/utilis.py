import numpy as np


def softmax(x):
    """ Caluate softmax distribution over float array x.
    """
    arr = np.exp(x)
    return arr / arr.sum()

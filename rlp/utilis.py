import numpy as np


def mean(lst):
    return sum(lst) / len(lst)


def argmax(scalarDict):
    """ get argmax from a dict where the key maps to a scalar
    """
    return max(scalarDict.items(), key=lambda x: x[1])[0]

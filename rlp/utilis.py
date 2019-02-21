import numpy as np


def softmax(x):
    """ Caluate softmax distribution over float array x.
    """
    arr = np.exp(x)
    return arr / arr.sum()


def argmax(container):
    """ argmax for container (list or dict).
    if there are multiple argmax in the result, return all of them.
    """
    if isinstance(container, dict):
        sorted_ = sorted(container.items(), key=lambda x: x[1], reverse=True)
        arg_max, max_ = sorted_[0]
    elif isinstance(container, list) or isinstance(container, tuple):
        sorted_ = sorted(enumerate(container), key=lambda x: x[1], reverse=True)
        arg_max, max_ = sorted_[0]
    else:
        raise TypeError('invliad container type %s' % type(container))

    ret = [arg_max]
    for key, val in sorted_[1:]:
        if val == max_:
            ret.append(key)
        else:
            break
    return ret

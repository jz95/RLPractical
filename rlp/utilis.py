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


def element_wise_product(dict1, dict2):
    assert dict1.keys() == dict2.keys()
    ret = {}
    for key in dict1:
        ret[key] = dict1[key] * dict2[key]
    return ret


def possion_prob(n, lambda_, truncate_threshold=None):
    if truncate_threshold is not None and n >= truncate_threshold:
        return 1 - sum([possion_prob(i, lambda_) for i in range(truncate_threshold)])

    if (n, lambda_) in possion_prob.memo:
        return possion_prob.memo[(n, lambda_)]

    if n in possion_prob.factorial:
        factorial = possion_prob.factorial[n]
    else:
        for i in sorted(possion_prob.factorial.keys(), reverse=True):
            if i < n:
                factorial = possion_prob.factorial[i]
                i += 1
                while i <= n:
                    factorial = factorial * i
                    possion_prob.factorial[i] = factorial
                    i += 1
                break
            else:
                continue
    ret = np.power(lambda_, n) * np.exp(- 1 * lambda_) / factorial
    possion_prob.memo[(n, lambda_)] = ret
    return ret

possion_prob.factorial = {0: 1, 1: 1}
possion_prob.memo = {}

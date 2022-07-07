import numpy as np
import copy

#####################################
#                                   #
#             UTILITIES             #
#                                   #
#####################################


def vcol(v):
    """
    Automatically reshapes an array into a row vector.
    """
    return v.reshape((v.size, 1))


def vrow(v):
    """
    Automatically reshapes an array into a column vector.
    """
    return v.reshape((1, v.size))


def get_m_ML(X):
    """
    Computes the empirical dataset mean.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    """
    return vcol(X.mean(1))


def get_C_ML(X, mu=0):
    """
    Computes the empirical dataset covariance.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param mu is the empirical mean of the dataset
    """
    return np.dot((X - mu), (X - mu).T) / X.shape[1]


def accuracy(predL, L):
    """
    Computes the accuracy of a prediction.
    notice the following relation holds: err_rate = 1 - accuracy
    :param predL is the array of predicted labels
    :param L is the array of actual labels
    """
    return np.sum(predL == L) / L.size


def err_rate(predL, L):
    """
    Computes the error rate of a prediction.
    notice the following relation holds: err_rate = 1 - accuracy
    :param predL is the array of predicted labels
    :param L is the array of actual labels
    """
    return np.sum(predL != L) / L.size


def params_combinations_R(params_items, pair_idx, res, curr_sol):
    """
    A recursive function to support params_combinations
    """
    if pair_idx == len(params_items):
        res.append(copy.deepcopy(curr_sol))
        return
    # get current pair (key, value)
    key, value = params_items[pair_idx]
    for v in value:
        # add v in the current solution
        curr_sol[key] = v
        params_combinations_R(params_items, pair_idx + 1, res, curr_sol)


def params_combinations(params):
    """
    Computes all parameters combinations.
    :param params is a dictionary of parameters we want to test
    """
    res = []
    params_combinations_R(list(params.items()), 0, res, {})
    return res

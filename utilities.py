import numpy as np

#####################################
#                                   #
#             UTILITIES             #
#                                   #
#####################################

"""
Automatically reshapes an array into a row vector.
"""
def vcol(v):
    return v.reshape((v.size, 1))


"""
Automatically reshapes an array into a column vector.
"""
def vrow(v):
    return v.reshape((1, v.size))

"""
Computes the empirical dataset mean.
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
"""
def get_m_ML(X):
    return vcol(X.mean(1))


"""
Computes the empirical dataset covariance.
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param mu is the empirical mean of the dataset
"""
def get_C_ML(X, mu=0):
    return np.dot((X - mu), (X - mu).T) / X.shape[1]


"""
Computes the accuracy of a prediction.
notice the following relation holds: err_rate = 1 - accuracy
:param predL is the array of predicted labels
:param L is the array of actual labels
"""
def accuracy(predL, L):
    return np.sum(predL == L) / L.size


"""
Computes the error rate of a prediction.
notice the following relation holds: err_rate = 1 - accuracy
:param predL is the array of predicted labels
:param L is the array of actual labels
"""
def err_rate(predL, L):
    return np.sum(predL != L) / L.size

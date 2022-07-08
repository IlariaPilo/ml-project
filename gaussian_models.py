import numpy as np
import utilities as u

#####################################
#                                   #
#          GAUSSIAN MODELS          #
#                                   #
#####################################

# -------------- log-density and log-likelihood -------------- #
def logpdf_GAU_ND(X, mu, C):
    """
    Computes the log-density of the dataset X.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param mu is the mean of each feature, having size (D,1) -> it is a column vector
    :param C is the covariance matrix, having size (D,D)
    """
    D = np.shape(C)[0]
    mu = u.vcol(mu)
    _, log_C = np.linalg.slogdet(C)
    const = -0.5*D*np.log(2*np.pi) - 0.5*log_C
    A = (X-mu).T
    B = np.dot(np.linalg.inv(C), (X-mu))
    return const - 0.5*np.sum(A.T*B, axis=0)


def loglikelihood(X, mu, C):
    """
    Computes the log-likelihood of the dataset X.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param mu is the mean of each feature, having size (D,1) -> it is a column vector
    :param C is the covariance matrix, having size (D,D)
    """
    return np.sum(logpdf_GAU_ND(X, mu, C))


# -------------- gaussian classifiers -------------- #
def mvg_fit(X, L):
    """
    Trains the model for a Multivariate Gaussian Classifier.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    """
    mu_s = np.empty((X.shape[0], 0))
    C_s = np.empty((0, X.shape[0]))
    # compute the number of classes
    K = np.max(L)+1
    # for each class
    for i in range(K):
        # select the samples for that class
        X_i = X[:, L == i]
        # compute the class means
        mu_s = np.append(mu_s, u.get_m_ML(X_i), axis=1)
        # compute the covariance matrix
        C_s = np.append(C_s, u.get_C_ML(X_i, u.vcol(mu_s[:,i])), axis=0)

    return mu_s, np.reshape(C_s, (K, X.shape[0], X.shape[0]))


def mvg_naive_bayes_fit(X, L):
    """
    Trains the model for a Naive Bayes Multivariate Gaussian Classifier.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    """
    # re-use the mgc
    mu_s, C_s = mvg_fit(X, L)
    # diagonalize C_s
    C_s = C_s*np.eye(X.shape[0])

    return mu_s, C_s


def mvg_tied_covariance_fit(X, L):
    """
    Trains the model for a Tied Covariance Multivariate Gaussian Classifier.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    """
    mu_s = np.empty((X.shape[0], 0))
    Sw = 0
    # compute the number of classes
    K = np.max(L)+1
    # for each class
    for i in range(K):
        # select the samples for that class
        X_i = X[:, L == i]
        # compute the class means
        mu_s = np.append(mu_s, u.get_m_ML(X_i), axis=1)
        # center the class data
        X_iC = X_i - u.vcol(mu_s[:,i])
        # the covariance matrix (already multiplied by Nc) is
        Sw_i = np.dot(X_iC, X_iC.T)
        # update Sw
        Sw += Sw_i

    return mu_s, Sw / X.shape[1]


def mvg_tied_naive_bayes_fit(X, L):
    """
    Trains the model for a Tied Naive Bayes Gaussian Classifier.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    """
    # re-use the tied cgc
    mu_s, C_s = mvg_tied_covariance_fit(X, L)
    # diagonalize C_s
    C_s = C_s*np.eye(X.shape[0])

    return mu_s, C_s


# -------------- binary gaussian estimators -------------- #
def mvg_log_predict(X, mu_s, C_s, pi=0.5):
    """
    Applies any logarithmic Gaussian classifier.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param mu_s is the set of means of the gaussian model
    :param C_s is the set of covariance matrices of the gaussian model
    :param pi is the prior probability of class 1
    """
    # compute threshold
    t = -np.log(pi/(1-pi))
    # get log-density for class 1
    ll1 = logpdf_GAU_ND(X, u.vcol(mu_s[:,1]), C_s[1] if len(C_s) == 2 else C_s)
    # get log-density for class 0
    ll0 = logpdf_GAU_ND(X, u.vcol(mu_s[:,0]), C_s[0] if len(C_s) == 2 else C_s)
    S = ll1-ll0
    predL = S > t
    return predL.astype(int), S

import numpy as np
import scipy.special
import utilities as u
import gaussian_models as gm

#####################################
#                                   #
#      GAUSSIAN MIXTURE MODELS      #
#                                   #
#####################################

def logpdf_GMM(X, gmm):
    """
    Computes the log-density of a GMM.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param gmm is the gmm model. It is an array of M tuples (w, mu, C)
    """
    N = X.shape[1]
    S = np.empty((0, N))

    for (w, mu, C) in gmm:
        row = gm.logpdf_GAU_ND(X, mu, C) + np.log(w)
        S = np.vstack((S, row))

    return scipy.special.logsumexp(S, axis=0), S


def em(X, start_gmm, version=None, threshold=10 ** (-6), isPrint=False, psi=None):
    """
    Estimates the parameters of a GMM maximizing the likelihood of the training set by using the EM algorithm.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param start_gmm is the initial gmm model. It is an array of M tuples (w, mu, C)
    :param version is the version of em: None - classic version, 'diag' - diagonal matrix,
    'tied' - tied covariance matrix, 'tied-diag' - tied and diagonal covariance matrix
    :param threshold is the hyperparameter (when is the conversion reached?)
    :param isPrint when set to True the function prints information about iterations
    :param psi should be used to constrain the eigenvalues of the covariance matrices, to avoid unbounded solutions
    """
    eps = 10**(-12)
    D, N = X.shape
    M = len(start_gmm)
    gmm = start_gmm
    llNew = None
    while True:
        llOld = llNew
        # compute the logpdf_GMM
        logpdf, S = logpdf_GMM(X, gmm)
        # compute llNew
        llNew = np.sum(logpdf) / N
        isPrint and print(llNew)
        if llOld is not None and llNew - llOld + eps < 0:
            raise Exception("The log likelihood decreased!")
        if llOld is not None and llNew - llOld < threshold:
            # the algorithm is done
            break
        # compute the responsibilities (this should be a MxN matrix)
        gamma = np.exp(S - logpdf)
        # compute the statistics
        Z = u.vcol(np.sum(gamma, axis=1))  # M size array
        F = np.dot(X, gamma.T).T  # MxD matrix
        gamma_ = np.reshape(gamma, (M, 1, N))
        X_ = np.reshape(X, (1, D, N))
        S = np.dot(X_ * gamma_, X.T)  # MxDxD matrix
        # compute the new parameters
        mu = F / Z  # MxD matrix
        w = Z / np.sum(Z)  # M size array
        C = S / np.reshape(Z, (M, 1, 1)) - np.reshape(mu, (M, D, 1)) * np.reshape(mu, (M, 1, D))
        if version=='tied' or version=='tied-diag':
            # compute the covariance matrix
            C = np.sum(np.reshape(Z, (M, 1, 1)) * C, axis=0) / N
            C = np.repeat(np.reshape(C, (1, C.shape[0], C.shape[1])), M, axis=0)
        if version == 'diagonal' or version=='tied-diag':
            # make it diagonal
            C = C * np.eye(C.shape[1])
        if psi is not None:
            U, s, _ = np.linalg.svd(C)
            s[s < psi] = psi
            C = np.matmul(U, np.reshape(s, (M, D, 1)) * np.transpose(U, (0, 2, 1)))
        # update the gmm
        gmm = list(zip(w, mu, C))
    return gmm


def diag_em(X, start_gmm, threshold=10 ** (-6), isPrint=False, psi=None):
    """
    A wrapper of em() with version='diag'.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param start_gmm is the initial gmm model. It is an array of M tuples (w, mu, C)
    :param threshold is the hyperparameter (when is the conversion reached?)
    :param isPrint when set to True the function prints information about iterations
    :param psi should be used to constrain the eigenvalues of the covariance matrices, to avoid unbounded solutions
    """
    return em(X, start_gmm, 'diag', threshold, isPrint, psi)


def tied_em(X, start_gmm, threshold=10 ** (-6), isPrint=False, psi=None):
    """
    A wrapper of em() with version='tied'.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param start_gmm is the initial gmm model. It is an array of M tuples (w, mu, C)
    :param threshold is the hyperparameter (when is the conversion reached?)
    :param isPrint when set to True the function prints information about iterations
    :param psi should be used to constrain the eigenvalues of the covariance matrices, to avoid unbounded solutions
    """
    return em(X, start_gmm, 'tied', threshold, isPrint, psi)


def tied_diag_em(X, start_gmm, threshold=10 ** (-6), isPrint=False, psi=None):
    """
    A wrapper of em() with version='tied-diag'.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param start_gmm is the initial gmm model. It is an array of M tuples (w, mu, C)
    :param threshold is the hyperparameter (when is the conversion reached?)
    :param isPrint when set to True the function prints information about iterations
    :param psi should be used to constrain the eigenvalues of the covariance matrices, to avoid unbounded solutions
    """
    return em(X, start_gmm, 'tied-diag', threshold, isPrint, psi)


def split_LBG(gmm_G, alpha=0.1):
    """
    Constructs a GMM with 2*G components starting from a GMM with G components.
    :param gmm_G is the initial gmm
    :alpha hyperparameter of the LGB algorithm
    """
    gmm_2G = []
    for (w, mu, C) in gmm_G:
        mu = u.vcol(mu)
        w_2G = w / 2
        U, s, _ = np.linalg.svd(C)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        gmm_2G.append((w_2G, mu + d, C))
        gmm_2G.append((w_2G, mu - d, C))
    return gmm_2G


def gmm_LBG(X, G, em_algorithm=em, alpha=0.1, threshold=10 ** (-6), isPrint=False, psi=None):
    """
    Generates a GMM of exponentially-increasing size using the LBG algorithm fitting the dataset X.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param G is the final number of components we want to gain
    :param em_algorithm is the em algorithm we want to use (em, diag_em, tied_em)
    :alpha hyperparameter of the LGB algorithm
    :param threshold is the hyperparameter (when is the conversion reached?)
    :param isPrint when set to True the function prints information about iterations
    :param psi should be used to constrain the eigenvalues of the covariance matrices, to avoid unbounded solutions
    """
    mu = u.get_m_ML(X)
    C = u.get_C_ML(X, mu)
    if em_algorithm==diag_em or em_algorithm==tied_diag_em:
        C = C * np.eye(C.shape[0])
    if psi is not None:
        U, s, _ = np.linalg.svd(C)
        s[s < psi] = psi
        C = np.dot(U, u.vcol(s) * U.T)
    gmm = [(1.0, mu, C)]
    M = 1
    while M < G:
        # split gmm
        gmm = split_LBG(gmm, alpha)
        # call EM
        gmm = em_algorithm(X, gmm, threshold=threshold, isPrint=isPrint, psi=psi)
        M *= 2
    return gmm


def gmm_fit(X, L, G, em_algorithm=em, alpha=0.1, threshold=10 ** (-6), isPrint=False, psi=0.01):
    """
    Trains a GMM of exponentially-increasing size.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    :param G is the final number of components we want to gain
    :param em_algorithm is the em algorithm we want to use (em, diag_em, tied_em)
    :param alpha hyperparameter of the LGB algorithm
    :param threshold is the hyperparameter (when is the conversion reached?)
    :param isPrint when set to True the function prints information about iterations
    :param psi should be used to constrain the eigenvalues of the covariance matrices, to avoid unbounded solutions
    """
    gmm_list = []
    # compute number of classes
    K = np.max(L) + 1
    # for each class
    for i in range(K):
        # select the samples for that class
        L_i = X[:, L == i]
        # compute the gmm
        gmm = gmm_LBG(L_i, G, em_algorithm, alpha, threshold, isPrint, psi)
        gmm_list.append(gmm)

    return gmm_list


def gmm_predict(X, gmm_list, pi=0.5):
    """
    Classifies a dataset applying the given GMM model.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param gmm_list is the GMM model
    :param pi is the prior associated to class 1
    """
    # compute threshold
    t = -np.log(pi / (1 - pi))
    # get log-density for class 1
    ll1, _ = logpdf_GMM(X, gmm_list[1])
    # get log-density for class 0
    ll0, _ = logpdf_GMM(X, gmm_list[0])
    S = ll1-ll0
    predL = S > t
    return predL.astype(int), S

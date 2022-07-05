import numpy as np
import scipy.special
import utilities as u

#####################################
#                                   #
#          GAUSSIAN MODELS          #
#                                   #
#####################################

# -------------- log-density and log-likelihood -------------- #
"""
Computes the log-density of the dataset X. 
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param mu is the mean of each feature, having size (D,1) -> it is a column vector
:param C is the covariance matrix, having size (D,D)
"""
def logpdf_GAU_ND(X, mu, C):
    D = np.shape(C)[0]
    _, log_C = np.linalg.slogdet(C)
    const = -0.5*D*np.log(2*np.pi) - 0.5*log_C
    A = (X-mu).T
    B = np.dot(np.linalg.inv(C), (X-mu))
    return const - 0.5*np.sum(A.T*B, axis=0)


"""
Computes the log-likelihood of the dataset X. 
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param mu is the mean of each feature, having size (D,1) -> it is a column vector
:param C is the covariance matrix, having size (D,D)
"""
def loglikelihood(X, mu, C):
    return np.sum(logpdf_GAU_ND(X, mu, C))


# -------------- gaussian classifiers -------------- #
"""
Trains the model for a Multivariate Gaussian Classifier.
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param L is the array of knows labels for such samples
"""
def mvg_fit(X, L):
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


"""
Trains the model for a Naive Bayes Multivariate Gaussian Classifier.
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param L is the array of knows labels for such samples
"""
def mvg_naive_bayes_fit(X, L):
    # re-use the mgc
    mu_s, C_s = mvg_fit(X, L)
    # diagonalize C_s
    C_s = C_s*np.eye(X.shape[0])

    return mu_s, C_s


"""
Trains the model for a Tied Covariance Multivariate Gaussian Classifier.
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param L is the array of knows labels for such samples
"""
def mvg_tied_covariance_fit(X, L):
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


"""
Trains the model for a Tied Naive Bayes Gaussian Classifier.
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param L is the array of knows labels for such samples
"""
def mvg_tied_naive_bayes_fit(X, L):
    # re-use the tied cgc
    mu_s, C_s = mvg_tied_covariance_fit(X, L)
    # diagonalize C_s
    C_s = C_s*np.eye(X.shape[0])

    return mu_s, C_s


# -------------- gaussian estimators -------------- #
"""
Applies any Gaussian classifier.
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param mu_s is the set of means of the gaussian model
:param C_s is the set of covariance matrices of the gaussian model
:param K is the number of classes (default 2: binary problem)
"""
def mvg_predict(X, mu_s, C_s, K=2):
    SJoint = np.empty((0, X.shape[1]))
    # for each class
    for i in range(K):
        # compute the likelihoods for the class i and save them in the score matrix
        SJoint = np.append(SJoint, u.vrow(np.exp(logpdf_GAU_ND(X, u.vcol(mu_s[:, i]), C_s[i] if len(C_s.shape) == 3 else C_s))) / 3, axis=0)

    # compute the marginal
    SMarginal = u.vrow(SJoint.sum(0))
    # now, compute the posterior probabilities
    SPost = SJoint/SMarginal

    # predict the label as the class associated with the highest probability
    predictedL = np.argmax(SPost, axis=0)

    return predictedL


"""
Applies any logarithmic Gaussian classifier.
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param mu_s is the set of means of the gaussian model
:param C_s is the set of covariance matrices of the gaussian model
:param K is the number of classes (default 2: binary problem)
"""
def mvg_log_predict(X, mu_s, C_s, K=2):
    logSJoint = np.empty((0, X.shape[1]))

    # for each class
    for i in range(2):
        # compute the likelihoods for the class i and save them in the score matrix
        logSJoint = np.append(logSJoint, u.vrow(logpdf_GAU_ND(X, u.vcol(mu_s[:, i]), C_s[i] if len(C_s.shape) == 3 else C_s)) - np.log(3), axis=0)

    # compute the marginal
    logSMarginal = u.vrow(scipy.special.logsumexp(logSJoint, axis=0))

    # now, compute the posterior probabilities
    logSPost = logSJoint - logSMarginal

    SPost = np.exp(logSPost)
    # predict the label as the class associated with the highest probability
    predictedL = np.argmax(SPost, axis=0)

    return predictedL


# -------------- k-fold -------------- #
# TODO --- either expand or delete these, if not useful
"""
Applies a basic k-fold approach.
:param k is the number of folds
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param L is the array of knows labels for such samples
:param classifier is the type of gaussian classifier we want to use (eg, mvg_tied_naive_bayes_fit)
:param evaluator is the type of evaluator we want to use (eg, mvg_log_predict) 
"""
def k_fold(k, X, L, classifier, evaluator):
    N = X.shape[1]  # N is the number of samples
    numberTestSamples = N//k
    predL = np.empty(N)
    for i in range(0, k):
        # compute train and test set
        testMask = [False]*N
        testMask[i*numberTestSamples:(i+1)*numberTestSamples if i!=(k-1) else N] = [True]*(numberTestSamples + (N % k if i==(k-1) else 0))
        testX = X[:, testMask]
        trainX = X[:, np.logical_not(testMask)]
        trainL = L[np.logical_not(testMask)]
        # create classifier
        mu_s, C_s = classifier(trainX, trainL)
        predL[i*numberTestSamples] = evaluator(testX, mu_s, C_s)
    # compute accuracy and error and return them
    return u.accuracy(predL, L), u.err_rate(predL, L)


"""
Simply calls k_fold with k=N, where N is the number of samples in the dataset.
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param L is the array of knows labels for such samples
:param classifier is the type of gaussian classifier we want to use (eg, mvg_tied_naive_bayes_fit)
:param evaluator is the type of evaluator we want to use (eg, mvg_log_predict) 
"""
def leave_one_out(X, L, classifier, evaluator):
    N = X.shape[1]  # N is the number of samples
    return k_fold(N, X, L, classifier, evaluator)

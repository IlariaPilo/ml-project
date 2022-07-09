import numpy as np
import scipy.optimize as opt
import utilities as u

#######################################
#                                     #
#         LOGISTIC REGRESSION         #
#                                     #
#######################################

# -------------- binary logistic regression -------------- #
def binary_lr_fit(X, L, l, pi):
    """
    Trains the model for a Binary Logistic Regression classifier.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    :param l is the hyperparameter lambda
    :param pi is the prior probability of class 1
    """
    # build the binary logistic regression classifier
    def logreg_obj_wrap(DTR, LTR, l):
        Z = LTR * 2.0 - 1.0
        M = DTR.shape[0]
        def logreg_obj(v):
            w = u.vcol((v[0:M]))
            b = v[-1]
            S = np.dot(w.T, DTR) + b
            cxe = np.logaddexp(0, -S*Z).mean()
            return cxe + 0.5*l * np.linalg.norm(w)**2
        return logreg_obj
    logreg_obj = logreg_obj_wrap(X, L, l)
    _v, _J, _d = opt.fmin_l_bfgs_b(logreg_obj, np.zeros(X.shape[0]+1), approx_grad=True)
    return _v[0:X.shape[0]], _v[-1], _J


def binary_lr_predict(X, w, b):
    """
    Applies a Binary Logistic Regression classifier.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param w is the first parameter of the LR classifier
    :param b is the second parameter of the LR classifier
    """
    # fix w dimensions
    w = u.vcol(w)
    # compute scores (posterior log-likelihood ratio)
    STE = np.dot(w.T, X) + b
    LP = STE > 0
    return LP, np.reshape(STE, STE.size)


# TODO --- check weird results
def quadratic_expansion(X):
    """
    Computes the quadratic expansion of feature vectors.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    """
    D, N = X.shape
    # reshape X
    X_ = np.reshape(X.T, (N,D,1))
    X_T = np.transpose(X_, (0,2,1))
    XX_T = np.matmul(X_,X_T)            # NxDxD matrix
    vecXX_T = np.reshape(np.reshape(XX_T.T, N*D*D), (D*D,N))
    return np.vstack((vecXX_T, X))

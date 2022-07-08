import numpy as np
import scipy.optimize as opt
import utilities as u

#######################################
#                                     #
#         LOGISTIC REGRESSION         #
#                                     #
#######################################

# -------------- binary logistic regression -------------- #
def binary_lr_fit(X, L, l):
    """
    Trains the model for a Binary Logistic Regression classifier.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    :param l is the hyperparameter lambda
    """
    # build the binary logistic regression classifier
    def logreg_obj_wrap(X, L, l):
        def logreg_obj(arg):
            # unpack the arguments
            w, b = u.vcol(arg[0:-1]), arg[-1]
            # norm term
            norm = 0.5 * l * (np.linalg.norm(w) ** 2)
            # compute z
            z = 2 * L - 1
            # summation term
            summation = np.sum(np.logaddexp(0, -z * (np.sum(w * X, axis=0) + b)))
            n = X.shape[1]
            return float(norm + summation / n)
        return logreg_obj

    logreg_obj = logreg_obj_wrap(X, L, l)
    x, f, d = opt.fmin_l_bfgs_b(func=logreg_obj, x0=np.zeros(X.shape[0] + 1), approx_grad=True, maxiter=30000, iprint=0)
    return x[0:-1], x[-1], f    # that is, w, b and J(w,b)


def binary_lr_predict(X, w, b, pi=0.5):
    """
    Applies a Binary Logistic Regression classifier.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param w is the first parameter of the LR classifier
    :param b is the second parameter of the LR classifier
    :param pi is the prior probability of class 1
    """
    # fix w dimensions
    w = u.vcol(w)
    # compute scores (posterior log-likelihood ratio)
    S = np.zeros((2, X.shape[1]))
    S[1,:] = np.sum(w * X, axis=0) + b
    # assign labels
    predL = np.argmax(S, axis=0)
    return predL, S[1,:]-np.log(pi/(1-pi))


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

import utilities as u
import numpy as np
import scipy.optimize as opt

####################################
#                                  #
#      SUPPORT VECTOR MACHINE      #
#                                  #
####################################

def svm_obj_wrap(H_):
    """
    Define the function to be minimized and its gradient.
    """
    def svm_obj(arg):
        # unpack the argument
        alpha = u.vcol(arg)
        # compute the result
        J = 0.5 * np.linalg.multi_dot([alpha.T, H_, alpha]) - np.sum(alpha)
        return float(J)

    def svm_obj_prime(arg):
        # unpack the argument
        alpha = u.vcol(arg)
        return np.dot(H_, alpha) - 1

    return svm_obj, svm_obj_prime


# -------------- linear support vector machine -------------- #
# TODO ------------this does not give the exact prof's results-----------------------------------
def linear_svm_fit(X, L, C, K):
    """
    Builds the linear SVM solution through the dual formulation.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    :param C is a hyperparameter of SVM
    :param K is a hyperparameter of SVM (to reduce the effect of regularizing b)
    """
    # build the extended matrix of training data
    X_ = np.vstack((X, u.vrow(np.array([K] * X.shape[1]))))
    # compute z
    z = 2*L - 1
    # compute matrix H_ (NO LOOPS)
    G_ = np.dot(X_.T, X_)
    H_ = u.vcol(z) * G_ * u.vrow(z)

    svm_obj, svm_obj_prime = svm_obj_wrap(H_)
    # prepare the bounds
    bounds = [(0,C)] * X.shape[1]
    # call the optimizer
    x, f, d = opt.fmin_l_bfgs_b(func=svm_obj, x0=np.zeros(X.shape[1]), fprime=svm_obj_prime, bounds=bounds,
                                factr=1.0, iprint=False)

    # we have considered the negative in order to minimize instead of maximize
    J = -f
    # recover the primal solution
    w_ = np.sum(x*z*X_, axis=1)

    return w_, J


def linear_svm_predict(X, w_, K):
    """
    Classifies a dataset applying the linear SVM.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param w_ is the linear SVM model
    :param K is a hyperparameter of SVM (to reduce the effect of regularizing b)
    """
    w_ = u.vcol(w_)
    # build the extended matrix of training data
    X_ = np.vstack([X, u.vrow(np.array([K] * X.shape[1]))])
    # compute scores
    S = np.zeros((2, X.shape[1]))
    S[1,:] = np.sum(w_*X_, axis=0)
    # assign labels
    predL = np.argmax(S, axis=0)
    return predL


def primal_solution(X, L, w_, C, K):
    """
    Computes the SVM primal solution starting from the dual one, to check for its correctness.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    :param w_ is the linear SVM model
    :param C is a hyperparameter of SVM
    :param K is a hyperparameter of SVM (to reduce the effect of regularizing b)
    """
    w_ = u.vcol(w_)
    # build the extended matrix of training data
    X_ = np.vstack([X, u.vrow(np.array([K] * X.shape[1]))])
    z = 2*L - 1
    # norm term
    norm = 0.5 * (np.linalg.norm(w_) ** 2)
    # summation term
    summation = C * np.sum(np.max(np.vstack([np.zeros(L.shape), 1 - z * np.sum(w_*X_, axis=0)]), axis=0))

    return norm + summation


# -------------- linear support vector machine -------------- #
def poly_kernel(d, c=0):
    """
    A polynomial kernel.
    :param d is the degree of the polynomial
    :param c is the kernel hyperparameter
    """
    def kernel(x1, x2):
        return (np.dot(x1.T,x2)+c)**d
    return kernel


def radial_kernel(gamma):
    """
    A Radial Basis Function kernel.
    :param gamma is the kernel hyperparameter
    """
    def kernel(x1, x2):
        return np.exp(-gamma*(np.linalg.norm(x1-x2) ** 2))
    return kernel


def kernel_svm_fit(X, L, C, K, kernel):
    """
    Builds the kernel SVM solution.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    :param C is a hyperparameter of SVM
    :param K is a hyperparameter of SVM (to reduce the effect of regularizing b)
    :param kernel is the kernel function we want to apply
    """
    # build the dual SVM solution through the dual formulation
    # compute z
    z = 2*L - 1
    # compute matrix H_
    H_ = np.empty((X.shape[1], X.shape[1]))
    for i in range(0, X.shape[1]):
        x_i = u.vcol(X[:, i])
        for j in range(i, X.shape[1]):
            x_j = u.vcol(X[:, j])
            H_[i,j] = H_[j,i] = z[i]*z[j]*(kernel(x_i,x_j)+(K**0.5))

    svm_obj, svm_obj_prime = svm_obj_wrap(H_)
    # prepare the bounds
    bounds = [(0,C)] * X.shape[1]
    # call the optimizer
    x, f, d = opt.fmin_l_bfgs_b(func=svm_obj, x0=np.zeros(X.shape[1]), fprime=svm_obj_prime, bounds=bounds,
                                factr=1.0, iprint=False)

    # we have considered the negative in order to minimize instead of maximize
    J = -f
    return x, J


def kernel_svm_predict(X_test, alpha, X_train, L_train, K, kernel):
    """
    Classifies a dataset applying the kernel SVM.
    :param X_test is the TEST dataset matrix
    :param alpha is the kernel SVM model
    :param X_train is the TRAIN dataset matrix
    :param L_train is the array of knows labels for TRAIN samples
    :param K is a hyperparameter of SVM (to reduce the effect of regularizing b)
    :param kernel is the kernel function we want to apply
    """
    z = 2 * L_train - 1
    predL = np.empty(X_test.shape[1])
    for t in range(0, X_test.shape[1]):
        x_t = u.vcol(X_test[:, t])
        score = 0
        for i in range(0, X_train.shape[1]):
            x_i = u.vcol(X_train[:, i])
            score += alpha[i]*z[i]*(kernel(x_i,x_t)+(K**0.5))
        predL[t] = 0 if score<0 else 1
    return predL

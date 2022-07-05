import numpy as np
import scipy
import utilities as u

#####################################
#                                   #
#           PREPROCESSING           #
#                                   #
#####################################

# -------------- PCA -------------- #
"""
Computes the projection matrix for PCA.
Once computed the projection matrix, we can appy it to a matrix of samples by performing XP = np.dot(P.T, X)
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param m is the dimensionality of the projection matrix we want to obtain
"""
def pca(X, m):
    # first, compute the dataset mean
    mu = u.get_m_ML(X)
    # now, compute the covariance matrix
    C = u.get_C_ML(X, mu)
    # compute eigenvectors and eigenvalues (sorted in descending order) by means of svd
    U, _, _ = np.linalg.svd(C)
    # the projection matrix is
    P = U[:, 0:m]
    return P


# -------------- LDA -------------- #
"""
Returns the within class covariance matrix.
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param L is the array of knows labels for such samples  
"""
def within_class(X, L):
    # initialize Sw
    Sw = 0
    # compute the number of classes
    K = np.max(L)+1
    # for each class
    for i in range(K):
        # select the samples for that class
        X_i = X[:, L == i]
        # compute the class mean
        mu_i = u.get_m_ML(X_i)
        # get the covariance matrix and multiply it by Nc (the number of elements in class c=i)
        C_i = u.get_C_ML(X_i, mu_i) * X_i.shape[1]
        # update Sw
        Sw += C_i

    # divide Sw by N
    return Sw / X.shape[1]


"""
Returns the between class covariance matrix.
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param L is the array of knows labels for such samples  
"""
def betweenClass(X, L):
    # initialize Sb
    Sb = 0
    # compute the dataset mean
    mu = u.get_m_ML(X)
    # compute the number of classes
    K = np.max(L) + 1
    # for each class
    for i in range(K):
        # select the samples for that class
        X_i = X[:, L == i]
        # compute the class means difference
        mu_diff_i = u.get_m_ML(X_i) - mu
        # for the i-th class, the matrix is
        Sb_i = np.dot(mu_diff_i, mu_diff_i.T) * X_i.shape[1]
        # update Sb
        Sb += Sb_i

    # divide Sb by N
    return Sb / X.shape[1]


"""
Computes the transformation matrix for LDA by solving the generalized eigenvalue problem.
Once computed the transformation matrix, we can appy it to a matrix of samples by performing XP = np.dot(W.T, X)
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param L is the array of knows labels for such samples  
:param m is the dimensionality of the transformation matrix we want to obtain (AT MOST K-1, where K is the 
number of classes)
:param orth_basis should be set to True whenever we want W to be an orthogonal basis (usually it is not necessary)
"""
def lda_generalized(X, L, m, orth_basis=False):
    # first, compute the within class covariance matrix
    Sw = within_class(X, L)
    # then, compute the between class covariance matrix
    Sb = betweenClass(X, L)
    # solve the generalized eigenvalue problem
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    # find a basis U for the subspace spanned by W (OPTIONAL)
    if orth_basis:
        Uw, _, _ = np.linalg.svd(W)
        W = Uw[:, 0:m]

    return W


"""
Computes the transformation matrix for LDA by solving the eigenvalue problem by joint diagonalization of Sb and Sw.
Once computed the transformation matrix, we can appy it to a matrix of samples by performing XP = np.dot(W.T, X)
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param L is the array of knows labels for such samples  
:param m is the dimensionality of the transformation matrix we want to obtain (AT MOST K-1, where K is the 
number of classes)
"""
def lda_joint_diag(X, L, m):
    # first, compute the within class covariance matrix
    Sw = within_class(X, L)
    # then, compute the between class covariance matrix
    Sb = betweenClass(X, L)

    # compute the SVD of Sw
    Uw, s, _ = np.linalg.svd(Sw)
    # compute P1
    P1 = np.dot(Uw * u.vrow(1.0/(s**0.5)), Uw.T)
    # the transformed between class covariance is
    Sbt = np.dot(np.dot(P1, Sb), P1.T)
    # compute the matrix of eigenvectors of Sbt correspondent to the m-highest eigenvalues
    Ub, _, _ = np.linalg.svd(Sbt)
    # the projection matrix is
    P2 = Ub[:, 0:m]

    return np.dot(P1.T, P2)
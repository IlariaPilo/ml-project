import numpy as np
import scipy
import utilities as u
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

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
def between_class(X, L):
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
    Sb = between_class(X, L)
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
    Sb = between_class(X, L)

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


# -------------- features gaussianization -------------- #
"""
Computes the features gaussianization of the given dataset.
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
"""
def feats_gaussianization(X):
    N = X.shape[1]
    # compute ranks of features
    order = X.argsort(axis=1)
    ranks = order.argsort(axis=1)
    # add 1 and divide by N+2
    ranks = (ranks+1)/(N+2)
    # compute the inverse of the cumulative distribution function of the standard normal distribution
    return norm.ppf(ranks)


# -------------- plot functions -------------- #
"""
Plots two histograms for each feature: one (blue) representing values for male samples, and another (red) representing
values for female samples.
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param L is the array of knows labels for such samples
"""
def feats_hist(X, L):
    X0 = X[:, L == 0]   # keep only columns related to label 0 (male)
    X1 = X[:, L == 1]   # keep only columns related to label 1 (female)

    plt.figure()
    plt.suptitle("Features histograms")

    for a in range(12):  # for each attribute
        plt.subplot(4,3,a+1)
        plt.hist(X0[a, :], bins=20, density=True, alpha=0.4, color='blue')  # Male
        plt.hist(X1[a, :], bins=20, density=True, alpha=0.4, color='red')   # Female
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure

    plt.show()


"""
Prints three heatmaps to detect correlation between attributes.
:param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
:param L is the array of knows labels for such samples
"""
def feats_correlation(X, L):
    X0 = X[:, L == 0]  # keep only columns related to label 0 (male)
    X1 = X[:, L == 1]  # keep only columns related to label 1 (female)

    # compute the Pearson Correlation Coefficient
    corrX = np.corrcoef(X)
    corrX0 = np.corrcoef(X0)
    corrX1 = np.corrcoef(X1)

    # print heatmaps
    plt.figure()
    plt.suptitle('Features correlation')

    plt.subplot(1, 3, 1)
    plt.title('Full dataset')
    sns.heatmap(corrX, cmap='Greys', cbar=False, square=True)

    plt.subplot(1, 3, 2)
    plt.title('Male samples')
    sns.heatmap(corrX0, cmap='Blues', cbar=False, square=True)

    plt.subplot(1, 3, 3)
    plt.title('Female samples')
    sns.heatmap(corrX1, cmap='Reds', cbar=False, square=True)

    plt.show()
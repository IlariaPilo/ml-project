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
def pca(X, m):
    """
    Computes the projection matrix for PCA.
    Once computed the projection matrix, we can appy it to a matrix of samples by performing XP = np.dot(P.T, X)
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param m is the dimensionality of the projection matrix we want to obtain
    """
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
def within_class(X, L):
    """
    Returns the within class covariance matrix.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    """
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


def between_class(X, L):
    """
    Returns the between class covariance matrix.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    """
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


def lda_generalized(X, L, m, orth_basis=False):
    """
    Computes the transformation matrix for LDA by solving the generalized eigenvalue problem.
    Once computed the transformation matrix, we can appy it to a matrix of samples by performing XP = np.dot(W.T, X)
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    :param m is the dimensionality of the transformation matrix we want to obtain (AT MOST K-1, where K is the
    number of classes)
    :param orth_basis should be set to True whenever we want W to be an orthogonal basis (usually it is not necessary)
    """
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


def lda_joint_diag(X, L, m):
    """
    Computes the transformation matrix for LDA by solving the eigenvalue problem by joint diagonalization of Sb and Sw.
    Once computed the transformation matrix, we can appy it to a matrix of samples by performing XP = np.dot(W.T, X)
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    :param m is the dimensionality of the transformation matrix we want to obtain (AT MOST K-1, where K is the
    number of classes)
    """
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
def feats_gaussianization(trainX, testX):
    """
    Computes the features gaussianization of the given dataset.
    :param X is the TRAIN dataset matrix
    :param X is the TEST dataset matrix
    """
    D, N = trainX.shape
    # compute ranks of train features
    order = trainX.argsort(axis=1)
    ranks = order.argsort(axis=1)
    # compute the inverse of the cumulative distribution function of the standard normal distribution
    gauss_train = norm.ppf((ranks+1)/(N+2))
    # now, compute ranks of test features
    ranks = np.empty((D,0), dtype=int)
    for i in range(testX.shape[1]):
        sample = u.vcol(testX[:,i])
        mask = trainX<sample
        ranks = np.hstack((ranks, u.vcol(np.sum(mask, axis=1))))
    gauss_test = norm.ppf((ranks + 1) / (N + 2))
    return gauss_train, gauss_test


# -------------- plot functions -------------- #
def feats_hist(X, L):
    """
    Plots two histograms for each feature: one (blue) representing values for male samples, and another (red) representing
    values for female samples.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    """
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


def feats_correlation(X, L):
    """
    Prints three heatmaps to detect correlation between attributes.
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    """
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


# -------------- split dataset -------------- #
def split_dataset(X, L, train_perc, seed=0):
    """
    Splits the dataset into a training part and an evaluation part.
    :param X is the dataset
    :param L is the array of labels
    :param train_perc is the percentage of samples we want to keep for training
    :param seed is used to set up the random generator to obtain consistent results
    """
    nTrain = int(X.shape[1] * train_perc / 100.0)
    np.random.seed(seed)
    idx = np.random.permutation(X.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = X[:, idxTrain]
    DTE = X[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def k_fold(k, X, L, seed=0):
    """
    Applies a basic k-fold approach.
    :param k is the number of folds
    :param X is the dataset matrix having size (D,N) -> a row for each feature, a column for each sample
    :param L is the array of knows labels for such samples
    :param seed is used to set up the random generator to obtain consistent results
    """
    N = X.shape[1]  # N is the number of samples
    np.random.seed(seed)
    idx = np.random.permutation(N)
    fold_size = N // k
    foldsX = []
    foldsL = []
    # fill the first k-1 folds
    for i in range(0, k-1):
        idxFold = idx[i*fold_size:(i+1)*fold_size]
        foldsX.append(X[:, idxFold])
        foldsL.append(L[idxFold])
    # fill the last fold
    idxFold = idx[(k-1)*fold_size:]
    foldsX.append(X[:, idxFold])
    foldsL.append(L[idxFold])
    return foldsX, foldsL


def k_fold_split_folds(i, foldsX, foldsL):
    """
    Splits the folds in training and validation folds
    :param i is the iteration number
    :param foldsX is the list of samples folds
    :param foldsL is the list of labels folds
    """
    XTE = foldsX[i]
    LTE = foldsL[i]
    slice1_X = foldsX[:i]
    slice2_X = foldsX[i+1:]
    if len(slice1_X) != 0 and len(slice2_X) != 0:
        XTR = np.hstack((np.hstack(slice1_X), np.hstack(slice2_X)))
        LTR = np.concatenate((np.concatenate(foldsL[:i]), np.concatenate(foldsL[i + 1:])))
    elif len(slice2_X) == 0:
        XTR = np.hstack(slice1_X)
        LTR = np.concatenate(foldsL[:i])
    else:
        XTR = np.hstack(slice2_X)
        LTR = np.concatenate(foldsL[i+1:])
    return (XTR, LTR), (XTE, LTE)

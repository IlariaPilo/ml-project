import sys

import numpy as np
import matplotlib.pyplot as plt

################################################
#                                              #
#    CONFUSION MATRICES & OPTIMAL DECISIONS    #
#                                              #
################################################

# -------------- confusion matrices utilities -------------- #
def confusion(trueL, predL):
    """
    Computes the confusion matrix for the provided sets of labels.
    :param trueL stores the actual labels of the dataset
    :param predL stores the predicted labels for sch dataset
    """
    # compute K = number of classes
    K = np.max(trueL)+1
    # initialize the confusion matrix
    conf = np.zeros((K,K))
    for (p, t) in zip(predL, trueL):
        conf[p, t] += 1
    return conf


def get_FNR(confusion):
    """
    Computes the False Negative Rate of a given confusion matrix.
    """
    return confusion[0,1]/(confusion[0,1]+confusion[1,1])


def get_FPR(confusion):
    """
    Computes the False Positive Rate of a given confusion matrix.
    """
    return confusion[1,0]/(confusion[0,0]+confusion[1,0])


# -------------- Bayes Binary Evaluation -------------- #
def binary_optimal_decision(S, pi_1, C_fn, C_fp):
    """
    Computes optimal bayes decision from a binary loglikelihood ratio.
    :param S is the binary log-likelihood ratio
    :param pi_1 is the prior probability of class 1
    :param C_fn is the error cost of false negatives
    :param C_fp is the error cost of false positives
    """
    # compute threshold
    t = - np.log((pi_1*C_fn)/(1-pi_1)/C_fp)
    # generate predL array
    predL = np.zeros(S.size, dtype=int)
    # fill it
    predL[S > t] = 1
    return predL


def dcf(confusion, pi_1, C_fn, C_fp, normalized=False):
    """
    Computes the Bayes (normalized) risk.
    :param confusion is the confusion matrix
    :param pi_1 is the prior probability of class 1
    :param C_fn is the error cost of false negatives
    :param C_fp is the error cost of false positives
    :param normalized when set to true, the normalized bayes risk is computed
    """
    FNR = get_FNR(confusion)
    FPR = get_FPR(confusion)
    DCF = pi_1*C_fn*FNR + (1-pi_1)*C_fp*FPR
    if normalized:
        return DCF / min(pi_1*C_fn, (1-pi_1)*C_fp)
    else:
        return DCF


def minimum_detection_cost(S, trueL, pi_1, C_fn, C_fp, normalized=False):
    """
    Computes the DCF obtained with the optimal threshold to understand the effect of mis-calibration.
    :param S is the binary log-likelihood ratio
    :param trueL stores the actual labels of the dataset
    :param pi_1 is the prior probability of class 1
    :param C_fn is the error cost of false negatives
    :param C_fp is the error cost of false positives
    :param normalized when set to true, the normalized bayes risk is computed
    """
    # sort the values
    sortS = np.sort(S)
    # set up the minimum DCF
    minDCF = sys.float_info.max
    # generate the confusion matrix for each threshold (value in sortS)
    for t in sortS:
        predL = np.zeros(S.size, dtype=int)
        predL[S > t] = 1
        conf = confusion(trueL, predL)
        # compute the (normalized) DCF
        DCF = dcf(conf, pi_1, C_fn, C_fp, normalized)
        # if it is lower than the minimum, update it
        if minDCF > DCF:
            minDCF = DCF
    return minDCF


# -------------- plot functions -------------- #
def roc_plot(S, trueL, file_name=None):
    """
    Plots the ROC curve.
    :param S is the binary log-likelihood ratio
    :param trueL stores the actual labels of the dataset
    :param file_name is the name of the file where we want to save the image, if any
    """
    # sort the values
    sortS = np.sort(S)
    # allocate FPR and TPR
    FPR = []
    TPR = []
    # generate the confusion matrix for each threshold (value in sortS)
    for t in sortS:
        predL = np.zeros(S.size, dtype=int)
        predL[S > t] = 1
        conf = confusion(trueL, predL)
        FPR.append(get_FPR(conf))
        TPR.append(1-get_FNR(conf))

    FPR = np.array(FPR)
    TPR = np.array(TPR)
    # plot the curve
    plt.plot(FPR, TPR)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid(visible=True, linestyle='--')
    plt.show()
    if file_name is not None:
        plt.savefig(file_name)


def bayes_error_plot(S, trueL, min_p_=-3, max_p_=3, points=21, file_name=None):
    """
    Plots the Bayes error plots.
    :param S is the binary log-likelihood ratio
    :param trueL stores the actual labels of the dataset
    :param min_p is the minimum prior log-odd we want to consider
    :param max_p is the maximum prior log-odd we want to consider
    :param points is the number of points we want to consider
    :param file_name is the name of the file where we want to save the image, if any
    """
    # generate linear prior log-odds
    effPriorLogOdds = np.linspace(min_p_, max_p_, points)
    # generate the correspondent pi_
    effPrior = 1/(1+np.exp(-effPriorLogOdds))
    # set up DCF and min DCF
    DCF = []
    minDCF = []
    for pi_ in effPrior:
        conf = confusion(trueL, binary_optimal_decision(S, pi_, 1, 1))
        DCF.append(dcf(conf, pi_, 1, 1, True))
        minDCF.append(minimum_detection_cost(S, trueL, pi_, 1, 1, True))
    DCF = np.array(DCF)
    minDCF = np.array(minDCF)
    # plot the curves
    plt.plot(effPriorLogOdds, DCF, label='DCF', color='r')
    plt.plot(effPriorLogOdds, minDCF, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([min_p_, max_p_])
    plt.xlabel('Prior log-odds')
    plt.ylabel('DCF value')
    plt.legend(loc='lower left')
    plt.show()
    if file_name is not None:
        plt.savefig(file_name)

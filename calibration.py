import numpy as np
import preprocessing
import optimal_decisions
import logistic_regression
import utilities as u


# TODO add doc strings

def calibrate(mode, scores, labels, pi_t: float, K: int = None):
    """
    main function for score calibration. It creates the calibration folds, then, depending on the mode,
    it calls the appropriate function on them
    :param mode: "simple", "recalibration_func" or "fusion"
    :param scores: row vector of scores (result of ML algorithm). Multiple row vectors can be stacked vertically for the fusion
    :param labels: array of the true labels 
    :param pi_t: the effective prior of the application
    :param K: the number of folds for K-fold algorithm. If None, use simple split 80-20
    :return: if "simple", the threshold (minDCF computed over the calibration set).
            If "recalibration_func", the recalibration function f(s)
    """
    # let's be sure we have a single array of score if we are not fusing
    assert ((mode != "fusion" and scores.shape[0] == 1) or (mode == "fusion" and scores.shape[0] > 1))
    # we want a single or just a few row vectors stacked on each other
    assert (scores.shape[0] <= scores.shape[1])

    if K is None:
        # split scores
        (XTR, LTR), (XTE, LTE) = preprocessing.split_dataset(scores, labels, 80)

        # call the right calibration function
        if mode == "simple":
            dcf = calibration_simple(XTR, LTR, XTE, LTE, pi_t)
            return dcf
        elif mode == "recalibration_func":
            f_s = recalibration(XTR, LTR, XTE, LTE, pi_t)
            return f_s
        elif mode == "fusion":
            f_s = fusion(XTR, LTR, XTE, LTE, pi_t)
            return f_s
    else:
        # divide scores using Kfold
        trainX, trainL = preprocessing.k_fold(K, scores, labels)

        # for each fold...
        for i in range(K):
            # extract the test and train sets
            (XTR, LTR), (XTE, LTE) = preprocessing.k_fold_split_folds(i, trainX, trainL)

            # call the right calibration function
            if mode == "simple":
                calibration_simple(XTR[0, :], LTR, XTE[0, :], LTE, pi_t)
            elif mode == "recalibration_func":
                recalibration(XTR, LTR, XTE, LTE, pi_t)
            elif mode == "fusion":
                fusion(XTR, LTR, XTE, LTE, pi_t)

        if mode == "simple":
            # re estimate the final threshold all the original scores
            minDCF_global = optimal_decisions.minimum_detection_cost(scores[0, :], labels, pi_t, 1, 1, True)
            return minDCF_global
        elif mode == "recalibration_func":
            funcs = [recalibration_func_train(scores, labels, pi_1, l) for l in _lambda for pi_1 in _pi_1]
            return funcs
        elif mode == "fusion":
            f_s = [recalibration_func_train(scores, labels, pi_1, l) for l in _lambda for pi_1 in _pi_1]
            return f_s

    # I should not be here
    raise ValueError("Invalid calibration function")


def calibration_simple(XTR, LTR, XTE, LTE, pi_t: float):
    """
    Calibrate the scores using the minDCF of the training set
    :param XTR: train scores of the calibration set
    :param LTR: train labels of the calibration set
    :param XTE: test scores of the calibration set
    :param LTE: test labels of the calibration set
    :param pi_t: the effective prior of the application
    """

    trainX = XTR.flatten()
    testX = XTE.flatten()

    # compute minDCF over the train and test folds
    minDCF_train = optimal_decisions.minimum_detection_cost(trainX, LTR, pi_t, 1, 1, True)
    minDCF_test = optimal_decisions.minimum_detection_cost(testX, LTE, pi_t, 1, 1, True)
    # get the actual DCF with the optimal threshold
    t = -np.log(pi_t/(1 - pi_t))
    actualDCF_t = optimal_decisions.actual_dcf(testX, LTE, t, pi_t)
    # compute the actual DCF using minDCF as threshold
    actualDCF_t_star = optimal_decisions.actual_dcf(testX, LTE, minDCF_train, pi_t)
    print(
        f"simple calibration (pi_t={pi_t}): minDCF = {minDCF_test}, actDCF(t) = {actualDCF_t}, actualDCF(t_star) = {actualDCF_t_star}")

    return actualDCF_t_star


def recalibration(XTR, LTR, XTE, LTE, pi_tilde: float):
    """
    Train the recalibration function on the calibration set and evaluate it
    :param XTR: train scores of the calibration set
    :param LTR: train labels of the calibration set
    :param XTE: test scores of the calibration set
    :param LTE: test labels of the calibration set
    :param pi_tilde: the effective prior of the application
    """
    testX = XTE.flatten()

    # print DCFs for the evaluation
    # compute minDCF over the test fold
    minDCF_test = optimal_decisions.minimum_detection_cost(testX, LTE, pi_tilde, 1, 1, True)
    # get the actual DCF with the optimal threshold (uncalibrated)
    t = -np.log(pi_tilde/(1 - pi_tilde))
    actualDCF_uncal = optimal_decisions.actual_dcf(testX, LTE, t, pi_tilde)
    print(
        f"recalibration_func (pi_t={pi_tilde}): minDCF = {minDCF_test}, actDCF(uncalibrated) = {actualDCF_uncal}")

    funcs = []
    for pi_1 in _pi_1:
        print("\t", end='')
        for l in _lambda:
            # compute the logistic regression on the scores to get the function coefficients
            f_s = recalibration_func_train(XTR, LTR, pi_1, l)

            # now, recalibrate the score to test the recalibration function
            s_cal = f_s(testX)
            # compute atualDCF with theoretical threshold over recalibrated scores
            actualDCF_cal = optimal_decisions.actual_dcf(s_cal, LTE, t, pi_tilde)
            print(f"actualDCF(pi={pi_1},l={l}) = {actualDCF_cal}\t", end='')

            funcs.append(f_s)
        print()

    return funcs


def fusion(XTR, LTR, XTE, LTE, pi_tilde: float):
    """
    Train the recalibration function on the calibration set and evaluate it
    :param XTR: matrix of scores of the train calibration set (each row is a set of scores)
    :param LTR: train labels of the calibration set
    :param XTE: matrix of scores of the test calibration set (each row is a set of scores)
    :param LTE: test labels of the calibration set
    :param pi_tilde: the effective prior of the application
    """

    print(f"fusion (pi_t={pi_tilde}): ")
    # compute the theoretical threashold
    t = -np.log(pi_tilde/(1 - pi_tilde))

    funcs = []
    for pi_1 in _pi_1:
        print("\t", end='')
        for l in _lambda:
            # compute the logistic regression on the scores to get the function coefficients
            f_s = recalibration_func_train(XTR, LTR, pi_1, l)

            # now, recalibrate the score to test the recalibration function
            s_cal = f_s(XTE)
            s_cal = s_cal.flatten()
            # compute atualDCF with theoretical threshold over recalibrated scores
            actualDCF_cal = optimal_decisions.actual_dcf(s_cal, LTE, t, pi_tilde)
            print(f"actualDCF(pi={pi_1},l={l}) = {actualDCF_cal}\t", end='')

            funcs.append(f_s)
        print()

    return funcs


def recalibration_func_train(XTR, LTR, pi_1, l: float = 0):
    """
    Train the recalibration function on the calibration set
    :param XTR: train scores of the calibration set
    :param LTR: train labels of the calibration set
    :param pi_1: the prior of the logistic regression
    :param l: lambda for the logistic regression
    """
    alpha, beta_prime, J_a_b = logistic_regression.binary_lr_fit(XTR, LTR, l, pi_1)

    # define recalibration function using alpha, beta_prime
    def f_s(s):
        """
        recalibration function f(s)
        :param s: scores to recalibrate using the logistic regression method
        """
        # compute beta offset
        th = np.log(pi_1/(1 - pi_1))
        # recalibrate scores
        new_scores = np.dot(u.vrow(alpha), s) + beta_prime - th
        return new_scores

    return f_s


if __name__ == '__main__':
    # type of calibration: ["simple", "recalibration_func", "fusion"]
    _calibration_type = "fusion"
    # scores to calibrate
    _scores = u.vrow(np.load("scores/GMM4.npy"))
    # actual labels of the dataset
    _labels = np.load("scores/GMM_labels_4.npy")
    _pi_tilde = [0.1, 0.5, 0.9]  # array of pi_tilde to test
    _K = None  # None, or the number of folds

    # recalibration function/fusion settings
    _pi_1 = [0.1, 0.5, 0.9]  # array of priors (for the logistic regression of the recalibration function f(s))
    _lambda = [0]  # lambda for the logistic regression
    _scores_fusion = u.vrow(np.load(
        "scores/MVG_tied_ pca8.npy"))  # set None to compute normal recalibration function on _scores, else fuse _scores with these scores

    if _calibration_type == "fusion":
        _scores = np.vstack([_scores, _scores_fusion])
    for pi in _pi_tilde:
        calibrate(_calibration_type, _scores, _labels, pi, _K)
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
    :param mode: "simple" or "recalibration_func"
    :param scores: array of scores (result of ML algorithm)
    :param labels: array of the true labels 
    :param pi_t: the effective prior of the application
    :param K: the number of folds for K-fold algorithm. If None, use simple split 80-20
    :return: if "simple", the threshold (minDCF computed over the calibration set).
            If "recalibration_func", the recalibration function f(s)
    """

    if K is None:
        # split scores
        (XTR, LTR), (XTE, LTE) = preprocessing.split_dataset(u.vrow(scores), labels, 80)

        # call the right calibration function
        if mode == "simple":
            dcf = calibration_simple(XTR, LTR, XTE, LTE, pi_t)
            return dcf
        elif mode == "recalibration_func":
            f_s = recalibration(XTR, LTR, XTE, LTE, pi_t)
            return f_s
    else:
        # divide scores using Kfold
        trainX, trainL = preprocessing.k_fold(K, u.vrow(scores), labels)

        # for each fold...
        for i in range(K):
            # extract the test and train sets
            (XTR, LTR), (XTE, LTE) = preprocessing.k_fold_split_folds(i, trainX, trainL)

            # call the right calibration function
            if mode == "simple":
                calibration_simple(XTR, LTR, XTE, LTE, pi_t)
            elif mode == "recalibration_func":
                recalibration(XTR, LTR, XTE, LTE, pi_t)

        if mode == "simple":
            # re estimate the final threshold all the original scores
            minDCF_global = optimal_decisions.minimum_detection_cost(scores, labels, pi_t, 1, 1, True)
            return minDCF_global
        elif mode == "recalibration_func":
            funcs = [recalibration_func_train(scores, labels, pi_1) for pi_1 in _pi_1]
            return funcs

    # I should not be here
    raise ValueError("Invalid calibration function")


def calibration_simple(XTR, LTR, XTE, LTE, pi_t: float):
    """
    Calibrate the scores using the
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
    print(f"simple calibration (pi_t={pi_t}): minDCF = {minDCF_test}, actDCF(t) = {actualDCF_t}, actualDCF(t_star) = {actualDCF_t_star}")

    return actualDCF_t_star


def recalibration(XTR, LTR, XTE, LTE, pi_tilde: float):
    testX = XTE.flatten()

    # print DCFs for the evaluation
    # compute minDCF over the test fold
    minDCF_test = optimal_decisions.minimum_detection_cost(testX, LTE, pi_tilde, 1, 1, True)
    # get the actual DCF with the optimal threshold (uncalibrated)
    t = -np.log(pi_tilde/(1 - pi_tilde))
    actualDCF_uncal = optimal_decisions.actual_dcf(testX, LTE, t, pi_tilde)
    print(
        f"recalibration_func (pi_t={pi_tilde}): minDCF = {minDCF_test}, actDCF(uncalibrated) = {actualDCF_uncal}", end='')

    funcs = []
    for pi_1 in _pi_1:
        # compute the logistic regression on the scores to get the function coefficients
        f_s = recalibration_func_train(XTR, LTR, pi_1)

        # now, recalibrate the score to test the recalibration function
        s_cal = f_s(testX)
        # compute atualDCF with theoretical threshold over recalibrated scores
        actualDCF_cal = optimal_decisions.actual_dcf(s_cal, LTE, t, pi_tilde)
        print(f", actualDCF(pi={pi_1}) = {actualDCF_cal}", end='')

        funcs.append(f_s)

    print()
    return funcs


def recalibration_func_train(XTR, LTR, pi_1):
    alpha, beta_prime, J_a_b = logistic_regression.binary_lr_fit(u.vrow(XTR), LTR, 0, pi_1)

    # define recalibration function using alpha, beta_prime
    def f_s(s):
        """
        recalibration function f(s)
        :param s: scores to recalibrate using the logistic regression method
        """
        # compute beta offset
        th = np.log(pi_1/(1 - pi_1))
        # recalibrate scores
        new_scores = alpha*s + beta_prime - th
        return new_scores

    return f_s


if __name__ == '__main__':
    _calibration_type = "recalibration_func"  # ["simple", "recalibration_func"]
    _scores = np.load("scores/GMM4.npy")
    _labels = np.load("scores/GMM_labels_4.npy")
    _pi_tilde = [0.1, 0.5, 0.8]  # array of pi_tilde to test
    _K = None  # None, or the number of folds

    _pi_1 = [0.1, 0.5, 0.8]  # array of priors (for the logistic regression of the racalibration function f(s))
    _lambda = [0, 0.01, 0.1]

    for pi in _pi_tilde:
        calibrate(_calibration_type, _scores, _labels, pi, _K)

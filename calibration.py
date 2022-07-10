import numpy as np
import preprocessing
import optimal_decisions
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
            recalibration()
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
                recalibration()

        if mode == "simple":
            # re estimate the final threshold all the original scores
            minDCF_global = optimal_decisions.minimum_detection_cost(scores, labels, pi_t, 1, 1, True)
            return minDCF_global

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

    # compute minDCF over the train and test fold
    minDCF_train = optimal_decisions.minimum_detection_cost(trainX, LTR, pi_t, 1, 1, True)
    minDCF_test = optimal_decisions.minimum_detection_cost(testX, LTE, pi_t, 1, 1, True)

    # get the actual DCF with the optimal threshold
    t = -np.log(pi_t/(1 - pi_t))
    actualDCF_t = optimal_decisions.actual_dcf(testX, LTE, t, pi_t)

    # compute the actual DCF using minDCF as threshold
    actualDCF_t_star = optimal_decisions.actual_dcf(testX, LTE, minDCF_train, pi_t)

    print(
        f"simple calibration: minDCF = {minDCF_test}, actDCF(t) = {actualDCF_t}, actualDCF(t_star) = {actualDCF_t_star}")

    return actualDCF_t_star


def recalibration():
    raise NotImplemented


def calibration_main():
    calibration_type = "simple"  # ["simple", "recalibration_func"]
    scores = np.load("scores/GMM4.npy")
    labels = np.load("scores/GMM_labels_4.npy")
    pi_tilde = [0.1, 0.5, 0.8]  # array of pi_tilde to test
    K = None  # None, or the number of folds

    for pi in pi_tilde:
        calibrate(calibration_type, scores, labels, pi, K)


if __name__ == '__main__':
    calibration_main()


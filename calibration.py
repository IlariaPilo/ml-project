import numpy as np
import preprocessing
import optimal_decisions
import utilities as u

# TODO add doc strings

# calibration
calibration_type = "simple"
# calibration =  ["simple", "recalibration_func"]

def calibration_main():
    #TODO load scores
    calibrate(calibration_type, scores, labels, pi, K)



def calibrate(mode, scores, labels, pi, K=None):
    """
    main function for score calibration. It creates the calibration folds, then, depending on the mode,
    it calls the appropriate function on them
    :param mode: "simple" or "recalibration_func"
    :return:
    """

    if K is None:
        # split scores
        (XTR, LTR), (XTE, LTE) = preprocessing.split_dataset(u.vrow(scores), labels, 80)

        # call the right calibration function
        if mode == "simple":
            dcf = calibration_simple(XTR, LTR, XTE, LTE, pi)
            return dcf
        elif mode == "recalibration_func":
            recalibration()
    else:
        # divide scores using Kfold.
        trainX, trainL = preprocessing.k_fold(K, u.vrow(scores), labels)

        # for each fold...
        for i in range(K):
            # extract the test and train sets
            (XTR, LTR), (XTE, LTE) = preprocessing.k_fold_split_folds(i, trainX, trainL)

            # call the right calibration function
            if mode == "simple":
                calibration_simple(XTR, LTR, XTE, LTE, pi)
            elif mode == "recalibration_func":
                recalibration()

        if mode == "simple":
            # TODO re estimate the final threshold on the all original scores
            return

    # I should not be here
    raise ValueError("Invalid calibration function")


def calibration_simple(XTR, LTR, XTE, LTE, pi):
    """
    Calibrate the scores using the
    :param XTR:
    :param LTR:
    :param XTE:
    :param LTE:
    :param pi:
    """

    trainX = XTR.flatten()
    testX = XTE.flatten()

    # compute minDCF over the train and test fold
    minDCF_train = optimal_decisions.minimum_detection_cost(trainX, LTR, pi, 1, 1, True)
    minDCF_test = optimal_decisions.minimum_detection_cost(testX, LTE, pi, 1, 1, True)

    # get the actual DCF with the optimal threshold
    t = -np.log(pi/(1 - pi))
    actualDCF_t = optimal_decisions.actual_dcf(testX, LTE, t, pi)

    # compute the actual DCF using minDCF as threshold
    actualDCF_t_star = optimal_decisions.actual_dcf(testX, LTE, minDCF_train, pi)

    print(f"simple calibration: minDCF = {minDCF_test}, actDCF(t) = {actualDCF_t}, actualDCF(t_star) = {actualDCF_t_star}")

    return actualDCF_t_star

def recalibration():
    raise NotImplemented

if __name__ == '__main__':
    calibration_main()








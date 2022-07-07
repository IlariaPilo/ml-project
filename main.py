import numpy as np

import load
import preprocessing
import utilities
import gaussian_models
import optimal_decisions

def data_preprocessing(trainX, testX, config, trainL, testL):
    if config["gaussianization"]:
        # apply features gaussianization
        trainX, testX = preprocessing.feats_gaussianization(trainX, testX)
        if config["is_print"]:
            # print the (new) distribution
            preprocessing.feats_hist(np.hstack((trainX, testX)), np.append(trainL, testL))
            # print the correlation between gaussianized features
            # preprocessing.feats_correlation(np.hstack((trainX, testX)), np.append(trainL, testL))   # <- same as above
    if config["pca"] is not None:
        # apply pca
        P = preprocessing.pca(trainX, config["pca"])
        # transform training data
        trainX = np.dot(P.T, trainX)
        # transform test data
        testX = np.dot(P.T, testX)
    return trainX, testX


def model_fit(trainX, trainL, gaussian_fit, _something_else_):
    if gaussian_fit is not None:
        return gaussian_fit(trainX, trainL)


def model_predict(testX, params, gaussian_fit, _something_else_):
    if gaussian_fit is not None:
        mu_s, C_s = params
        return gaussian_models.mvg_log_predict(testX, mu_s, C_s)


def main(config):
    # ----------- 1. read the training dataset ----------- #
    trainX, trainL = load.load("data/Train")
    # ----------- 2. feature analysis ----------- #
    if config["is_print"]:
        # print the distribution of all features
        preprocessing.feats_hist(trainX, trainL)
        # print the correlation between features
        preprocessing.feats_correlation(trainX, trainL)
    if config["k_fold"] is None:
        # >>>> SINGLE SPLIT <<<< #
        # split the dataset: we use 80% for training and 20% for evaluation
        (XTR, LTR), (XTE, LTE) = preprocessing.split_dataset(trainX, trainL, 80)
        # ----------- 3a. preprocessing ----------- #
        XTR, XTE = data_preprocessing(XTR, XTE, config, LTR, LTE)
        # ----------- 4a. modelling ----------- #
        params = model_fit(XTR, LTR, config["gaussian_fit"], None)
        # ----------- 5a. predicting ----------- #
        predL, S = model_predict(XTE, params, config["gaussian_fit"], None)
        # ----------- 6a. evaluation ----------- #
        # we are using normalized min_dcf to evaluate the model
        minDCF = optimal_decisions.minimum_detection_cost(S, LTE, 0.5, 1, 1, normalized=True)
        print('minDCF: %f' % minDCF)
        err = utilities.err_rate(predL, LTE) * 100
        print('Error rate: %.3f' % err)
    else:
        do_something_else = 0





if __name__ == '__main__':
    # ----------- 0. configuration ----------- #
    config = {
        # is_print - if true, we generate plots
        "is_print": False,
        # k_fold - if None, we use single fold. otherwise, it is an int storing the number of folds
        "k_fold": None,
        # gaussianization - if true, we gaussianize the features
        "gaussianization": False,
        # pca - if None, no PCA is applied. otherwise, it is an int storing the number of features we want to have after
        # the pca operation
        "pca": None,
        # gaussian_fit - the type of basic gaussian fit we want to apply (if any)
        "gaussian_fit": gaussian_models.mvg_fit
    }
    main(config)








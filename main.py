import numpy as np

import load
import preprocessing
import utilities
import gaussian_models
import optimal_decisions

def data_preprocessing(trainX, testX, hyperparams, is_print, trainL, testL):
    if hyperparams["gaussianization"]:
        # apply features gaussianization
        trainX, testX = preprocessing.feats_gaussianization(trainX, testX)
        if is_print:
            # print the (new) distribution
            preprocessing.feats_hist(np.hstack((trainX, testX)), np.append(trainL, testL))
            # print the correlation between gaussianized features
            # preprocessing.feats_correlation(np.hstack((trainX, testX)), np.append(trainL, testL))   # <- same as above
    if hyperparams["pca"] is not None:
        # apply pca
        P = preprocessing.pca(trainX, hyperparams["pca"])
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

    # TODO
    # get parameters combination
    # hyperparams = preprocessing.hyperparams_combinations(config["hyperparams"])

    if config["k_fold"] is None:
        # >>>> SINGLE SPLIT <<<< #
        # split the dataset: we use 80% for training and 20% for evaluation
        (XTR, LTR), (XTE, LTE) = preprocessing.split_dataset(trainX, trainL, 80)

        print('--- hyperparameters: ' + config["hyperparams"] + " ---")

        # ----------- 3a. preprocessing ----------- #
        XTR, XTE = data_preprocessing(XTR, XTE, config["hyperparams"], config["is_print"], LTR, LTE)

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
        # >>>> K-FOLD <<<< #
        k = config["k_fold"]
        # get folds
        foldsX, foldsL = preprocessing.k_fold(k, trainX, trainL)
        predL = np.empty(0)
        S = np.empty(0)
        # for each fold
        for i in range(0, k):
            (XTR, LTR), (XTE, LTE) = preprocessing.k_fold_split_folds(i, foldsX, foldsL)

            # ----------- 3b. preprocessing ----------- #
            XTR, XTE = data_preprocessing(XTR, XTE, config["hyperparams"], config["is_print"], LTR, LTE)

            # ----------- 4b. modelling ----------- #
            params = model_fit(XTR, LTR, config["gaussian_fit"], None)

            # ----------- 5b. predicting ----------- #
            predL_, S_ = model_predict(XTE, params, config["gaussian_fit"], None)
            predL = np.concatenate((predL, predL_))
            S = np.concatenate((S, S_))

        # ----------- 6b. evaluation ----------- #
        trueL = np.hstack(foldsL)
        # we are using normalized min_dcf to evaluate the model
        minDCF = optimal_decisions.minimum_detection_cost(S, trueL, 0.5, 1, 1, normalized=True)
        print('minDCF: %f' % minDCF)
        err = utilities.err_rate(predL, trueL) * 100
        print('Error rate: %.3f' % err)


if __name__ == '__main__':
    # ----------- 0. configuration ----------- #
    config = {
        # is_print - if true, we generate plots
        "is_print": False,
        # k_fold - if None, we use single fold. otherwise, it is an int storing the number of folds.
        "k_fold": 5,
        # gaussian_fit - the type of basic gaussian fit we want to apply (if any)
        "gaussian_fit": gaussian_models.mvg_fit,
        "hyperparams": {
            # gaussianization - if true, we gaussianize the features
            "gaussianization": True,
            # pca - if None, no PCA is applied. otherwise, it is an int storing the number of features we want to have
            # after the pca operation
            "pca": None
        }
    }
    main(config)

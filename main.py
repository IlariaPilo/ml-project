import numpy as np

import load
import preprocessing
import utilities
import gaussian_models
import optimal_decisions


def data_preprocessing(trainX, testX, params, is_print, trainL, testL):
    if params["gaussianization"]:
        # apply features gaussianization
        trainX, testX = preprocessing.feats_gaussianization(trainX, testX)
        if is_print:
            # print the (new) distribution
            preprocessing.feats_hist(np.hstack((trainX, testX)), np.append(trainL, testL))
            # print the correlation between gaussianized features
            # preprocessing.feats_correlation(np.hstack((trainX, testX)), np.append(trainL, testL))   # <- same as above
    if params["pca"] is not None:
        # apply pca
        P = preprocessing.pca(trainX, params["pca"])
        # transform training data
        trainX = np.dot(P.T, trainX)
        # transform test data
        testX = np.dot(P.T, testX)
    return trainX, testX


def model_fit(trainX, trainL, gaussian_fit, _something_else_):
    if gaussian_fit is not None:
        return gaussian_fit(trainX, trainL)


def model_predict(testX, model, gaussian_fit, _something_else_):
    if gaussian_fit is not None:
        mu_s, C_s = model
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

    # get parameters combination
    params_list = utilities.params_combinations(config["params"])

    if config["k_fold"] is None:
        # >>>> SINGLE SPLIT <<<< #
        # split the dataset: we use 80% for training and 20% for evaluation
        (XTR, LTR), (XTE, LTE) = preprocessing.split_dataset(trainX, trainL, 80)

        # for each param
        for param in params_list:
            print("--------------------------------------------------------------------------------------")
            print(param)

            # ----------- 3a. preprocessing ----------- #
            XTR_, XTE_ = data_preprocessing(XTR, XTE, param, config["is_print"], LTR, LTE)

            # ----------- 4a. modelling ----------- #
            model = model_fit(XTR_, LTR, param["gaussian_fit"], None)

            # ----------- 5a. predicting ----------- #
            predL, S = model_predict(XTE_, model, param["gaussian_fit"], None)

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

        # for each param
        for param in params_list:
            print("--------------------------------------------------------------------------------------")
            print(param)

            predL = np.empty(0)
            S = np.empty(0)
            # for each fold
            for i in range(0, k):
                (XTR, LTR), (XTE, LTE) = preprocessing.k_fold_split_folds(i, foldsX, foldsL)

                # ----------- 3b. preprocessing ----------- #
                XTR, XTE = data_preprocessing(XTR, XTE, param, config["is_print"], LTR, LTE)

                # ----------- 4b. modelling ----------- #
                model = model_fit(XTR, LTR, param["gaussian_fit"], None)

                # ----------- 5b. predicting ----------- #
                predL_, S_ = model_predict(XTE, model, param["gaussian_fit"], None)
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
        "params": {
            # gaussianization - if true, we gaussianize the features
            "gaussianization": [False, True],
            # pca - if None, no PCA is applied. otherwise, it is an int storing the number of features we want to have
            # after the pca operation
            "pca": [None,10,9,8,6],
            # gaussian_fit - the type of basic gaussian fit we want to apply (if any)
            "gaussian_fit": [gaussian_models.mvg_fit, gaussian_models.mvg_naive_bayes_fit,
                             gaussian_models.mvg_tied_covariance_fit, gaussian_models.mvg_tied_naive_bayes_fit]
        }
    }
    main(config)

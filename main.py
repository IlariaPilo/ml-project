import numpy as np

import load
import preprocessing
import support_vector_machines
import utilities
import gaussian_models
import optimal_decisions
import logistic_regression


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


def model_fit(trainX, trainL, params):
    if params["gaussian_fit"] is not None:
        return params["gaussian_fit"](trainX, trainL)
    if params["logistic_regression"] is not None:
        return logistic_regression.binary_lr_fit(trainX, trainL, params["logistic_regression"])
    if params["quadratic_regression"] is not None:
        phi = logistic_regression.quadratic_expansion(trainX)
        return logistic_regression.binary_lr_fit(phi, trainL, params["quadratic_regression"])
    if params["svm"]:
        if params["kernel"] is None:
            return support_vector_machines.linear_svm_fit(trainX, trainL, params["C"], params["K"])
        return support_vector_machines.kernel_svm_fit(trainX, trainL, params["C"], params["K"], params["kernel"])


def model_predict(testX, model, params):
    if params["gaussian_fit"] is not None:
        mu_s, C_s = model
        return gaussian_models.mvg_log_predict(testX, mu_s, C_s)
    if params["logistic_regression"] is not None:
        w, b, _ = model
        return logistic_regression.binary_lr_predict(testX, w, b)
    if params["quadratic_regression"] is not None:
        w, b, _ = model
        phi = logistic_regression.quadratic_expansion(testX)
        return logistic_regression.binary_lr_predict(phi, w, b)
    if params["svm"]:
        if params["kernel"] is None:
            w_, _ = model
            return support_vector_machines.linear_svm_predict(testX, w_, params["K"])
        alpha, _, trainX, trainL = model
        return support_vector_machines.kernel_svm_predict(testX, alpha, trainX, trainL, params["K"], params["kernel"])


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
            model = model_fit(XTR_, LTR, param)

            # ----------- 5a. predicting ----------- #
            predL, S = model_predict(XTE_, model, param)

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
                model = model_fit(XTR, LTR, param)

                # ----------- 5b. predicting ----------- #
                predL_, S_ = model_predict(XTE, model, param)
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
        "k_fold": None,
        "params": {
            # gaussianization - if true, we gaussianize the features
            "gaussianization": [False],
            # pca - if None, no PCA is applied. otherwise, it is an int storing the number of features we want to have
            # after the pca operation
            "pca": [None, 8],
            # gaussian_fit - the type of basic gaussian fit we want to apply (if any)
            # "gaussian_fit": [gaussian_models.mvg_fit, gaussian_models.mvg_naive_bayes_fit,
            #                 gaussian_models.mvg_tied_covariance_fit, gaussian_models.mvg_tied_naive_bayes_fit]
            "gaussian_fit": [None],
            # logistic_regression - the value of hyperparameter lambda of logistic regression (if any)
            "logistic_regression": [None],
            # "pi_t": [0.5, 0.1, 0.9],
            # quadratic_regression - the value of hyperparameter lambda of quadratic logistic regression (if any)
            # TODO --- check weird results of quadratic regression
            # "quadratic_regression": [10 ** (-6), 10 ** (-3), 10 ** (-1), 1, 10],
            "quadratic_regression": [None],
            # svm - True if we want to use it. C and K are the related hyperparameters
            "svm": [True],
            "kernel": [support_vector_machines.poly_kernel(2)],     # None if we want linear svm
            "C": [1, 10],
            "K": [1],

        }
    }
    main(config)

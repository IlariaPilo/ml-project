import numpy as np

import load
import preprocessing

def main(is_print, gaussianization):
    # ----------- 1. read the training dataset ----------- #
    trainX, trainL = load.load("data/Train")
    # ----------- 2. feature analysis ----------- #
    if is_print:
        # print the distribution of all features
        preprocessing.feats_hist(trainX, trainL)
        # print the correlation between features
        preprocessing.feats_correlation(trainX, trainL)
    if gaussianization:
        # apply features gaussianization
        trainX = preprocessing.feats_gaussianization(trainX)
        if is_print:
            # print the (new) distribution
            preprocessing.feats_hist(trainX, trainL)
            # print the correlation between gaussianized features
            preprocessing.feats_correlation(trainX, trainL)

if __name__ == '__main__':
    main(False, False)



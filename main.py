import numpy as np

import load
import preprocessing

if __name__ == '__main__':
    # 1. read the training dataset
    trainX, trainL = load.load("data/Train")
    # 2. feature analysis
    # print the distribution of all features
    # preprocessing.feats_hist(trainX, trainL)
    # apply features gaussianization
    trainX = preprocessing.feats_gaussianization(trainX)
    # print the (new) distribution
    preprocessing.feats_hist(trainX, trainL)

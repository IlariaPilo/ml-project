import numpy as np
from sklearn.svm import SVC
import load
import preprocessing
import utilities

trainX, trainL = load.load("data/Train")
(XTR, LTR), (XTE, LTE) = preprocessing.split_dataset(trainX, trainL, 80)
XTR = XTR.T
XTE = XTE.T

svm = SVC(kernel='linear', C=10.0)
svm.fit(XTR, LTR)
predL = svm.predict(XTE)

err = utilities.err_rate(predL, LTE)*100

print(err)
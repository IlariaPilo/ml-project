import numpy as np
import matplotlib.pyplot as plt

import load
import optimal_decisions
import preprocessing


def linear_svm_plot():
    C = np.array([0.01, 0.1, 1, 10])
    gaussF_pcaF_K_1 = np.array([0.058389, 0.060039, 0.060306, 0.056706])
    gaussF_pcaF_K_10 = np.array([0.060073, 0.060273, 0.060239, 0.060273])
    gaussF_pca8_K_1 = np.array([0.061623, 0.058322, 0.058456, 0.058256])
    gaussF_pca8_K_10 = np.array([0.058389, 0.058423, 0.056806, 0.058423])
    gaussT_pcaF_K_1 = np.array([0.076841, 0.063173, 0.063140, 0.061489])
    gaussT_pcaF_K_10 = np.array([0.076841, 0.063173, 0.063140, 0.063140])
    gaussT_pca8_K_1 = np.array([0.180585, 0.177118, 0.177218, 0.175534])
    gaussT_pca8_K_10 = np.array([0.180585, 0.177118, 0.177218, 0.180118])

    plt.figure()
    plt.suptitle("Linear SVM results, gaussianized features comparison")
    plt.subplot(2, 2, 1)
    plt.title("No PCA, K = 1")
    plt.semilogx(C, gaussT_pcaF_K_1, label='Gaussianized features')
    plt.semilogx(C, gaussF_pcaF_K_1, label='Standard features')
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.title("No PCA, K = 10")
    plt.semilogx(C, gaussT_pcaF_K_10, label='Gaussianized features')
    plt.semilogx(C, gaussF_pcaF_K_10, label='Standard features')
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.title("PCA (m = 8), K = 1")
    plt.semilogx(C, gaussT_pca8_K_1, label='Gaussianized features')
    plt.semilogx(C, gaussF_pca8_K_1, label='Standard features')
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.title("PCA (m = 8), K = 10")
    plt.semilogx(C, gaussT_pca8_K_10, label='Gaussianized features')
    plt.semilogx(C, gaussF_pca8_K_10, label='Standard features')
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend()
    plt.show()

    plt.figure()
    plt.suptitle("Linear SVM results, PCA comparison")
    plt.subplot(1, 2, 1)
    plt.title("K = 1")
    plt.semilogx(C, gaussF_pcaF_K_1, label='No PCA')
    plt.semilogx(C, gaussF_pca8_K_1, label='PCA (m = 8)')
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("K = 10")
    plt.semilogx(C, gaussF_pcaF_K_10, label='No PCA')
    plt.semilogx(C, gaussF_pca8_K_10, label='PCA (m = 8)')
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend()
    plt.show()

    plt.figure()
    plt.suptitle("Linear SVM results, K comparison")
    plt.subplot(1, 2, 1)
    plt.title("No PCA")
    plt.semilogx(C, gaussF_pcaF_K_1, label='K = 1')
    plt.semilogx(C, gaussF_pcaF_K_10, label='K = 10')
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("PCA (m = 8)")
    plt.semilogx(C, gaussF_pca8_K_1, label='K = 1')
    plt.semilogx(C, gaussF_pca8_K_10, label='K = 10')
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend()
    plt.show()


def gmm_plot(noPca, pca8):
    labels = [2,4,8,16,32,64,128,256]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, noPca, width, label='No PCA')
    rects2 = ax.bar(x + width / 2, pca8, width, label='PCA (m = 8)')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('minDCF')
    ax.set_xlabel('GMM components number')
    ax.set_xticks(x, labels)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()


def gmm_plot2(em, diag, tied, tied_diag):
    labels = [2,4,8,16,32,64,128,256]

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width*1.5, em, width, label='Full-Cov')
    rects2 = ax.bar(x - width / 2, diag, width, label='Diag-Cov')
    rects3 = ax.bar(x + width / 2, tied, width, label='Tied-Cov')
    rects4 = ax.bar(x + width*1.5, tied_diag, width, label='Tied-Diag-Cov')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('minDCF')
    ax.set_xlabel('GMM components number')
    ax.set_xticks(x, labels)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()


def lr_plot():
    l = [10**(-6), 0.001, 0.1, 1, 10]
    gaussF_pcaF_pi5 = [0.061723, 0.061723, 0.065457, 0.083542, 0.160183]
    gaussF_pcaF_pi1 = [0.143114, 0.143114, 0.156016, 0.280378, 0.553555]
    gaussF_pcaF_pi9 = [0.121996, 0.121996, 0.120612, 0.204187, 0.548105]
    gaussF_pca8_pi5 = [0.060073, 0.060073, 0.061956, 0.086575, 0.161866]
    gaussF_pca8_pi1 = [0.157966, 0.151365, 0.170867, 0.293879, 0.555956]
    gaussF_pca8_pi9 = [0.108828, 0.107144, 0.125663, 0.202504, 0.556522]
    gaussT_pcaF_pi5 = [0.059873, 0.068657, 0.162166, 0.271277, 0.494849]
    gaussT_pcaF_pi1 = [0.207171, 0.222022, 0.528653, 0.798530, 0.841734]
    gaussT_pcaF_pi9 = [0.165467, 0.165467, 0.434527, 0.764010, 0.784212]
    gaussT_pca8_pi5 = [0.177151, 0.177251, 0.220189, 0.301230, 0.509668]
    gaussT_pca8_pi1 = [0.409391, 0.420642, 0.614761, 0.846985, 0.873837]
    gaussT_pca8_pi9 = [0.422742, 0.434227, 0.533253, 0.814098, 0.846501]

    plt.figure()
    plt.suptitle("Logistic regression results")
    plt.subplot(2, 2, 1)
    plt.title("Raw features, no PCA")
    plt.semilogx(l, gaussF_pcaF_pi5, 'r', label=r'$\pi = 0.5$')
    plt.semilogx(l, gaussF_pcaF_pi1, 'b', label=r'$\pi = 0.1$')
    plt.semilogx(l, gaussF_pcaF_pi9, 'g', label=r'$\pi = 0.9$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("minDCF")
    plt.xticks(l)
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.title("Raw features, PCA (m = 8)")
    plt.semilogx(l, gaussF_pca8_pi5, 'r', label=r'$\pi = 0.5$')
    plt.semilogx(l, gaussF_pca8_pi1, 'b', label=r'$\pi = 0.1$')
    plt.semilogx(l, gaussF_pca8_pi9, 'g', label=r'$\pi = 0.9$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("minDCF")
    plt.xticks(l)
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.title("Gaussianized features, no PCA")
    plt.semilogx(l, gaussT_pcaF_pi5, 'r', label=r'$\pi = 0.5$')
    plt.semilogx(l, gaussT_pcaF_pi1, 'b', label=r'$\pi = 0.1$')
    plt.semilogx(l, gaussT_pcaF_pi9, 'g', label=r'$\pi = 0.9$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("minDCF")
    plt.xticks(l)
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.title("Gaussianized features, PCA (m = 8)")
    plt.semilogx(l, gaussT_pca8_pi5, 'r', label=r'$\pi = 0.5$')
    plt.semilogx(l, gaussT_pca8_pi1, 'b', label=r'$\pi = 0.1$')
    plt.semilogx(l, gaussT_pca8_pi9, 'g', label=r'$\pi = 0.9$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("minDCF")
    plt.xticks(l)
    plt.legend()
    plt.show()

def gmm_plot_main():
    pcaF_em = np.array([0.054989, 0.046805, 0.035020, 0.046538, 0.048455, 0.078324, 0.116378, 0.211705])
    pcaF_diag = np.array([0.371721, 0.138897, 0.107761, 0.099810, 0.091926, 0.085259, 0.093476, 0.125463])
    pca8_em = np.array([0.051789, 0.044988, 0.039971, 0.038387, 0.048255, 0.058389, 0.089776, 0.121162])
    pca8_diag = np.array([0.095260, 0.093443, 0.096943, 0.086609, 0.091492, 0.093343, 0.103210, 0.146615])
    pcaF_tied = np.array([0.061823, 0.036670, 0.036637, 0.039971, 0.039971, 0.040237, 0.038287, 0.040037])
    pcaF_tied_diag = np.array([0.366137, 0.134963, 0.099877, 0.106511, 0.080141, 0.079808, 0.086409, 0.086542])
    pca8_tied = np.array([0.056772, 0.050005, 0.041721, 0.043338, 0.046671, 0.076474, 0.061523, 0.060139])
    pca8_tied_diag = np.array([0.088459, 0.083542, 0.092976, 0.081658, 0.054989, 0.051822, 0.058356, 0.063273])
    gmm_plot(pcaF_em, pca8_em)
    gmm_plot(pcaF_diag, pca8_diag)
    gmm_plot(pcaF_tied, pca8_tied)
    gmm_plot(pcaF_tied_diag, pca8_tied_diag)
    gmm_plot2(pcaF_em, pcaF_diag, pcaF_tied, pcaF_tied_diag)
    gmm_plot2(pca8_em, pca8_diag, pca8_tied, pca8_tied_diag)

if __name__ == '__main__':
    trainX, trainL = load.load("data/Train")
    (XTR, LTR), (XTE, LTE) = preprocessing.split_dataset(trainX, trainL, 80)
    S1 = np.load("scores/GMM_4_tied.npy")
    S2 = np.load("scores/MVG_tied_ pca8.npy")
    optimal_decisions.det_plot([(S1, "Tied GMM, 4 components, no PCA"), (S2, "MVG, PCA (m = 8)")], LTE)
    # optimal_decisions.roc_plot(S1, LTE)

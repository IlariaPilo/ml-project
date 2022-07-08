import numpy as np
import matplotlib.pyplot as plt


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

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    pcaF_em = np.array([0.054989,0.046805,0.035020,0.046538,0.048455,0.078324,0.116378,0.211705])
    pcaF_diag = np.array([0.054989,0.046805,0.035020,0.046538,0.048455,0.078324,0.116378,0.211705])
    pca8_em = np.array([0.051789,0.044988,0.039971,0.038387,0.048255,0.058389,0.089776,0.121162])
    pca8_diag = np.array([0.051789,0.044988,0.039971,0.038387,0.048255,0.058389,0.089776,0.121162])
    gmm_plot(pcaF_em, pca8_em)
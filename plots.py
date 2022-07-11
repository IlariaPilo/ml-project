import numpy as np
import matplotlib as mlp
mlp.use('TkAgg')
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

    """plt.figure()
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
    plt.show()"""

    C = np.array([0.01, 0.1, 1, 10, 100])
    labels = [0.01, 0.1, 1, 10, 100]
    gaussF_pcaF_ptilde_0_1_pt_0_1 = np.array([0.151365, 0.151365, 0.154665, 0.146715, 0.191569])
    gaussF_pcaF_ptilde_0_1_pt_0_5 = np.array([0.149715, 0.141464, 0.136514, 0.146415, 0.906091])
    gaussF_pcaF_ptilde_0_1_pt_0_9 = np.array([0.176118, 0.156316, 0.153015, 0.184368, 0.320582])
    gaussF_pcaF_ptilde_0_5_pt_0_1 = np.array([0.061990, 0.061856, 0.060206, 0.063273, 0.068223])
    gaussF_pcaF_ptilde_0_5_pt_0_5 = np.array([0.058389, 0.060039, 0.058656, 0.058689, 0.728573])
    gaussF_pcaF_ptilde_0_5_pt_0_9 = np.array([0.058389, 0.061723, 0.061656, 0.063440, 0.143014])
    gaussF_pcaF_ptilde_0_9_pt_0_1 = np.array([0.115562, 0.107144, 0.118929, 0.150615, 0.164083])
    gaussF_pcaF_ptilde_0_9_pt_0_5 = np.array([0.134080, 0.122296, 0.116945, 0.108828, 0.870070])
    gaussF_pcaF_ptilde_0_9_pt_0_9 = np.array([0.143881, 0.121996, 0.127046, 0.145565, 0.399240])
    gaussF_pca8_ptilde_0_1_pt_0_1 = np.array([0.157966, 0.159616, 0.161266, 0.144764, 0.711371])
    gaussF_pca8_ptilde_0_1_pt_0_5 = np.array([0.148065, 0.146415, 0.149715, 0.167567, 0.470447])
    gaussF_pca8_ptilde_0_1_pt_0_9 = np.array([0.157966, 0.166217, 0.161566, 0.153315, 0.307081])
    gaussF_pca8_ptilde_0_5_pt_0_1 = np.array([0.058389, 0.058556, 0.058489, 0.056839, 0.572391])
    gaussF_pca8_ptilde_0_5_pt_0_5 = np.array([0.061623, 0.058322, 0.060039, 0.066540, 0.186719])
    gaussF_pca8_ptilde_0_5_pt_0_9 = np.array([0.063340, 0.064990, 0.063506, 0.061756, 0.111561])
    gaussF_pca8_ptilde_0_9_pt_0_1 = np.array([0.120312, 0.113878, 0.112195, 0.108828, 0.706471])
    gaussF_pca8_ptilde_0_9_pt_0_5 = np.array([0.117245, 0.113878, 0.118629, 0.159033, 0.473731])
    gaussF_pca8_ptilde_0_9_pt_0_9 = np.array([0.130713, 0.113878, 0.107144, 0.120612, 0.300930])

    plt.figure()
    plt.suptitle("Linear SVM results, varing C")
    plt.subplot(1, 2, 1)
    plt.title("No PCA")
    plt.semilogx(C, gaussF_pcaF_ptilde_0_1_pt_0_5, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, gaussF_pcaF_ptilde_0_5_pt_0_5, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, gaussF_pcaF_ptilde_0_9_pt_0_5, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim([0,0.5])
    plt.grid(visible=True, linestyle='--')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("PCA (m = 8)")
    plt.semilogx(C, gaussF_pca8_ptilde_0_1_pt_0_5, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, gaussF_pca8_ptilde_0_5_pt_0_5, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, gaussF_pca8_ptilde_0_9_pt_0_5, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim([0,0.5])
    plt.grid(visible=True, linestyle='--')
    plt.legend()
    plt.show()

def quadratic_svm_plot():
    labels = [0.0001, .001, 0.01, 0.1, 1, 10, 100]
    C = np.array(labels)
    gaussF_pcaF_ptilde_0_1_c_0 = np.array([0.484248, 0.522502, 0.588509, 0.923042, 1.000000, 0.995350, 1.000000])
    gaussF_pcaF_ptilde_0_5_c_0 = np.array([0.164966, 0.160016, 0.190486, 0.578124, 0.829016, 0.898523, 0.851302])
    gaussF_pcaF_ptilde_0_9_c_0 = np.array([0.425093, 0.427976, 0.410541, 0.900257, 0.967413, 0.994949, 1.000000])
    gaussF_pca8_ptilde_0_1_c_0 = np.array([0.857486, 0.841434, 0.987099, 0.989049, 0.993399, 1.000000, 0.998350])
    gaussF_pca8_ptilde_0_5_c_0 = np.array([0.335850, 0.334133, 0.424342, 0.621179, 0.838984, 0.983665, 0.943044])
    gaussF_pca8_ptilde_0_9_c_0 = np.array([0.678635, 0.694386, 0.763293, 0.876204, 1.000000, 0.996633, 1.000000])
    gaussF_pcaF_ptilde_0_1_c_1 = np.array([0.269127, 0.211971, 0.260426, 0.570957, 1.000000, 0.998350, 1.000000])
    gaussF_pcaF_ptilde_0_5_c_1 = np.array([0.063573, 0.056706, 0.073574, 0.216955, 0.774811, 0.838167, 0.779161])
    gaussF_pcaF_ptilde_0_9_c_1 = np.array([0.180618, 0.153082, 0.176651, 0.513651, 0.998316, 0.998316, 0.950878])
    gaussF_pca8_ptilde_0_1_c_1 = np.array([0.273177, 0.214071, 0.232073, 0.735374, 0.973597, 0.991749, 0.995050])
    gaussF_pca8_ptilde_0_5_c_1 = np.array([0.056839, 0.053339, 0.060206, 0.520419, 0.552772, 0.946578, 0.604260])
    gaussF_pca8_ptilde_0_9_c_1 = np.array([0.161016, 0.153382, 0.140814, 0.981481, 0.966030, 0.998316, 0.996633])

    plt.figure()
    plt.suptitle("Quadratic SVM results, varing C")
    plt.subplot(2, 2, 1)
    plt.title("No PCA, c = 0")
    plt.semilogx(C, gaussF_pcaF_ptilde_0_1_c_0, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, gaussF_pcaF_ptilde_0_5_c_0, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, gaussF_pcaF_ptilde_0_9_c_0, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim()
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.title("PCA (m = 8), c = 0")
    plt.semilogx(C, gaussF_pca8_ptilde_0_1_c_0, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, gaussF_pca8_ptilde_0_5_c_0, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, gaussF_pca8_ptilde_0_9_c_0, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim()
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.title("No PCA, c = 1")
    plt.semilogx(C, gaussF_pcaF_ptilde_0_1_c_1, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, gaussF_pcaF_ptilde_0_5_c_1, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, gaussF_pcaF_ptilde_0_9_c_1, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim()
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.title("PCA (m = 8), c = 1")
    plt.semilogx(C, gaussF_pca8_ptilde_0_1_c_1, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, gaussF_pca8_ptilde_0_5_c_1, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, gaussF_pca8_ptilde_0_9_c_1, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim()
    plt.legend()
    plt.show()

def radial_svm_plot():
    C = np.array([0.01, 0.1, 1, 10, 100])
    labels = [0.01, 0.1, 1, 10, 100]
    gaussF_pcaF_ptilde_0_1_l_0_1 = np.array([0.487699, 0.487699, 0.490999, 0.487249, 0.487249])
    gaussF_pcaF_ptilde_0_5_l_0_1 = np.array([0.091793, 0.091793, 0.093443, 0.091759, 0.091759])
    gaussF_pcaF_ptilde_0_9_l_0_1 = np.array([0.369654, 0.369654, 0.384505, 0.384505, 0.384505])
    gaussF_pca8_ptilde_0_1_l_0_1 = np.array([0.381788, 0.381788, 0.393639, 0.410441, 0.410441])
    gaussF_pca8_ptilde_0_5_l_0_1 = np.array([0.088492, 0.088492, 0.098527, 0.098527, 0.098527])
    gaussF_pca8_ptilde_0_9_l_0_1 = np.array([0.315482, 0.315482, 0.328049, 0.328049, 0.328049])
    gaussF_pcaF_ptilde_0_1_l_0_01 = np.array([0.335134, 0.161266, 0.170417, 0.193519, 0.191869])
    gaussF_pcaF_ptilde_0_5_l_0_01 = np.array([0.124862, 0.069740, 0.048221, 0.060106, 0.060106])
    gaussF_pcaF_ptilde_0_9_l_0_01 = np.array([0.305381, 0.154582, 0.128430, 0.143581, 0.141898])
    gaussF_pca8_ptilde_0_1_l_0_01 = np.array([0.351935, 0.194569, 0.181368, 0.260126, 0.292079])
    gaussF_pca8_ptilde_0_5_l_0_01 = np.array([0.126379, 0.068390, 0.060173, 0.063373, 0.068390])
    gaussF_pca8_ptilde_0_9_l_0_01 = np.array([0.305197, 0.181401, 0.188436, 0.220905, 0.213088])

    plt.figure()
    plt.suptitle("RBF SVM results, varing C")
    plt.subplot(2, 2, 1)
    plt.title("No PCA, "+r"""$\gamma = 0.1$""")
    plt.semilogx(C, gaussF_pcaF_ptilde_0_1_l_0_1, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, gaussF_pcaF_ptilde_0_5_l_0_1, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, gaussF_pcaF_ptilde_0_9_l_0_1, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim()
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.title("PCA (m = 8), "+r"""\gamma = 0.1$""")
    plt.semilogx(C, gaussF_pca8_ptilde_0_1_l_0_1, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, gaussF_pca8_ptilde_0_5_l_0_1, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, gaussF_pca8_ptilde_0_9_l_0_1, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim()
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.title("No PCA, "+r"""$\gamma = 0.01$""")
    plt.semilogx(C, gaussF_pcaF_ptilde_0_1_l_0_01, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, gaussF_pcaF_ptilde_0_5_l_0_01, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, gaussF_pcaF_ptilde_0_9_l_0_01, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim()
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.title("PCA (m = 8), "+r"""$\gamma = 0.01$""")
    plt.semilogx(C, gaussF_pca8_ptilde_0_1_l_0_01, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, gaussF_pca8_ptilde_0_5_l_0_01, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, gaussF_pca8_ptilde_0_9_l_0_01, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim()
    plt.legend()
    plt.show()

def gmm_plot(noPca, pca8, title, file_name):
    labels = [2,4,8,16,32,64,128,256]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    # plt.suptitle(title)
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

    plt.savefig(file_name)


def gmm_plot2(em, diag, tied, tied_diag, title, file_name):
    labels = [2,4,8,16,32,64,128,256]

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    # plt.suptitle(title)
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

    plt.savefig(file_name)


def gmm_plot_main():
    """
    Single fold
    pcaF_em = np.array([0.054989, 0.046805, 0.035020, 0.046538, 0.048455, 0.078324, 0.116378, 0.211705])
    pcaF_diag = np.array([0.371721, 0.138897, 0.107761, 0.099810, 0.091926, 0.085259, 0.093476, 0.125463])
    pca8_em = np.array([0.051789, 0.044988, 0.039971, 0.038387, 0.048255, 0.058389, 0.089776, 0.121162])
    pca8_diag = np.array([0.095260, 0.093443, 0.096943, 0.086609, 0.091492, 0.093343, 0.103210, 0.146615])
    pcaF_tied = np.array([0.061823, 0.036670, 0.036637, 0.039971, 0.039971, 0.040237, 0.038287, 0.040037])
    pcaF_tied_diag = np.array([0.366137, 0.134963, 0.099877, 0.106511, 0.080141, 0.079808, 0.086409, 0.086542])
    pca8_tied = np.array([0.056772, 0.050005, 0.041721, 0.043338, 0.046671, 0.076474, 0.061523, 0.060139])
    pca8_tied_diag = np.array([0.088459, 0.083542, 0.092976, 0.081658, 0.054989, 0.051822, 0.058356, 0.063273])
    """
    pcaF_em = np.array([0.043000, 0.032667, 0.032333, 0.044000, 0.056000, 0.087667, 0.150667, 0.313000])
    pcaF_diag = np.array([0.381667, 0.107667, 0.086667, 0.081000, 0.088000, 0.085667, 0.095667, 0.120000])
    pcaF_tied = np.array([0.048000, 0.029667, 0.030000, 0.031667, 0.032333, 0.036667, 0.040333, 0.044000])
    pcaF_tied_diag = np.array([0.380333, 0.104667, 0.082333, 0.083333, 0.072000, 0.067667, 0.069000, 0.070667])
    pca8_em = np.array([0.045000, 0.036000, 0.036333, 0.037667, 0.047667, 0.068000, 0.098667, 0.169333])
    pca8_diag = np.array([0.081333, 0.077333, 0.087000, 0.077333, 0.080000, 0.077333, 0.085000, 0.106000])
    pca8_tied = np.array([0.045667, 0.044667, 0.034333, 0.035667, 0.037333, 0.041667, 0.039667, 0.049667])
    pca8_tied_diag = np.array([0.081333, 0.073333, 0.081333, 0.067000, 0.061000, 0.056000, 0.056333, 0.060333])
    gmm_plot(pcaF_em, pca8_em, 'GMM results, Full Covariance', 'plots/gmm_em.png')
    gmm_plot(pcaF_diag, pca8_diag, 'GMM results, Diagonal Covariance', 'plots/gmm_diag.png')
    gmm_plot(pcaF_tied, pca8_tied, 'GMM results, Tied Full Covariance', 'plots/gmm_tied.png')
    gmm_plot(pcaF_tied_diag, pca8_tied_diag, 'GMM results, Tied Diagonal Covariance', 'plots/gmm_tied_diag.png')
    gmm_plot2(pcaF_em, pcaF_diag, pcaF_tied, pcaF_tied_diag, 'GMM results, no PCA', 'plots/gmm_pcaF.png')
    gmm_plot2(pca8_em, pca8_diag, pca8_tied, pca8_tied_diag, 'GMM results, PCA (m = 8)', 'plots/gmm_pca8.png')


def lr_plot():
    l = [10**(-6), 0.001, 0.1, 1, 10]
    """
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
    """
    gaussF_pcaF_pi5 = [0.047333, 0.047333, 0.049333, 0.068333, 0.148000]
    gaussF_pcaF_pi1 = [0.131333, 0.132000, 0.135667, 0.174333, 0.391667]
    gaussF_pcaF_pi9 = [0.125667, 0.124333, 0.129333, 0.157667, 0.363000]
    gaussF_pca8_pi5 = [0.047000, 0.046333, 0.049667, 0.070667, 0.149000]
    gaussF_pca8_pi1 = [0.136000, 0.135333, 0.134667, 0.173333, 0.396667]
    gaussF_pca8_pi9 = [0.122333, 0.123000, 0.126000, 0.160667, 0.368000]
    gaussT_pcaF_pi5 = [0.056000, 0.059333, 0.145333, 0.282000, 0.506000]
    gaussT_pcaF_pi1 = [0.162000, 0.172667, 0.370667, 0.639333, 0.793000]
    gaussT_pcaF_pi9 = [0.158000, 0.167667, 0.333000, 0.667000, 0.805667]
    gaussT_pca8_pi5 = [0.164000, 0.164667, 0.202667, 0.316000, 0.522333]
    gaussT_pca8_pi1 = [0.437667, 0.438333, 0.516000, 0.691333, 0.826333]
    gaussT_pca8_pi9 = [0.416333, 0.409333, 0.469000, 0.720667, 0.843000]

    plt.figure()
    plt.suptitle("Logistic regression results")
    plt.subplot(2, 2, 1)
    plt.title("Raw features, no PCA")
    plt.semilogx(l, gaussF_pcaF_pi5, 'r', label=r'$\tilde{\pi} = 0.5$')
    plt.semilogx(l, gaussF_pcaF_pi1, 'b', label=r'$\tilde{\pi} = 0.1$')
    plt.semilogx(l, gaussF_pcaF_pi9, 'g', label=r'$\tilde{\pi} = 0.9$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("minDCF")
    plt.grid(visible=True, linestyle='--')
    plt.xticks(l)
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.title("Raw features, PCA (m = 8)")
    plt.semilogx(l, gaussF_pca8_pi5, 'r', label=r'$\tilde{\pi} = 0.5$')
    plt.semilogx(l, gaussF_pca8_pi1, 'b', label=r'$\tilde{\pi} = 0.1$')
    plt.semilogx(l, gaussF_pca8_pi9, 'g', label=r'$\tilde{\pi} = 0.9$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("minDCF")
    plt.grid(visible=True, linestyle='--')
    plt.xticks(l)
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.title("Gaussianized features, no PCA")
    plt.semilogx(l, gaussT_pcaF_pi5, 'r', label=r'$\tilde{\pi} = 0.5$')
    plt.semilogx(l, gaussT_pcaF_pi1, 'b', label=r'$\tilde{\pi} = 0.1$')
    plt.semilogx(l, gaussT_pcaF_pi9, 'g', label=r'$\tilde{\pi} = 0.9$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("minDCF")
    plt.grid(visible=True, linestyle='--')
    plt.xticks(l)
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.title("Gaussianized features, PCA (m = 8)")
    plt.semilogx(l, gaussT_pca8_pi5, 'r', label=r'$\tilde{\pi} = 0.5$')
    plt.semilogx(l, gaussT_pca8_pi1, 'b', label=r'$\tilde{\pi} = 0.1$')
    plt.semilogx(l, gaussT_pca8_pi9, 'g', label=r'$\tilde{\pi} = 0.9$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("minDCF")
    plt.grid(visible=True, linestyle='--')
    plt.xticks(l)
    plt.legend()
    plt.show()


def quad_lr_plot():
    l = [10**(-15), 10**(-11), 10**(-9), 10**(-6), 0.001, 0.1, 1, 10]

    gaussF_pcaF_pi5 = [0.121667, 0.118333, 0.122333, 0.120000, 0.118333, 0.119333, 0.113333, 0.134667]
    gaussF_pcaF_pi1 = [0.354000, 0.371000, 0.378333, 0.367333, 0.379333, 0.352000, 0.373000, 0.402333]
    gaussF_pcaF_pi9 = [0.333667, 0.335333, 0.331000, 0.329333, 0.324333, 0.312667, 0.313333, 0.325333]
    gaussF_pca8_pi5 = [0.079333, 0.065667, 0.075000, 0.081333, 0.083667, 0.077667, 0.103333, 0.219000]
    gaussF_pca8_pi1 = [0.229667, 0.204000, 0.213333, 0.222667, 0.236667, 0.230333, 0.300000, 0.581000]
    gaussF_pca8_pi9 = [0.219000, 0.190667, 0.199333, 0.196333, 0.223667, 0.189000, 0.264667, 0.472000]

    plt.figure()
    plt.suptitle("Quadratic logistic regression results")
    plt.subplot(1, 2, 1)
    plt.title("No PCA")
    plt.semilogx(l, gaussF_pcaF_pi5, 'r', label=r'$\tilde{\pi} = 0.5$')
    plt.semilogx(l, gaussF_pcaF_pi1, 'b', label=r'$\tilde{\pi} = 0.1$')
    plt.semilogx(l, gaussF_pcaF_pi9, 'g', label=r'$\tilde{\pi} = 0.9$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("minDCF")
    plt.grid(visible=True, linestyle='--')
    plt.xticks(l)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("PCA (m = 8)")
    plt.semilogx(l, gaussF_pca8_pi5, 'r', label=r'$\tilde{\pi} = 0.5$')
    plt.semilogx(l, gaussF_pca8_pi1, 'b', label=r'$\tilde{\pi} = 0.1$')
    plt.semilogx(l, gaussF_pca8_pi9, 'g', label=r'$\tilde{\pi} = 0.9$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("minDCF")
    plt.grid(visible=True, linestyle='--')
    plt.xticks(l)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    """
    trainX, trainL = load.load("data/Train")
    foldsX, foldsL = preprocessing.k_fold(5, trainX, trainL)
    trueL = np.hstack(foldsL)
    S1 = np.load("scores/GMM_4_tied.npy")
    S2 = np.load("scores/MVG_tied_pca8.npy")
    # optimal_decisions.det_plot([(S1, "Tied GMM, 4 components, no PCA"), (S2, "MVG, PCA (m = 8)")], trueL)
    # optimal_decisions.roc_plot(S1, LTE)
    optimal_decisions.bayes_error_plot([(S1, "GMM", 'r'), (S2, "MVG", 'b')], trueL)
    """
    # quad_lr_plot()
    # linear_svm_plot()
    quadratic_svm_plot()
    # radial_svm_plot()

import numpy as np
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt


import load
import optimal_decisions
import preprocessing


def ev_linear_poly_svm_plot():

    # linear scores
    C = np.array([0.01, 0.1, 1, 10, 100])
    labels = [0.01, 0.1, 1, 10, 100]
    pcaF_ptilde_0_1_pt_0_5 = np.array([0.149715, 0.141464, 0.136514, 0.146415, 0.906091])
    pcaF_ptilde_0_5_pt_0_5 = np.array([0.058389, 0.060039, 0.058656, 0.058689, 0.728573])
    pcaF_ptilde_0_9_pt_0_5 = np.array([0.134080, 0.122296, 0.116945, 0.108828, 0.870070])
    pca8_ptilde_0_1_pt_0_5 = np.array([0.148065, 0.146415, 0.149715, 0.167567, 0.470447])
    pca8_ptilde_0_5_pt_0_5 = np.array([0.061623, 0.058322, 0.060039, 0.066540, 0.186719])
    pca8_ptilde_0_9_pt_0_5 = np.array([0.117245, 0.113878, 0.118629, 0.159033, 0.473731])
    ev_pcaF_ptilde_0_1_pt_0_5 = np.array([0.136500, 0.136000, 0.138000, 0.141000, 0.699000])
    ev_pcaF_ptilde_0_5_pt_0_5 = np.array([0.054000, 0.052500, 0.052000, 0.057000, 0.566000])
    ev_pcaF_ptilde_0_9_pt_0_5 = np.array([0.130500, 0.135500, 0.135000, 0.151000, 0.771000])
    ev_pca8_ptilde_0_1_pt_0_5 = np.array([0.140500, 0.143500, 0.139000, 0.174000, 0.733500])
    ev_pca8_ptilde_0_5_pt_0_5 = np.array([0.056000, 0.054000, 0.052500, 0.064500, 0.362000])
    ev_pca8_ptilde_0_9_pt_0_5 = np.array([0.135000, 0.136000, 0.139000, 0.179000, 0.746000])

    # poly scores
    labels2 = [0.001, 0.01, 0.1, 1, 10]
    C2 = np.array(labels2)
    pcaF_ptilde_0_1_c_0 = np.array([0.522502, 0.588509, 0.923042, 1.000000, 0.995350])
    pcaF_ptilde_0_5_c_0 = np.array([0.160016, 0.190486, 0.578124, 0.829016, 0.898523])
    pcaF_ptilde_0_9_c_0 = np.array([0.427976, 0.410541, 0.900257, 0.967413, 0.994949])
    pca8_ptilde_0_1_c_0 = np.array([0.841434, 0.987099, 0.989049, 0.993399, 1.000000])
    pca8_ptilde_0_5_c_0 = np.array([0.334133, 0.424342, 0.621179, 0.838984, 0.983665])
    pca8_ptilde_0_9_c_0 = np.array([0.694386, 0.763293, 0.876204, 1.000000, 0.996633])
    pcaF_ptilde_0_1_c_1 = np.array([0.211971, 0.260426, 0.570957, 1.000000, 0.998350])
    pcaF_ptilde_0_5_c_1 = np.array([0.056706, 0.073574, 0.216955, 0.774811, 0.838167])
    pcaF_ptilde_0_9_c_1 = np.array([0.153082, 0.176651, 0.513651, 0.998316, 0.998316])
    pca8_ptilde_0_1_c_1 = np.array([0.214071, 0.232073, 0.735374, 0.973597, 0.991749])
    pca8_ptilde_0_5_c_1 = np.array([0.053339, 0.060206, 0.520419, 0.552772, 0.946578])
    pca8_ptilde_0_9_c_1 = np.array([0.153382, 0.140814, 0.981481, 0.966030, 0.998316])
    ev_pcaF_ptilde_0_5_c_0 = np.array([0.179500, 0.213000, 0.636500, 0.823500, 0.917500])
    ev_pcaF_ptilde_0_1_c_0 = np.array([0.517500, 0.662500, 0.987000, 0.984000, 1.000000])
    ev_pcaF_ptilde_0_9_c_0 = np.array([0.420500, 0.475000, 0.944000, 0.998500, 1.000000])
    ev_pca8_ptilde_0_5_c_0 = np.array([0.341000, 0.668000, 0.760000, 0.875500, 0.856500])
    ev_pca8_ptilde_0_1_c_0 = np.array([0.868000, 0.997500, 0.999000, 0.998000, 0.993500])
    ev_pca8_ptilde_0_9_c_0 = np.array([0.713500, 0.869000, 0.991500, 0.999500, 0.998500])
    ev_pcaF_ptilde_0_5_c_1 = np.array([0.056000, 0.068500, 0.317000, 0.726500, 0.780500])
    ev_pcaF_ptilde_0_1_c_1 = np.array([0.176000, 0.190500, 0.654500, 0.997000, 0.999000])
    ev_pcaF_ptilde_0_9_c_1 = np.array([0.146500, 0.161500, 0.800500, 0.986500, 0.999500])
    ev_pca8_ptilde_0_5_c_1 = np.array([0.061000, 0.104000, 0.489500, 0.870000, 0.886500])
    ev_pca8_ptilde_0_1_c_1 = np.array([0.160500, 0.302500, 0.894000, 0.999000, 0.999000])
    ev_pca8_ptilde_0_9_c_1 = np.array([0.136000, 0.264000, 0.967500, 0.999000, 0.999500])

    plt.figure()
    plt.suptitle("Linear and polynomial SVM results")
    plt.subplot(2, 2, 1)
    plt.title("Linear SVM, No PCA")
    plt.semilogx(C, pcaF_ptilde_0_1_pt_0_5, 'b--', label=r"""$\~{\pi} = 0.1$ [Val]""")
    plt.semilogx(C, pcaF_ptilde_0_5_pt_0_5, 'r--', label=r"""$\~{\pi} = 0.5$ [Val]""")
    plt.semilogx(C, pcaF_ptilde_0_9_pt_0_5, 'g--', label=r"""$\~{\pi} = 0.9$ [Val]""")
    plt.semilogx(C, ev_pcaF_ptilde_0_1_pt_0_5, 'b', label=r"""$\~{\pi} = 0.1$ [Eval]""")
    plt.semilogx(C, ev_pcaF_ptilde_0_5_pt_0_5, 'r', label=r"""$\~{\pi} = 0.5$ [Eval]""")
    plt.semilogx(C, ev_pcaF_ptilde_0_9_pt_0_5, 'g', label=r"""$\~{\pi} = 0.9$ [Eval]""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim([0,0.5])
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.title("Linear SVM, PCA (m = 8)")
    plt.semilogx(C, pca8_ptilde_0_1_pt_0_5, 'b--', label=r"""$\~{\pi} = 0.1$ [Val]""")
    plt.semilogx(C, pca8_ptilde_0_5_pt_0_5, 'r--', label=r"""$\~{\pi} = 0.5$ [Val]""")
    plt.semilogx(C, pca8_ptilde_0_9_pt_0_5, 'g--', label=r"""$\~{\pi} = 0.9$ [Val]""")
    plt.semilogx(C, ev_pca8_ptilde_0_1_pt_0_5, 'b', label=r"""$\~{\pi} = 0.1$ [Eval]""")
    plt.semilogx(C, ev_pca8_ptilde_0_5_pt_0_5, 'r', label=r"""$\~{\pi} = 0.5$ [Eval]""")
    plt.semilogx(C, ev_pca8_ptilde_0_9_pt_0_5, 'g', label=r"""$\~{\pi} = 0.9$ [Eval]""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim([0,0.5])
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.title("Polynomial SVM, No PCA")
    plt.semilogx(C2, pcaF_ptilde_0_1_c_1, 'b--', label=r"""$\~{\pi} = 0.1$ [Val]""")
    plt.semilogx(C2, pcaF_ptilde_0_5_c_1, 'r--', label=r"""$\~{\pi} = 0.5$ [Val]""")
    plt.semilogx(C2, pcaF_ptilde_0_9_c_1, 'g--', label=r"""$\~{\pi} = 0.9$ [Val]""")
    plt.semilogx(C2, ev_pcaF_ptilde_0_1_c_1, 'b', label=r"""$\~{\pi} = 0.1$ [Eval]""")
    plt.semilogx(C2, ev_pcaF_ptilde_0_5_c_1, 'r', label=r"""$\~{\pi} = 0.5$ [Eval]""")
    plt.semilogx(C2, ev_pcaF_ptilde_0_9_c_1, 'g', label=r"""$\~{\pi} = 0.9$ [Eval]""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels2)
    plt.ylim()
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.title("Polynomial SVM, PCA (m = 8)")
    plt.semilogx(C2, pca8_ptilde_0_1_c_1, 'b--', label=r"""$\~{\pi} = 0.1$ [Val]""")
    plt.semilogx(C2, pca8_ptilde_0_5_c_1, 'r--', label=r"""$\~{\pi} = 0.5$ [Val]""")
    plt.semilogx(C2, pca8_ptilde_0_9_c_1, 'g--', label=r"""$\~{\pi} = 0.9$ [Val]""")
    plt.semilogx(C2, ev_pca8_ptilde_0_1_c_1, 'b', label=r"""$\~{\pi} = 0.1$ [Eval]""")
    plt.semilogx(C2, ev_pca8_ptilde_0_5_c_1, 'r', label=r"""$\~{\pi} = 0.5$ [Eval]""")
    plt.semilogx(C2, ev_pca8_ptilde_0_9_c_1, 'g', label=r"""$\~{\pi} = 0.9$ [Eval]""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels2)
    plt.ylim()
    plt.legend()
    plt.show()

def quadratic_svm_plot():
    labels = [0.0001, .001, 0.01, 0.1, 1, 10]
    C = np.array(labels)

    pcaF_ptilde_0_1_c_0 = np.array([0.484248, 0.522502, 0.588509, 0.923042, 1.000000, 0.995350])
    pcaF_ptilde_0_5_c_0 = np.array([0.164966, 0.160016, 0.190486, 0.578124, 0.829016, 0.898523])
    pcaF_ptilde_0_9_c_0 = np.array([0.425093, 0.427976, 0.410541, 0.900257, 0.967413, 0.994949])
    pca8_ptilde_0_1_c_0 = np.array([0.857486, 0.841434, 0.987099, 0.989049, 0.993399, 1.000000])
    pca8_ptilde_0_5_c_0 = np.array([0.335850, 0.334133, 0.424342, 0.621179, 0.838984, 0.983665])
    pca8_ptilde_0_9_c_0 = np.array([0.678635, 0.694386, 0.763293, 0.876204, 1.000000, 0.996633])
    pcaF_ptilde_0_1_c_1 = np.array([0.269127, 0.211971, 0.260426, 0.570957, 1.000000, 0.998350])
    pcaF_ptilde_0_5_c_1 = np.array([0.063573, 0.056706, 0.073574, 0.216955, 0.774811, 0.838167])
    pcaF_ptilde_0_9_c_1 = np.array([0.180618, 0.153082, 0.176651, 0.513651, 0.998316, 0.998316])
    pca8_ptilde_0_1_c_1 = np.array([0.273177, 0.214071, 0.232073, 0.735374, 0.973597, 0.991749])
    pca8_ptilde_0_5_c_1 = np.array([0.056839, 0.053339, 0.060206, 0.520419, 0.552772, 0.946578])
    pca8_ptilde_0_9_c_1 = np.array([0.161016, 0.153382, 0.140814, 0.981481, 0.966030, 0.998316])


    plt.figure()
    plt.suptitle("Quadratic SVM results, varing C")
    plt.subplot(2, 2, 1)
    plt.title("No PCA, c = 0")
    plt.semilogx(C, pcaF_ptilde_0_1_c_0, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, pcaF_ptilde_0_5_c_0, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, pcaF_ptilde_0_9_c_0, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim()
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.title("PCA (m = 8), c = 0")
    plt.semilogx(C, pca8_ptilde_0_1_c_0, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, pca8_ptilde_0_5_c_0, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, pca8_ptilde_0_9_c_0, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim()
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.title("No PCA, c = 1")
    plt.semilogx(C, pcaF_ptilde_0_1_c_1, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, pcaF_ptilde_0_5_c_1, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, pcaF_ptilde_0_9_c_1, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim()
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.title("PCA (m = 8), c = 1")
    plt.semilogx(C, pca8_ptilde_0_1_c_1, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, pca8_ptilde_0_5_c_1, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, pca8_ptilde_0_9_c_1, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim()
    plt.legend()
    plt.show()

def radial_svm_plot():
    C = np.array([0.01, 0.1, 1, 10, 100])
    labels = [0.01, 0.1, 1, 10, 100]
    pcaF_ptilde_0_1_l_0_1 = np.array([0.487699, 0.487699, 0.490999, 0.487249, 0.487249])
    pcaF_ptilde_0_5_l_0_1 = np.array([0.091793, 0.091793, 0.093443, 0.091759, 0.091759])
    pcaF_ptilde_0_9_l_0_1 = np.array([0.369654, 0.369654, 0.384505, 0.384505, 0.384505])
    pca8_ptilde_0_1_l_0_1 = np.array([0.381788, 0.381788, 0.393639, 0.410441, 0.410441])
    pca8_ptilde_0_5_l_0_1 = np.array([0.088492, 0.088492, 0.098527, 0.098527, 0.098527])
    pca8_ptilde_0_9_l_0_1 = np.array([0.315482, 0.315482, 0.328049, 0.328049, 0.328049])
    pcaF_ptilde_0_1_l_0_01 = np.array([0.335134, 0.161266, 0.170417, 0.193519, 0.191869])
    pcaF_ptilde_0_5_l_0_01 = np.array([0.124862, 0.069740, 0.048221, 0.060106, 0.060106])
    pcaF_ptilde_0_9_l_0_01 = np.array([0.305381, 0.154582, 0.128430, 0.143581, 0.141898])
    pca8_ptilde_0_1_l_0_01 = np.array([0.351935, 0.194569, 0.181368, 0.260126, 0.292079])
    pca8_ptilde_0_5_l_0_01 = np.array([0.126379, 0.068390, 0.060173, 0.063373, 0.068390])
    pca8_ptilde_0_9_l_0_01 = np.array([0.305197, 0.181401, 0.188436, 0.220905, 0.213088])

    plt.figure()
    plt.suptitle("RBF SVM results, varing C")
    plt.subplot(2, 2, 1)
    plt.title("No PCA, "+r"""$\gamma = 0.1$""")
    plt.semilogx(C, pcaF_ptilde_0_1_l_0_1, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, pcaF_ptilde_0_5_l_0_1, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, pcaF_ptilde_0_9_l_0_1, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim()
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.title("PCA (m = 8), "+r"""$\gamma = 0.1$""")
    plt.semilogx(C, pca8_ptilde_0_1_l_0_1, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, pca8_ptilde_0_5_l_0_1, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, pca8_ptilde_0_9_l_0_1, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim()
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.title("No PCA, "+r"""$\gamma = 0.01$""")
    plt.semilogx(C, pcaF_ptilde_0_1_l_0_01, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, pcaF_ptilde_0_5_l_0_01, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, pcaF_ptilde_0_9_l_0_01, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim()
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.title("PCA (m = 8), "+r"""$\gamma = 0.01$""")
    plt.semilogx(C, pca8_ptilde_0_1_l_0_01, 'b', label=r"""$\~{\pi} = 0.1$""")
    plt.semilogx(C, pca8_ptilde_0_5_l_0_01, 'r', label=r"""$\~{\pi} = 0.5$""")
    plt.semilogx(C, pca8_ptilde_0_9_l_0_01, 'g', label=r"""$\~{\pi} = 0.9$""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(labels)
    plt.ylim()
    plt.legend()
    plt.show()


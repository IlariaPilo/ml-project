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
    """# old scores (100%)
    ev_pcaF_ptilde_0_1_pt_0_5 = np.array([0.136500, 0.136000, 0.138000, 0.141000, 0.699000])
    ev_pcaF_ptilde_0_5_pt_0_5 = np.array([0.054000, 0.052500, 0.052000, 0.057000, 0.566000])
    ev_pcaF_ptilde_0_9_pt_0_5 = np.array([0.130500, 0.135500, 0.135000, 0.151000, 0.771000])
    ev_pca8_ptilde_0_1_pt_0_5 = np.array([0.140500, 0.143500, 0.139000, 0.174000, 0.733500])
    ev_pca8_ptilde_0_5_pt_0_5 = np.array([0.056000, 0.054000, 0.052500, 0.064500, 0.362000])
    ev_pca8_ptilde_0_9_pt_0_5 = np.array([0.135000, 0.136000, 0.139000, 0.179000, 0.746000])"""
    ev_pcaF_ptilde_0_1_pt_0_5 = np.array([0.144500, 0.137500, 0.140000, 0.150000, 0.939000])
    ev_pcaF_ptilde_0_5_pt_0_5 = np.array([0.055500, 0.051000, 0.051500, 0.058000, 0.732000])
    ev_pcaF_ptilde_0_9_pt_0_5 = np.array([0.136000, 0.131500, 0.139000, 0.166500, 0.894000])
    ev_pca8_ptilde_0_1_pt_0_5 = np.array([0.139000, 0.146000, 0.148000, 0.157000, 0.468500])
    ev_pca8_ptilde_0_5_pt_0_5 = np.array([0.055500, 0.052500, 0.054000, 0.065000, 0.232000])
    ev_pca8_ptilde_0_9_pt_0_5 = np.array([0.127000, 0.136000, 0.141000, 0.163000, 0.513000])

    # poly scores
    labels2 = [0.001, 0.01, 0.1, 1, 10]
    C2 = np.array(labels2)
    pcaF_ptilde_0_1_c_1 = np.array([0.211971, 0.260426, 0.570957, 1.000000, 0.998350])
    pcaF_ptilde_0_5_c_1 = np.array([0.056706, 0.073574, 0.216955, 0.774811, 0.838167])
    pcaF_ptilde_0_9_c_1 = np.array([0.153082, 0.176651, 0.513651, 0.998316, 0.998316])
    pca8_ptilde_0_1_c_1 = np.array([0.214071, 0.232073, 0.735374, 0.973597, 0.991749])
    pca8_ptilde_0_5_c_1 = np.array([0.053339, 0.060206, 0.520419, 0.552772, 0.946578])
    pca8_ptilde_0_9_c_1 = np.array([0.153382, 0.140814, 0.981481, 0.966030, 0.998316])
    """ # old scores (100%)
    ev_pcaF_ptilde_0_5_c_1 = np.array([0.056000, 0.068500, 0.317000, 0.726500, 0.780500])
    ev_pcaF_ptilde_0_1_c_1 = np.array([0.176000, 0.190500, 0.654500, 0.997000, 0.999000])
    ev_pcaF_ptilde_0_9_c_1 = np.array([0.146500, 0.161500, 0.800500, 0.986500, 0.999500])
    ev_pca8_ptilde_0_5_c_1 = np.array([0.061000, 0.104000, 0.489500, 0.870000, 0.886500])
    ev_pca8_ptilde_0_1_c_1 = np.array([0.160500, 0.302500, 0.894000, 0.999000, 0.999000])
    ev_pca8_ptilde_0_9_c_1 = np.array([0.136000, 0.264000, 0.967500, 0.999000, 0.999500])"""
    ev_pcaF_ptilde_0_5_c_1 = np.array([0.060000, 0.072000, 0.238500, 0.812500, 0.828000])
    ev_pcaF_ptilde_0_1_c_1 = np.array([0.192000, 0.265500, 0.647000, 0.999000, 0.999500])
    ev_pcaF_ptilde_0_9_c_1 = np.array([0.147500, 0.156500, 0.555500, 1.000000, 1.000000])
    ev_pca8_ptilde_0_5_c_1 = np.array([0.059500, 0.056500, 0.448000, 0.593000, 0.979000])
    ev_pca8_ptilde_0_1_c_1 = np.array([0.149000, 0.181500, 0.899000, 0.979500, 0.997500])
    ev_pca8_ptilde_0_9_c_1 = np.array([0.146000, 0.150500, 0.596000, 0.993500, 1.000000])


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


def ev_radial_svm_plot():
    C = [0.01, 0.1, 1, 10, 100]
    pcaF_ptilde_0_1_l_0_01 = np.array([0.335134, 0.161266, 0.170417, 0.193519, 0.191869])
    pcaF_ptilde_0_5_l_0_01 = np.array([0.124862, 0.069740, 0.048221, 0.060106, 0.060106])
    pcaF_ptilde_0_9_l_0_01 = np.array([0.305381, 0.154582, 0.128430, 0.143581, 0.141898])
    pca8_ptilde_0_1_l_0_01 = np.array([0.351935, 0.194569, 0.181368, 0.260126, 0.292079])
    pca8_ptilde_0_5_l_0_01 = np.array([0.126379, 0.068390, 0.060173, 0.063373, 0.068390])
    pca8_ptilde_0_9_l_0_01 = np.array([0.305197, 0.181401, 0.188436, 0.220905, 0.213088])
    ev_pcaF_ptilde_0_1_l_0_01 = np.array([0.332000, 0.171000, 0.112000, 0.147000, 0.145500])
    ev_pcaF_ptilde_0_5_l_0_01 = np.array([0.109000, 0.058000, 0.050000, 0.055000, 0.055000])
    ev_pcaF_ptilde_0_9_l_0_01 = np.array([0.255000, 0.169000, 0.129000, 0.167500, 0.167500])
    ev_pca8_ptilde_0_1_l_0_01 = np.array([0.312500, 0.168000, 0.135500, 0.190500, 0.226500])
    ev_pca8_ptilde_0_5_l_0_01 = np.array([0.106000, 0.059500, 0.053500, 0.070500, 0.082000])
    ev_pca8_ptilde_0_9_l_0_01 = np.array([0.258500, 0.186000, 0.154000, 0.204000, 0.244500])

    plt.figure()
    plt.suptitle("RBF SVM results")
    plt.subplot(1, 2, 1)
    plt.title("No PCA, "+r"""$\gamma = 0.01$""")
    plt.semilogx(C, pcaF_ptilde_0_1_l_0_01, 'b--', label=r"""$\~{\pi} = 0.1$ [Val]""")
    plt.semilogx(C, pcaF_ptilde_0_5_l_0_01, 'r--', label=r"""$\~{\pi} = 0.5$ [Val]""")
    plt.semilogx(C, pcaF_ptilde_0_9_l_0_01, 'g--', label=r"""$\~{\pi} = 0.9$ [Val]""")
    plt.semilogx(C, ev_pcaF_ptilde_0_1_l_0_01, 'b', label=r"""$\~{\pi} = 0.1$ [Eval]""")
    plt.semilogx(C, ev_pcaF_ptilde_0_5_l_0_01, 'r', label=r"""$\~{\pi} = 0.5$ [Eval]""")
    plt.semilogx(C, ev_pcaF_ptilde_0_9_l_0_01, 'g', label=r"""$\~{\pi} = 0.9$ [Eval]""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(C)
    plt.ylim()
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("PCA (m = 8), "+r"""$\gamma = 0.01$""")
    plt.semilogx(C, pca8_ptilde_0_1_l_0_01, 'b--', label=r"""$\~{\pi} = 0.1$ [Val]""")
    plt.semilogx(C, pca8_ptilde_0_5_l_0_01, 'r--', label=r"""$\~{\pi} = 0.5$ [Val]""")
    plt.semilogx(C, pca8_ptilde_0_9_l_0_01, 'g--', label=r"""$\~{\pi} = 0.9$ [Val]""")
    plt.semilogx(C, ev_pca8_ptilde_0_1_l_0_01, 'b', label=r"""$\~{\pi} = 0.1$ [Eval]""")
    plt.semilogx(C, ev_pca8_ptilde_0_5_l_0_01, 'r', label=r"""$\~{\pi} = 0.5$ [Eval]""")
    plt.semilogx(C, ev_pca8_ptilde_0_9_l_0_01, 'g', label=r"""$\~{\pi} = 0.9$ [Eval]""")
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xticks(C)
    plt.ylim()
    plt.legend()
    plt.show()

def lr_plot():
    l = [10**(-6), 0.001, 0.1, 1, 10]

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

    gaussF_pcaF_pi5_ev = [0.052500, 0.052500, 0.054000, 0.072000, 0.154500]
    gaussF_pcaF_pi1_ev = [0.135000, 0.132000, 0.148500, 0.194500, 0.389500]
    gaussF_pcaF_pi9_ev = [0.133000, 0.134500, 0.141500, 0.200000, 0.416500]
    gaussF_pca8_pi5_ev = [0.052500, 0.053000, 0.054500, 0.073000, 0.156000]
    gaussF_pca8_pi1_ev = [0.144500, 0.145500, 0.153000, 0.198000, 0.392000]
    gaussF_pca8_pi9_ev = [0.136000, 0.135500, 0.142500, 0.208500, 0.413000]
    gaussT_pcaF_pi5_ev = [0.059000, 0.065500, 0.140000, 0.280000, 0.513000]
    gaussT_pcaF_pi1_ev = [0.161500, 0.171000, 0.404000, 0.664500, 0.777500]
    gaussT_pcaF_pi9_ev = [0.164500, 0.177000, 0.351500, 0.656500, 0.839000]
    gaussT_pca8_pi5_ev = [0.185500, 0.185500, 0.206500, 0.312000, 0.530500]
    gaussT_pca8_pi1_ev = [0.465000, 0.468500, 0.527500, 0.719000, 0.810000]
    gaussT_pca8_pi9_ev = [0.434500, 0.440500, 0.502000, 0.721500, 0.869500]

    plt.figure()
    plt.suptitle("Logistic regression comparison")
    plt.subplot(2, 2, 1)
    plt.title("Raw features, no PCA")
    plt.semilogx(l, gaussF_pcaF_pi5, '--r', label=r'$\tilde{\pi} = 0.5$ [Val]')
    plt.semilogx(l, gaussF_pcaF_pi1, '--b', label=r'$\tilde{\pi} = 0.1$ [Val]')
    plt.semilogx(l, gaussF_pcaF_pi9, '--g', label=r'$\tilde{\pi} = 0.9$ [Val]')
    plt.semilogx(l, gaussF_pcaF_pi5_ev, 'r', label=r'$\tilde{\pi} = 0.5$ [Eval]')
    plt.semilogx(l, gaussF_pcaF_pi1_ev, 'b', label=r'$\tilde{\pi} = 0.1$ [Eval]')
    plt.semilogx(l, gaussF_pcaF_pi9_ev, 'g', label=r'$\tilde{\pi} = 0.9$ [Eval]')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("minDCF")
    plt.grid(visible=True, linestyle='--')
    plt.xticks(l)
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.title("Raw features, PCA (m = 8)")
    plt.semilogx(l, gaussF_pca8_pi5, '--r', label=r'$\tilde{\pi} = 0.5$ [Val]')
    plt.semilogx(l, gaussF_pca8_pi1, '--b', label=r'$\tilde{\pi} = 0.1$ [Val]')
    plt.semilogx(l, gaussF_pca8_pi9, '--g', label=r'$\tilde{\pi} = 0.9$ [Val]')
    plt.semilogx(l, gaussF_pca8_pi5_ev, 'r', label=r'$\tilde{\pi} = 0.5$ [Eval]')
    plt.semilogx(l, gaussF_pca8_pi1_ev, 'b', label=r'$\tilde{\pi} = 0.1$ [Eval]')
    plt.semilogx(l, gaussF_pca8_pi9_ev, 'g', label=r'$\tilde{\pi} = 0.9$ [Eval]')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("minDCF")
    plt.grid(visible=True, linestyle='--')
    plt.xticks(l)
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.title("Gaussianized features, no PCA")
    plt.semilogx(l, gaussT_pcaF_pi5, '--r', label=r'$\tilde{\pi} = 0.5$ [Val]')
    plt.semilogx(l, gaussT_pcaF_pi1, '--b', label=r'$\tilde{\pi} = 0.1$ [Val]')
    plt.semilogx(l, gaussT_pcaF_pi9, '--g', label=r'$\tilde{\pi} = 0.9$ [Val]')
    plt.semilogx(l, gaussT_pcaF_pi5_ev, 'r', label=r'$\tilde{\pi} = 0.5$ [Eval]')
    plt.semilogx(l, gaussT_pcaF_pi1_ev, 'b', label=r'$\tilde{\pi} = 0.1$ [Eval]')
    plt.semilogx(l, gaussT_pcaF_pi9_ev, 'g', label=r'$\tilde{\pi} = 0.9$ [Eval]')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("minDCF")
    plt.grid(visible=True, linestyle='--')
    plt.xticks(l)
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.title("Gaussianized features, PCA (m = 8)")
    plt.semilogx(l, gaussT_pca8_pi5, '--r', label=r'$\tilde{\pi} = 0.5$ [Val]')
    plt.semilogx(l, gaussT_pca8_pi1, '--b', label=r'$\tilde{\pi} = 0.1$ [Val]')
    plt.semilogx(l, gaussT_pca8_pi9, '--g', label=r'$\tilde{\pi} = 0.9$ [Val]')
    plt.semilogx(l, gaussT_pca8_pi5_ev, 'r', label=r'$\tilde{\pi} = 0.5$ [Eval]')
    plt.semilogx(l, gaussT_pca8_pi1_ev, 'b', label=r'$\tilde{\pi} = 0.1$ [Eval]')
    plt.semilogx(l, gaussT_pca8_pi9_ev, 'g', label=r'$\tilde{\pi} = 0.9$ [Eval]')
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

    gaussF_pcaF_pi5_ev = [0.117000, 0.118000, 0.124500, 0.106000, 0.107000, 0.119500, 0.121000, 0.151500]
    gaussF_pcaF_pi1_ev = [0.357500, 0.354500, 0.388500, 0.339000, 0.336500, 0.368000, 0.351500, 0.470000]
    gaussF_pcaF_pi9_ev = [0.290000, 0.281000, 0.287500, 0.272000, 0.271500, 0.297500, 0.282000, 0.351500]
    gaussF_pca8_pi5_ev = [0.072500, 0.090000, 0.088000, 0.079000, 0.095500, 0.071000, 0.119000, 0.242500]
    gaussF_pca8_pi1_ev = [0.201500, 0.219500, 0.226000, 0.195500, 0.260500, 0.204500, 0.339500, 0.621500]
    gaussF_pca8_pi9_ev = [0.165500, 0.220000, 0.197000, 0.236500, 0.215500, 0.161500, 0.317500, 0.565000]

    plt.figure()
    plt.suptitle("Quadratic logistic regression comparison")
    plt.subplot(1, 2, 1)
    plt.title("No PCA")
    plt.semilogx(l, gaussF_pcaF_pi5, '--r', label=r'$\tilde{\pi} = 0.5$ [Val]')
    plt.semilogx(l, gaussF_pcaF_pi1, '--b', label=r'$\tilde{\pi} = 0.1$ [Val]')
    plt.semilogx(l, gaussF_pcaF_pi9, '--g', label=r'$\tilde{\pi} = 0.9$ [Val]')
    plt.semilogx(l, gaussF_pcaF_pi5_ev, 'r', label=r'$\tilde{\pi} = 0.5$ [Eval]')
    plt.semilogx(l, gaussF_pcaF_pi1_ev, 'b', label=r'$\tilde{\pi} = 0.1$ [Eval]')
    plt.semilogx(l, gaussF_pcaF_pi9_ev, 'g', label=r'$\tilde{\pi} = 0.9$ [Eval]')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("minDCF")
    plt.grid(visible=True, linestyle='--')
    plt.xticks(l)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("PCA (m = 8)")
    plt.semilogx(l, gaussF_pca8_pi5, '--r', label=r'$\tilde{\pi} = 0.5$ [Val]')
    plt.semilogx(l, gaussF_pca8_pi1, '--b', label=r'$\tilde{\pi} = 0.1$ [Val]')
    plt.semilogx(l, gaussF_pca8_pi9, '--g', label=r'$\tilde{\pi} = 0.9$ [Val]')
    plt.semilogx(l, gaussF_pca8_pi5_ev, 'r', label=r'$\tilde{\pi} = 0.5$ [Eval]')
    plt.semilogx(l, gaussF_pca8_pi1_ev, 'b', label=r'$\tilde{\pi} = 0.1$ [Eval]')
    plt.semilogx(l, gaussF_pca8_pi9_ev, 'g', label=r'$\tilde{\pi} = 0.9$ [Eval]')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("minDCF")
    plt.grid(visible=True, linestyle='--')
    plt.xticks(l)
    plt.legend()
    plt.show()


def gmm_plot(noPca, pca8, noPca_ev, pca8_ev, file_name):
    labels = [2,4,8,16,32,64,128,256]

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    # plt.suptitle(title)
    rects1 = ax.bar(x - width * 1.5, noPca, width, label='No PCA [Val]', color='tab:orange', alpha=0.35)
    rects3 = ax.bar(x - width / 2, noPca_ev, width, label='No PCA [Eval]', color='tab:orange')
    rects4 = ax.bar(x + width / 2, pca8_ev, width, label='PCA (m = 8) [Eval]', color='tab:blue')
    rects2 = ax.bar(x + width * 1.5, pca8, width, label='PCA (m = 8) [Val]', color='tab:blue', alpha=0.35)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('minDCF')
    ax.set_xlabel('GMM components number')
    ax.set_xticks(x, labels)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    # plt.show()
    plt.savefig(file_name)


def gmm_plot_main():
    pcaF_em = np.array([0.043000, 0.032667, 0.032333, 0.044000, 0.056000, 0.087667, 0.150667, 0.313000])
    pcaF_diag = np.array([0.381667, 0.107667, 0.086667, 0.081000, 0.088000, 0.085667, 0.095667, 0.120000])
    pcaF_tied = np.array([0.048000, 0.029667, 0.030000, 0.031667, 0.032333, 0.036667, 0.040333, 0.044000])
    pcaF_tied_diag = np.array([0.380333, 0.104667, 0.082333, 0.083333, 0.072000, 0.067667, 0.069000, 0.070667])
    pca8_em = np.array([0.045000, 0.036000, 0.036333, 0.037667, 0.047667, 0.068000, 0.098667, 0.169333])
    pca8_diag = np.array([0.081333, 0.077333, 0.087000, 0.077333, 0.080000, 0.077333, 0.085000, 0.106000])
    pca8_tied = np.array([0.045667, 0.044667, 0.034333, 0.035667, 0.037333, 0.041667, 0.039667, 0.049667])
    pca8_tied_diag = np.array([0.081333, 0.073333, 0.081333, 0.067000, 0.061000, 0.056000, 0.056333, 0.060333])

    pcaF_em_ev = np.array([0.052000, 0.041500, 0.032000, 0.037000, 0.045000, 0.064000, 0.098000, 0.178500])
    pcaF_diag_ev = np.array([0.392000, 0.112000, 0.091000, 0.098000, 0.079000, 0.097000, 0.103500, 0.120500])
    pcaF_tied_ev = np.array([0.053000, 0.029500, 0.030500, 0.033500, 0.033000, 0.034500, 0.038500, 0.046000])
    pcaF_tied_diag_ev = np.array([0.390500, 0.111000, 0.087500, 0.095500, 0.086500, 0.085000, 0.074500, 0.074500])
    pca8_em_ev = np.array([0.051500, 0.035000, 0.038000, 0.038500, 0.045500, 0.064000, 0.091500, 0.131000])
    pca8_diag_ev = np.array([0.080500, 0.082500, 0.082500, 0.078500, 0.082000, 0.089000, 0.098500, 0.111000])
    pca8_tied_ev = np.array([0.052500, 0.038500, 0.037000, 0.039000, 0.040000, 0.043000, 0.046500, 0.052500])
    pca8_tied_diag_ev = np.array([0.079000, 0.078500, 0.073500, 0.067000, 0.060500, 0.058500, 0.063500, 0.063000])

    gmm_plot(pcaF_em, pca8_em, pcaF_em_ev, pca8_em_ev, '../plots/evaluation/gmm_em.png')
    gmm_plot(pcaF_diag, pca8_diag, pcaF_diag_ev, pca8_diag_ev, '../plots/evaluation/gmm_diag.png')
    gmm_plot(pcaF_tied, pca8_tied, pcaF_tied_ev, pca8_tied_ev, '../plots/evaluation/gmm_tied.png')
    gmm_plot(pcaF_tied_diag, pca8_tied_diag, pcaF_tied_diag_ev, pca8_tied_diag_ev, '../plots/evaluation/gmm_tied_diag.png')


if __name__ == '__main__':
    # quad_lr_plot()
    # gmm_plot_main()
    # ev_linear_poly_svm_plot()
    ev_radial_svm_plot()

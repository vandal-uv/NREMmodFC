import numpy as np
from scipy import signal
import graph_utils
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import Hopf_model_multi as HM

homotopic = np.array([(i, 90 - (i+1)) for i in range(90)])
        
# dist_placebo_rest = graph_utils.get_uptri(np.load('Results/Fitting/Nicotine_Empirical_FCs/FC_placebo_rest.npy'))
objective_FC = graph_utils.get_uptri(np.loadtxt("empirical/mean_mat_W_8dic24.txt"))

# Model parameters
HM.a = 0  # External input (bifurcation parameter)
HM.w = 0.05 * 2 * np.pi  # Oscillatory frequency
# Note: frequencies could be different between brain regions and also obtained using the real data
HM.beta = 0.032  # noise scaling factor

# Simulation parameters
HM.dt = 1E-1  # Integration step
HM.teq = 60  # Equilibrium time
HM.tmax = 600 + HM.teq * 2  # Signals' length
HM.downsamp = 1  # This reduces the number of points of the signals by X

# Load the original structural connectivity
deco_mat = np.loadtxt('empirical/structural_Deco_AAL.txt')
HM.M = deco_mat
HM.norm = np.mean(np.sum(HM.M, 0))  # global normalization
C = np.copy(HM.M)  # Copy of the original matrix
HM.nnodes = len(HM.M)  # number of nodes
HM.ones_vector = np.ones(HM.nnodes).reshape((1, HM.nnodes))
GG = 0.6 ##chosen after exploration
HM.G = GG  # Global coupling
HM.seed = 0  # Random seed

# Optimization
seeds = 10  # Number of random seeds
iters = 100  # Number of iterations
fitting = np.zeros((4, iters))  # matrix to store the measures of the fitting
# matrix to store all the optimized SCs
all_SCs = np.zeros((HM.nnodes, HM.nnodes, iters))
epsilon = 0.03  # Convergence rate

original_sum = np.sum(deco_mat)
for i in range(0, iters):
    now = time.time()
    # Static Functional Connectivity (sFC) matrix
    sFC_BOLD = np.zeros((HM.nnodes, HM.nnodes))
    for ss in range(0, seeds):
        all_SCs[:, :, i] = np.copy(C)

        # Run simulation
        HM.seed = ss
        y, time_vector = HM.Sim(verbose=False)
        BOLD_signals = y[:, :, 0]  # BOLD-like signals (non-filtered)

        # Filtering BOLD signals
        resolution = HM.dt * HM.downsamp
        Fmin, Fmax = 0.01, 0.1  # Allowed frequencies

        # Filter parameters
        a0, b0 = signal.bessel(
            3, [2 * resolution * Fmin, 2 * resolution * Fmax], btype='bandpass')
        BOLDfilt = signal.filtfilt(
            a0, b0, BOLD_signals, axis=0)  # filteres signals
        cut0, cut1 = int(60 / resolution), int((HM.tmax - 60) / resolution)
        # this cuts the tails of the signals for avoiding filtering artifacts
        BOLDfilt = BOLDfilt[cut0:cut1, :]
        sFC_BOLD += np.corrcoef((BOLDfilt.T))
        print([ss, i])
    print(time.time()-now)

    sFC_BOLD /= seeds
    # distribution of the simulated FC matrix
    dist_sim = graph_utils.get_uptri((sFC_BOLD))
    # difference in global correlations
    fitting[0, i] = np.mean(objective_FC) - np.mean(dist_sim)
    fitting[1, i] = stats.ks_2samp(objective_FC, dist_sim)[
        0]  # kolmogorov-smirnov distance
    fitting[2, i] = np.linalg.norm(
        objective_FC - dist_sim)  # euclidean distance
    fitting[3, i] = stats.pearsonr(objective_FC, dist_sim)[
        0]  # pearson correlation
    print(fitting[:, i])
    print(np.sum(C), np.sum(C > 0))

    # Update the SC
    # C += graph_utils.matrix_recon((epsilon*(objective_FC - dist_sim)))
    C[homotopic[:, 0], homotopic[:, 1]] += graph_utils.matrix_recon((epsilon*(objective_FC - dist_sim)))[homotopic[:, 0], homotopic[:, 1]]
    C[C < 0] = 0  # The SC cannot allow negative valuess
    # The next two steps are optional, based on explorations
    C = graph_utils.thresholding(C, 0.3)  # This locks the matrix density
    C = C * original_sum / np.sum(C)  # This locks the sum of weights of the matrix

    # Change again the SC of the model and update the normalization factor
    HM.M = C
    HM.norm = np.mean(np.sum(HM.M, 0))
    if i%4 == 0:
        print(C[homotopic[:, 0], homotopic[:, 1]].mean())
        plt.figure(i)
        sns.heatmap(C,cmap="jet",robust=True)
        plt.show()

# np.savetxt("SC_opti_25julio.txt", C)
# %%


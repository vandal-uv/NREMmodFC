# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:39:50 2024

@author: flehu
"""
import sys
sys.path.append("../")
import numpy as np
from scipy.io import loadmat
import nibabel as nb
# from nilearn.input_data import NiftiLabelsMasker
# from nilearn import image
from scipy import signal
from scipy.integrate import simps
from scipy.stats import linregress,normaltest
import os
from statsmodels.stats.multitest import multipletests
import pingouin as pg
import pandas as pd
from scipy.stats import ttest_rel,ttest_ind,mannwhitneyu
# from scipy.ndimage.measurements import center_of_mass
from plot_violins import violin_plot
import matplotlib.pyplot as plt
from scipy import stats
import pickle 
import utils
import HMA

states= ("W","N1","N2","N3")

with open("input/all_individuals_IDs.pickle","rb") as f:
    all_entries = pickle.load(f)
entries = all_entries["included"];excluded_entries = all_entries["excluded"]


def mean_mat(mat_of_mats,arctanh = True):
    out = np.zeros((90,90))
    
    if arctanh:
        for i in range(15):
            new = np.arctanh(mat_of_mats[:,:,i])
            out += new 
        out /= 15
        return np.tanh(out)
    else:
        for i in range(15):
            new = mat_of_mats[:,:,i]
            out += new 
        out /= 15
        return out

def reord(mat):
    l = range(0,90,2)
    r = range(1,90,2)[::-1]
    idx = list(l)+list(r)
    # print(idx)
    return mat[idx,:][:,idx]

def draw_significance_lines(ax, lines_info):
    """
    Draw horizontal lines of significance on the current axis with annotations.

    Parameters:
    ax : matplotlib.axes.Axes
        The axes where the lines will be drawn.
    lines_info : list of dicts
        A list where each dict contains:
        - 'y_position': the y-coordinate of the line,
        - 'x_start': the starting x-coordinate of the line,
        - 'x_end': the ending x-coordinate of the line,
        - 'significance': a string with significance level ("*" or "**").
    """
    for line in lines_info:
        y_position = line['y_position']
        x_start = line['x_start']
        x_end = line['x_end']
        significance = line['significance']

        # Draw the horizontal line
        ax.plot([x_start, x_end], [y_position, y_position], color='black', lw=2)

        # Annotate the significance at the center of the line
        ax.text((x_start + x_end) / 2, y_position, significance, 
                ha='center', va='center', fontsize=15, color='black')



BOLD_output = '../../wico_sleep_fit/transition_comparison_local/data/BOLD_empirical_AAL90'
TR = 2.08    #fMRI repetition time    
a0, b0 = signal.bessel(2, 2 * TR * np.array([0.01, 0.1]), btype = 'bandpass')
exclude = []
all_FCs = np.zeros((92,92,4,len(entries))) 
all_BOLD_WAKE = []
all_BOLD_N1 = []
all_BOLD_N2 = []
all_BOLD_N3 = []

# current_idx = 0
# excluded_entries = []
for rx,entry in enumerate(entries):
    
    #cerebral signals
    BOLD_signals = np.load(BOLD_output+"/" +f'BOLD_complete_nonfilt_S{entry}.npy')    
    BOLD_filt = signal.filtfilt(a0, b0, BOLD_signals, axis = 0)[:,0:90]
    
    #LC signals
    LC_signals = np.load(BOLD_output+"/" +f'LC_norm_nonfilt_S{entry}.npy')    
    LC_filt = signal.filtfilt(a0, b0, LC_signals, axis = 0)    

    #BF signals
    BF_signals = np.load(BOLD_output+"/" +f'BF_norm_nonfilt_S{entry}.npy')    
    BF_filt = signal.filtfilt(a0, b0, BF_signals, axis = 0)    
    
    #concatenete brain with nuclei
    BOLD_filt = np.column_stack((BOLD_filt, LC_filt))
    BOLD_filt = np.column_stack((BOLD_filt, BF_filt))
    vector = np.load(BOLD_output +"/" +f'align_stages_S{entry}.npy')
    idx_WAKE = vector == 0
    idx_N1 = vector == -2
    idx_N2 = vector == -3
    idx_N3 = vector == -4
    
    
    #Extracting the time series for each sleep stage, then compute the FC matrices
    BOLD_WAKE = BOLD_filt[idx_WAKE,:]
    BOLD_N1 = BOLD_filt[idx_N1,:]
    BOLD_N2 = BOLD_filt[idx_N2,:]
    BOLD_N3 = BOLD_filt[idx_N3,:]
    
    all_BOLD_WAKE.append(BOLD_WAKE)
    all_BOLD_N1.append(BOLD_N1)
    all_BOLD_N2.append(BOLD_N2)
    all_BOLD_N3.append(BOLD_N3)
    
    all_FCs[:,:,0,rx] = np.corrcoef(BOLD_WAKE.T)
    all_FCs[:,:,1,rx] = np.corrcoef(BOLD_N1.T)
    all_FCs[:,:,2,rx] = np.corrcoef(BOLD_N2.T)
    all_FCs[:,:,3,rx] = np.corrcoef(BOLD_N3.T)
#%% generate dic with HMA

# save_dic = {}
# for s,st in enumerate(states):
#     for i in range(15):
#         print(st,i)
#         sFC = reord(all_FCs[:90,:90,s,i])
#         Clus_num_sim,Clus_size_sim,H_all_sim = HMA.Functional_HP(sFC)
#         Hin_sim,Hse_sim = HMA.Balance(sFC, Clus_num_sim, Clus_size_sim)
#         # HMA_balance_sim = Hin_sim-Hse_sim
#         Hin_node_sim,Hse_node_sim = HMA.nodal_measures(sFC, Clus_num_sim, Clus_size_sim)
#         save_dic[(i,st)] = {"Hin_sim":Hin_sim,"Hse_sim":Hse_sim,
#                                      "Hin_node_sim":Hin_node_sim,"Hse_node_sim":Hse_node_sim,
#                                      "sFC":sFC}
        
# with open('data/emp_15inds_output_16dic.pickle', 'wb') as f:
#     pickle.dump(save_dic,f)

# halt
#%%

###########LC and BF###############

#Labels of AAL (90 ROIs)
AAL_labels = pd.read_csv('input/ROI_MNI_V4.csv')['label'].values[0:90]

#Create a mega matrix with all BOLD time series for WAKE for each subject
WAKE_matrix = np.zeros((0,93))
for i in range(0,15):
    WAKE_matrix = np.row_stack((WAKE_matrix,
                                np.column_stack((all_BOLD_WAKE[i],
                                                i*np.ones(all_BOLD_WAKE[i].shape[0])))))
    
#%%
columns = np.arange(0,92,1).astype('str') #subjects
columns = np.append(columns, 'sub')
WAKE_matrix = pd.DataFrame(WAKE_matrix, columns = columns)

#repeated measures correlation for WAKE (FDR corrected pvals)
pvals_wake_LC = np.zeros(90)
rvals_wake_LC = np.zeros(90)
for i in range(0,90):
    rvals_wake_LC[i] = pg.rm_corr(data = WAKE_matrix, 
                                x = '%i'%i, y = '90', subject = 'sub')['r'][0]
    pvals_wake_LC[i] = pg.rm_corr(data = WAKE_matrix, 
                                x = '%i'%i, y = '90', subject = 'sub')['pval'][0]
pvals_wake_LC =  multipletests(pvals_wake_LC, alpha=0.05, method='fdr_bh', 
                            is_sorted=False, returnsorted=False)[1]   


#Same as above for N1, N2 and N3

N1_matrix = np.zeros((0,93))
for i in range(0,15):
    N1_matrix = np.row_stack((N1_matrix,
                                np.column_stack((all_BOLD_N1[i],
                                                i*np.ones(all_BOLD_N1[i].shape[0])))))
columns = np.arange(0,92,1).astype('str')
columns = np.append(columns, 'sub')
N1_matrix = pd.DataFrame(N1_matrix, columns = columns)

pvals_N1_LC = np.zeros(90)
rvals_N1_LC = np.zeros(90)
for i in range(0,90):
    rvals_N1_LC[i] = pg.rm_corr(data = N1_matrix, 
                                x = '%i'%i, y = '90', subject = 'sub')['r'][0]
    pvals_N1_LC[i] = pg.rm_corr(data = N1_matrix, 
                                x = '%i'%i, y = '90', subject = 'sub')['pval'][0]
pvals_N1_LC =  multipletests(pvals_N1_LC, alpha=0.05, method='fdr_bh', 
                            is_sorted=False, returnsorted=False)[1]  
    
    

N2_matrix = np.zeros((0,93))
for i in range(0,15):
    N2_matrix = np.row_stack((N2_matrix,
                                np.column_stack((all_BOLD_N2[i],
                                                i*np.ones(all_BOLD_N2[i].shape[0])))))
columns = np.arange(0,92,1).astype('str')
columns = np.append(columns, 'sub')
N2_matrix = pd.DataFrame(N2_matrix, columns = columns)

pvals_N2_LC = np.zeros(90)
rvals_N2_LC = np.zeros(90)
for i in range(0,90):
    rvals_N2_LC[i] = pg.rm_corr(data = N2_matrix, 
                                x = '%i'%i, y = '90', subject = 'sub')['r'][0]
    pvals_N2_LC[i] = pg.rm_corr(data = N2_matrix, 
                                x = '%i'%i, y = '90', subject = 'sub')['pval'][0]
pvals_N2_LC =  multipletests(pvals_N2_LC, alpha=0.05, method='fdr_bh', 
                            is_sorted=False, returnsorted=False)[1]  


N3_matrix = np.zeros((0,93))
for i in range(0,15):
    N3_matrix = np.row_stack((N3_matrix,
                                np.column_stack((all_BOLD_N3[i],
                                                i*np.ones(all_BOLD_N3[i].shape[0])))))
columns = np.arange(0,92,1).astype('str')
columns = np.append(columns, 'sub')
N3_matrix = pd.DataFrame(N3_matrix, columns = columns)

pvals_N3_LC = np.zeros(90)
rvals_N3_LC = np.zeros(90)
for i in range(0,90):
    rvals_N3_LC[i] = pg.rm_corr(data = N3_matrix, 
                                x = '%i'%i, y = '90', subject = 'sub')['r'][0]
    pvals_N3_LC[i] = pg.rm_corr(data = N3_matrix, 
                                x = '%i'%i, y = '90', subject = 'sub')['pval'][0]
pvals_N3_LC =  multipletests(pvals_N3_LC, alpha=0.05, method='fdr_bh', 
                            is_sorted=False, returnsorted=False)[1]  


#plot only correlations with pvals < 0.05 (after FDR correction)
my_mat = np.column_stack(((rvals_wake_LC,# * (pvals_wake_LC < 0.05), 
                            rvals_N1_LC, #* (pvals_N1_LC < 0.05),
                            rvals_N2_LC, #* (pvals_N2_LC < 0.05),
                            rvals_N3_LC)))# * (pvals_N3_LC < 0.05))))
plt.figure(666, figsize = (8,6.5))
plt.clf()
plt.imshow(my_mat, vmin = -0.3, vmax = 0.3, cmap = 'RdBu', aspect = 'auto')
plt.xticks([0,1,2,3], ['Wake', 'N1', 'N2', 'N3'])
plt.yticks(np.arange(0,90,1), AAL_labels, fontsize = 5)
cax = plt.colorbar()
cax.set_label('Correlation')
plt.title('Correlation with LC')

#LC mean FC is very similar between N1 and Wake!!! 
print('WAKE LC MEAN FC', np.mean(rvals_wake_LC * (pvals_wake_LC < 0.05)))
print('N1 LC MEAN FC', np.mean(rvals_N1_LC * (pvals_N1_LC < 0.05)))
print('N2 LC MEAN FC', np.mean(rvals_N2_LC * (pvals_N2_LC < 0.05)))
print('N3 LC MEAN FC', np.mean(rvals_N3_LC * (pvals_N3_LC < 0.05)))

factor_N1 = 1
factor_N2 = 1
factor_N3 = 1


#Comparisons with WAKE (ROIs)
print(ttest_rel(rvals_N1_LC * (pvals_N1_LC < 0.05) * factor_N1, rvals_wake_LC * (pvals_wake_LC < 0.05)))
print(ttest_rel(rvals_N2_LC * (pvals_N2_LC < 0.05) * factor_N2, rvals_wake_LC * (pvals_wake_LC < 0.05)))
print(ttest_rel(rvals_N3_LC * (pvals_N3_LC < 0.05) * factor_N3, rvals_wake_LC * (pvals_wake_LC < 0.05)))

#Comparisons with WAKE (subjects)
print(ttest_rel(np.mean(all_FCs[0:90,90,0,:],0), np.mean(all_FCs[0:90,90,1,:],0)))
print(ttest_rel(np.mean(all_FCs[0:90,90,0,:],0), np.mean(all_FCs[0:90,90,2,:],0)))
print(ttest_rel(np.mean(all_FCs[0:90,90,0,:],0), np.mean(all_FCs[0:90,90,3,:],0)))





#%%
###########BF###############
   
# halt
#%%

#repeated measures correlation for WAKE (FDR corrected pvals)
pvals_wake_BF = np.zeros(90)
rvals_wake_BF = np.zeros(90)
for i in range(0,90):
    rvals_wake_BF[i] = pg.rm_corr(data = WAKE_matrix, 
                                x = '%i'%i, y = '91', subject = 'sub')['r'][0]
    pvals_wake_BF[i] = pg.rm_corr(data = WAKE_matrix, 
                                x = '%i'%i, y = '91', subject = 'sub')['pval'][0]
pvals_wake_BF =  multipletests(pvals_wake_BF, alpha=0.05, method='fdr_bh', 
                            is_sorted=False, returnsorted=False)[1]   


#Same as above for N1, N2 and N3

N1_matrix = np.zeros((0,93))
for i in range(0,15):
    N1_matrix = np.row_stack((N1_matrix,
                                np.column_stack((all_BOLD_N1[i],
                                                i*np.ones(all_BOLD_N1[i].shape[0])))))
columns = np.arange(0,92,1).astype('str')
columns = np.append(columns, 'sub')
N1_matrix = pd.DataFrame(N1_matrix, columns = columns)

pvals_N1_BF = np.zeros(90)
rvals_N1_BF = np.zeros(90)
for i in range(0,90):
    rvals_N1_BF[i] = pg.rm_corr(data = N1_matrix, 
                                x = '%i'%i, y = '91', subject = 'sub')['r'][0]
    pvals_N1_BF[i] = pg.rm_corr(data = N1_matrix, 
                                x = '%i'%i, y = '91', subject = 'sub')['pval'][0]
pvals_N1_BF =  multipletests(pvals_N1_BF, alpha=0.05, method='fdr_bh', 
                            is_sorted=False, returnsorted=False)[1]  
    
    

N2_matrix = np.zeros((0,93))
for i in range(0,15):
    N2_matrix = np.row_stack((N2_matrix,
                                np.column_stack((all_BOLD_N2[i],
                                                i*np.ones(all_BOLD_N2[i].shape[0])))))
columns = np.arange(0,92,1).astype('str')
columns = np.append(columns, 'sub')
N2_matrix = pd.DataFrame(N2_matrix, columns = columns)

pvals_N2_BF = np.zeros(90)
rvals_N2_BF = np.zeros(90)
for i in range(0,90):
    rvals_N2_BF[i] = pg.rm_corr(data = N2_matrix, 
                                x = '%i'%i, y = '91', subject = 'sub')['r'][0]
    pvals_N2_BF[i] = pg.rm_corr(data = N2_matrix, 
                                x = '%i'%i, y = '91', subject = 'sub')['pval'][0]
pvals_N2_BF =  multipletests(pvals_N2_BF, alpha=0.05, method='fdr_bh', 
                            is_sorted=False, returnsorted=False)[1]  


N3_matrix = np.zeros((0,93))
for i in range(0,15):
    N3_matrix = np.row_stack((N3_matrix,
                                np.column_stack((all_BOLD_N3[i],
                                                i*np.ones(all_BOLD_N3[i].shape[0])))))
columns = np.arange(0,92,1).astype('str')
columns = np.append(columns, 'sub')
N3_matrix = pd.DataFrame(N3_matrix, columns = columns)

pvals_N3_BF = np.zeros(90)
rvals_N3_BF = np.zeros(90)
for i in range(0,90):
    rvals_N3_BF[i] = pg.rm_corr(data = N3_matrix, 
                                x = '%i'%i, y = '91', subject = 'sub')['r'][0]
    pvals_N3_BF[i] = pg.rm_corr(data = N3_matrix, 
                                x = '%i'%i, y = '91', subject = 'sub')['pval'][0]
pvals_N3_BF =  multipletests(pvals_N3_BF, alpha=0.05, method='fdr_bh', 
                            is_sorted=False, returnsorted=False)[1]  


#plot only correlations with pvals < 0.05 (after FDR correction)
my_mat = np.column_stack(((rvals_wake_BF,# * (pvals_wake < 0.05), 
                            rvals_N1_BF,# * (pvals_N1 < 0.05),
                            rvals_N2_BF,# * (pvals_N2 < 0.05),
                            rvals_N3_BF)))# * (pvals_N3 < 0.05))))
plt.figure(666, figsize = (8,6.5))
plt.clf()
plt.imshow(my_mat, vmin = -0.3, vmax = 0.3, cmap = 'RdBu', aspect = 'auto')
plt.xticks([0,1,2,3], ['Wake', 'N1', 'N2', 'N3'])
plt.yticks(np.arange(0,90,1), AAL_labels, fontsize = 5)
cax = plt.colorbar()
cax.set_label('Correlation')
plt.title('Correlation with BF')

#%% analisis estadisticos simples



#BF mean FC
print('WAKE BF MEAN FC', np.mean(rvals_wake_BF * (pvals_wake_BF < 0.05)))
print('N1 BF MEAN FC', np.mean(rvals_N1_BF * (pvals_N1_BF < 0.05)))
print('N2 BF MEAN FC', np.mean(rvals_N2_BF * (pvals_N2_BF < 0.05)))
print('N3 BF MEAN FC', np.mean(rvals_N3_BF * (pvals_N3_BF < 0.05)))

#Comparisons with WAKE (ROIs)
print(ttest_rel(rvals_N1_BF * (pvals_N1_BF < 0.05) * factor_N1, rvals_wake_BF * (pvals_wake_BF < 0.05)))
print(ttest_rel(rvals_N2_BF * (pvals_N2_BF < 0.05) * factor_N2, rvals_wake_BF * (pvals_wake_BF < 0.05)))
print(ttest_rel(rvals_N3_BF * (pvals_N3_BF < 0.05) * factor_N3, rvals_wake_BF * (pvals_wake_BF < 0.05)))


##cohen's d
#Comparisons with WAKE (ROIs) BF
print("cohen's D BF stages against W")
print(utils.cohen_d(rvals_N1_BF * (pvals_N1_BF < 0.05), rvals_wake_BF * (pvals_wake_BF < 0.05)))
print(utils.cohen_d(rvals_N2_BF * (pvals_N2_BF < 0.05), rvals_wake_BF * (pvals_wake_BF < 0.05)))
print(utils.cohen_d(rvals_N3_BF * (pvals_N3_BF < 0.05), rvals_wake_BF * (pvals_wake_BF < 0.05)))

#Comparisons with WAKE (ROIs) LC
print("cohen's D LC stages against W")
print(utils.cohen_d(rvals_N1_LC * (pvals_N1_LC < 0.05), rvals_wake_LC * (pvals_wake_LC < 0.05)))
print(utils.cohen_d(rvals_N2_LC * (pvals_N2_LC < 0.05), rvals_wake_LC * (pvals_wake_LC < 0.05)))
print(utils.cohen_d(rvals_N3_LC * (pvals_N3_LC < 0.05), rvals_wake_LC * (pvals_wake_LC < 0.05)))

#%%


###############node strengths AAL90
NS_wake = np.mean(np.sum(all_FCs[:,:,0,:],1),1)
NS_N1 = np.mean(np.sum(all_FCs[:,:,1,:],1),1)
NS_N2 =  np.mean(np.sum(all_FCs[:,:,2,:],1),1)
NS_N3 =  np.mean(np.sum(all_FCs[:,:,3,:],1),1)

NS_wake = np.arctanh(np.mean(np.mean(all_FCs[:,:,0,:],1),1))
NS_N1 = np.arctanh(np.mean(np.mean(all_FCs[:,:,1,:],1),1))
NS_N2 =  np.arctanh(np.mean(np.mean(all_FCs[:,:,2,:],1),1))
NS_N3 =  np.arctanh(np.mean(np.mean(all_FCs[:,:,3,:],1),1))
NSs = [NS_wake,NS_N1,NS_N2,NS_N3]


################# Zscored LC and BF strength variation, from W
for s,st in enumerate(states):
    if s ==0:
        continue
    this_NS = NSs[s]
    deltas = this_NS-NS_wake
    print(normaltest(deltas))
    deltas_Z = (deltas-deltas.mean())/deltas.std()
    print(f"W and {st} difference Z-scored\nBF: {deltas_Z[91]:.3f}\nLC: {deltas_Z[90]:.3f}")

print("\nnow sequential!!!!!!!!!!\n")
################# Zscored LC and BF strength variation, sequential
for s,st in enumerate(states):
    if s ==0:
        continue
    deltas = NSs[s]-NSs[s-1]
    print(normaltest(deltas))
    deltas_Z = (deltas-deltas.mean())/deltas.std()
    print(f"{st} from {states[s-1]} difference Z-scored\nBF: {deltas_Z[91]:.3f}\nLC: {deltas_Z[90]:.3f}")

#%%% effect size node strengths of nuclei
##################BF effect size
p1,p2,p3 = [ttest_ind(eval(f"rvals_{st}_BF"), rvals_wake_BF)[1] for st in states[1:]]
p1,p2,p3 =  multipletests((p1,p2,p3), alpha=0.05, method='fdr_bh', 
                            is_sorted=False, returnsorted=False)[1]   

print("BF corrs pvals:",p1,p2,p3)
# print(utils.cohen_d(rvals_N1_BF[pvals_N1_BF<0.05], rvals_wake_BF[pvals_wake_BF<0.05]))
# print(utils.cohen_d(rvals_N2_BF[pvals_N2_BF<0.05], rvals_wake_BF[pvals_wake_BF<0.05]))
# print(utils.cohen_d(rvals_N3_BF[pvals_N3_BF<0.05], rvals_wake_BF[pvals_wake_BF<0.05]))


##################LC effect size
p1,p2,p3 = [ttest_ind(eval(f"rvals_{st}_LC"), rvals_wake_LC)[1] for st in states[1:]]
p1,p2,p3 =  multipletests((p1,p2,p3), alpha=0.05, method='fdr_bh', 
                            is_sorted=False, returnsorted=False)[1] 

print("LC corrs pvals:",p1,p2,p3)
# print(utils.cohen_d(rvals_N1_LC[pvals_N1_LC<0.05], rvals_wake_LC[pvals_wake_LC<0.05]))
# print(utils.cohen_d(rvals_N2_LC[pvals_N2_LC<0.05], rvals_wake_LC[pvals_wake_LC<0.05]))
# print(utils.cohen_d(rvals_N3_LC[pvals_N3_LC<0.05], rvals_wake_LC[pvals_wake_LC<0.05]))


##el dato de los primeros violines es la fuerza de nodo promediada a través de todos los individuos
#creo que lo correcto sería tomar la FUERZA DE NODO del LC y del BF en cada etapa, y verla como un Zscore con respecto al resto. 
##el LC es el área #90, el BF es el área #91



##means
ofc_wake = np.mean(np.sum(all_FCs[:90,:90,0,:],1),1)
ofc_N1 = np.mean(np.sum(all_FCs[:90,:90,1,:],1),1)
ofc_N2 =  np.mean(np.sum(all_FCs[:90,:90,2,:],1),1)
ofc_N3 =  np.mean(np.sum(all_FCs[:90,:90,3,:],1),1)

p1,p2,p3 = [ttest_ind(eval(f"ofc_{st}"), ofc_wake)[1] for st in states[1:]]
p1,p2,p3 =  multipletests((p1,p2,p3), alpha=0.05, method='fdr_bh', 
                            is_sorted=False, returnsorted=False)[1]
print("general means pvals:",p1,p2,p3)

d1,d2,d3 = [utils.cohen_d(eval(f"ofc_{st}"), ofc_wake) for st in states[1:]]
print("general means cohen's Ds:",d1,d2,d3)






#%% generate the plots



alfa = 0.7
titlesais = 20
labelsais = 20
ticsais = 16
legendsais = 13
   

plt.figure(1)
plt.clf()
plt.gcf().set_size_inches(12, 10)


ax = plt.subplot(2,2,1)
violin_plot(ax, [rvals_wake_BF, # * (pvals_wake < 0.05),
                 rvals_N1_BF,# * (pvals_N1 < 0.05),
                 rvals_N2_BF,# * (pvals_N2 < 0.05),
                 rvals_N3_BF],# * (pvals_N3 < 0.05)],
            color_names= ['crimson', 'orange', 'forestgreen', 'navy'],
            alpha_violin = 0.5)
# plt.xlabel('Stage', fontsize = labelsais)
plt.ylim(-0.2,1)
plt.yticks([0,0.5,1], fontsize = ticsais)
plt.xticks([0,1,2,3], ['W', 'N1', 'N2', 'N3'], fontsize = ticsais)
plt.ylabel('Mean correlation', fontsize = labelsais)
plt.title('Basal Forebrain connectivity', fontsize = titlesais)
plt.tight_layout()
lines_info = [{'y_position': 0.9, 'x_start': 0, 'x_end': 3, 'significance': '*'}]
draw_significance_lines(ax,lines_info)

ax = plt.subplot(2,2,2)
violin_plot(ax, [rvals_wake_LC,# * (pvals_wake < 0.05),
                  rvals_N1_LC,# * (pvals_N1 < 0.05),
                  rvals_N2_LC,# * (pvals_N2 < 0.05),
                  rvals_N3_LC],# * (pvals_N3 < 0.05)],
            color_names= ['crimson', 'orange', 'forestgreen', 'navy'],
            alpha_violin = 0.5)
# plt.xlabel('Stage', fontsize = labelsais)
plt.ylim(-0.1,0.4)
plt.yticks([0,0.25,0.5], fontsize = ticsais)
plt.xticks([0,1,2,3], ['W', 'N1', 'N2', 'N3'], fontsize = ticsais)
plt.ylabel('Mean correlation', fontsize = labelsais)
plt.title('Locus Coeruleus connectivity', fontsize = titlesais)
lines_info = [{'y_position': 0.4, 'x_start': 0, 'x_end': 2, 'significance': '*'},
              {'y_position': 0.43, 'x_start': 0, 'x_end': 3, 'significance': '*'}]
draw_significance_lines(ax,lines_info)

ax = plt.subplot(2,2,3)
violin_plot(ax, [ofc_wake,
                  ofc_N1,
                  ofc_N2,
                  ofc_N3],
            color_names= ['crimson', 'orange', 'forestgreen', 'navy'],
            alpha_violin = 0.5)
# plt.xlabel('Stage', fontsize = labelsais)
# plt.ylim(-1,80)
plt.yticks([0,40,80], fontsize = ticsais)
plt.xticks([0,1,2,3], ['W', 'N1', 'N2', 'N3'], fontsize=ticsais)
plt.ylabel('Nodal strength', fontsize = labelsais)
plt.title('Global correlations', fontsize = titlesais)
lines_info = [{'y_position': 65, 'x_start': 0, 'x_end': 1, 'significance': '*'},
              {'y_position': 70, 'x_start': 0, 'x_end': 3, 'significance': '*'}]
draw_significance_lines(ax,lines_info)


plt.tight_layout()
plt.savefig("figures/prefig1.png",dpi=300)

#%% plot individuals' FCs

plt.figure(100)
plt.clf()
plt.suptitle("W")
for rx,entry in enumerate(entries):
    mat = all_FCs[:,:,0,rx]
    plt.subplot(3,5,rx+1)
    plt.title(entry+f"\nmean={mat.mean():.4f}")
    plt.imshow(mat,vmin=0,vmax=1,cmap='jet')
plt.tight_layout()

###individuos raros corresponden a "20090217ML","20091020DK"
    
    
plt.figure(101)
plt.clf()
plt.suptitle("N1")
for rx,entry in enumerate(entries):
    mat = all_FCs[:,:,1,rx]
    plt.subplot(3,5,rx+1)
    plt.title(entry+f"\nmean={mat.mean():.4f}")
    plt.imshow(mat,vmin=0,vmax=1,cmap='jet')
plt.tight_layout()

plt.figure(102)
plt.clf()
plt.suptitle("N2")
for rx,entry in enumerate(entries):
    mat = all_FCs[:,:,2,rx]
    plt.subplot(3,5,rx+1)
    plt.title(entry+f"\nmean={mat.mean():.4f}")
    plt.imshow(mat,vmin=0,vmax=1,cmap='jet')
plt.tight_layout()

plt.figure(103)
plt.clf()
plt.suptitle("N3")
for rx,entry in enumerate(entries):
    mat = all_FCs[:,:,3,rx]
    plt.subplot(3,5,rx+1)
    plt.title(entry+f"\nmean={mat.mean():.4f}")
    plt.imshow(mat,vmin=0,vmax=1,cmap='jet')
plt.tight_layout()        
            


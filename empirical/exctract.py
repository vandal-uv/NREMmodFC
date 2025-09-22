# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import nibabel as nb
from nilearn.input_data import NiftiLabelsMasker
from scipy import signal
from scipy.stats import linregress
import pickle 

# halt

#Cohen's D effect size
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x) ** 2 + (ny-1)*np.std(y) ** 2) / dof)


#Folder with all the subfolders, huge size data
folder_input = '../../../../filer3/EEG_enzo/volume1/BEOSRV1T_BACKUP/eegfmri/p0802/conc_sleep_stages/'


##chosen entries
with open("empirical/all_individuals_IDs.pickle","rb") as f:
    all_entries = pickle.load(f)
##only 15 chosen ones
entries = all_entries["included"];excluded_entries = all_entries["excluded"]

BOLD_output = 'BOLD_empirical_AAL90/'
# print(entries,len(entries))


#%%brain parcellation folder
masks_folder = "masks_for_extraction/"

atlas_filename = masks_folder+'AAL.nii'
# Pedunculopontine nucleus (PN), Locus Coeruleus (LC) masks and Basal Forebrain (BF)
atlas_PN = masks_folder + 'AAN_PPN_MNI152_1mm_v1p0_20150630.nii' 
atlas_LC = masks_folder + "LCmetaMask_billateral_resampled_MNI05_s01f_plus50.nii"
atlas_BF = masks_folder + 'BF-2005_MNI_resliced_to_1.5mm.nii'

#%% Extract ALL fMRI time series
##THESE AND THE ALIGNING VECTORS ARE SAVED IN THE ZENODO


for rx,entry in enumerate(entries):
    
    # print(rx,entry)
    current_folder = folder_input + entries[rx]
        
    ###PART 1: reading imgs and extract fMRI time series
    
    #load images with nibabel
    imgs = nb.load(current_folder + '/conc_data_brain_filt_reg.img')
    
    
    #fetch data
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize = True,
                                memory='nilearn_cache', verbose = 0, detrend = False)
    #pick the first 90 ROIs (cortical and subcortical non-cerebelar brain regions)
    time_series = masker.fit_transform(imgs)
    
    
    ##extract LC and BF time series
    
    #Pedunculopontine nucleus
    masker = NiftiLabelsMasker(labels_img=atlas_PN, standardize = True,
                                memory='nilearn_cache', verbose = 0, detrend = False)
    ts_PN = masker.fit_transform(imgs)    

    #Locus Coeruleus
    masker = NiftiLabelsMasker(labels_img=atlas_LC, standardize = True,
                                memory='nilearn_cache', verbose = 0, detrend = False)
    ts_LC = masker.fit_transform(imgs)        
    
    #Regressing out PN time series from LC time series
    ##note: not adding the intercept does not change correlations later
    slope,intercept = linregress(ts_PN[:,0], ts_LC[:,0])[:2]
    ts_LC_res = ts_LC[:,0] - (slope * ts_PN[:,0] + intercept) 
 
    #Basal Forebrain
    masker = NiftiLabelsMasker(labels_img=atlas_BF, standardize = True,
                                memory='nilearn_cache', verbose = 0, detrend = False)
    ts_BF = masker.fit_transform(imgs)            
 
    ### Saving all the results
    np.save(BOLD_output + f'BOLD_complete_nonfilt_S{entry}.npy',time_series)
    np.save(BOLD_output + f'LC_bilateral_norm_nonfilt_S{entry}.npy',ts_LC_res)
    np.save(BOLD_output + f'BF_norm_nonfilt_S{entry}.npy',ts_BF)    

    print(rx)

#%%
##PART 6: computing FCs, we're using the deco symmetrized order
##BOLD timeseries use the original order


TR = 2.08    #fMRI repetition time    
a0, b0 = signal.bessel(3, 2 * TR * np.array([0.01, 0.1]), btype = 'bandpass')

all_BOLD_WAKE = []
all_BOLD_N1 = []
all_BOLD_N2 = []
all_BOLD_N3 = []

all_FCs = np.zeros((90,90,4,len(entries)))

for rx,entry in enumerate(entries):

    
    #Load and filter the BOLD signals
    # BOLD_signals = np.load('Z:/EEG-fMRI/BOLD_empirical/BOLD_complete_nonfilt_S%i.npy'%rx)
    BOLD_signals = np.load(BOLD_output + f'BOLD_complete_nonfilt_S{entry}.npy')    
    BOLD_filt = signal.filtfilt(a0, b0, BOLD_signals, axis = 0)[:,0:90]
    
    # current_folder = folder_input + entry
    vector = np.load(BOLD_output + f'/align_stages_S{entry}.npy')
    # np.save(BOLD_output + f'/align_stages_S{entry}.npy',vector)  
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
    
    ##indexed by stage
    all_FCs[:,:,0,rx] = np.corrcoef(BOLD_WAKE.T)
    all_FCs[:,:,1,rx] =np.corrcoef(BOLD_N1.T)
    all_FCs[:,:,2,rx] = np.corrcoef(BOLD_N2.T)
    all_FCs[:,:,3,rx] = np.corrcoef(BOLD_N3.T)
    
    print(rx)

##average and save
def reord(matrix):
    left = range(0,90,2)
    right = range(1,90,2)
    ids = list(left)+list(right[::-1])
    return matrix[ids,:][:,ids]

mean_WAKE = reord(np.mean(all_FCs[:,:,0,:],axis=-1))
mean_N1 = reord(np.mean(all_FCs[:,:,1,:],axis=-1))
mean_N2 = reord(np.mean(all_FCs[:,:,2,:],axis=-1))
mean_N3 = reord(np.mean(all_FCs[:,:,3,:],axis=-1))

###check if they are in the order of the deco matrix
np.savetxt("mean_mat_W_8dic24.txt",mean_WAKE)
np.savetxt("mean_mat_N1_8dic24.txt",mean_N1)
np.savetxt("mean_mat_N2_8dic24.txt",mean_N2)
np.savetxt("mean_mat_N3_8dic24.txt",mean_N3)

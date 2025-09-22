from nilearn import image, plotting, datasets
import nibabel as nib
import sys
from nilearn.input_data import NiftiLabelsMasker
from scipy.stats import linregress
from scipy import signal
import numpy as np
sys.path.append("../../")
import HMA
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp as ks, linregress as LR

folder = "../../maps/plot_masks/"

atlas_filename = folder+'AAL.nii'
atlas_LC_right = folder+'LCmetaMask_right_MNI05_s01f_plus50.nii'
atlas_LC_left = folder+'LCmetaMask_left_MNI05_s01f_plus50.nii'
atlas_LC_combined = folder+"LCmetaMask_billateral_using_mirror_MNI05_s01f_plus50.nii"


atlas_PN = folder+'AAN_PPN_MNI152_1mm_v1p0_20150630.nii'


#%%extract relevant shit
print("extracting...")


imgs = nib.load(folder+'/rfMRI_REST1_LR_hp2000_clean.nii.gz')

###AAL
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize = True,
                                memory='nilearn_cache', verbose = 0, detrend = False)
#pick the first 90 ROIs (cortical and subcortical non-cerebelar brain regions)
time_series = masker.fit_transform(imgs)


#Pedunculopontine nucleus
masker = NiftiLabelsMasker(labels_img=atlas_PN, standardize = True,
                            memory='nilearn_cache', verbose = 0, detrend = False)
ts_PN = masker.fit_transform(imgs)    

#Locus Coeruleus
#right
masker = NiftiLabelsMasker(labels_img=atlas_LC_right, standardize = True,
                            memory='nilearn_cache', verbose = 0, detrend = False)
ts_LC_right = masker.fit_transform(imgs)        

#left
masker = NiftiLabelsMasker(labels_img=atlas_LC_left, standardize = True,
                            memory='nilearn_cache', verbose = 0, detrend = False)
ts_LC_left = masker.fit_transform(imgs)        

#combined
masker = NiftiLabelsMasker(labels_img=atlas_LC_combined, standardize = True,
                            memory='nilearn_cache', verbose = 0, detrend = False)
ts_LC_combined = masker.fit_transform(imgs)        




#Regressing out PN time series from LC time series
slope = linregress(ts_PN[:,0], ts_LC_right[:,0])[0]
ts_LC_right_res = ts_LC_right[:,0] - slope * ts_PN[:,0]

slope = linregress(ts_PN[:,0], ts_LC_left[:,0])[0]
ts_LC_left_res = ts_LC_left[:,0] - slope * ts_PN[:,0]

slope = linregress(ts_PN[:,0], ts_LC_combined[:,0])[0]
ts_LC_combined_res = ts_LC_combined[:,0] - slope * ts_PN[:,0]


print("...extracted!")


#%% filter!

TR = 0.72    #fMRI repetition time    

a0, b0 = signal.bessel(3, 2 * TR * np.array([0.01, 0.1]), btype = 'bandpass')

###fMRI all areas
BOLD_filt = signal.filtfilt(a0, b0, time_series, axis = 0)[:,0:90]
fc = np.corrcoef(BOLD_filt.T)

###fMRI LC
ts_LC_right_filt = signal.filtfilt(a0, b0, ts_LC_right_res)
ts_LC_left_filt = signal.filtfilt(a0, b0, ts_LC_left_res)
ts_LC_combined_filt = signal.filtfilt(a0, b0, ts_LC_combined_res)




#%% obtain correlations

LC_right_FC = np.array([np.corrcoef(BOLD_filt[:,i],ts_LC_right_filt)[0,1] for i in range(90)])
LC_left_FC = np.array([np.corrcoef(BOLD_filt[:,i],ts_LC_left_filt)[0,1] for i in range(90)])

LC_combined_FC = np.array([np.corrcoef(BOLD_filt[:,i],ts_LC_left_filt)[0,1] for i in range(90)])

Clus_num,Clus_size,H_all = HMA.Functional_HP(fc)
Hin,Hse = HMA.Balance(fc, Clus_num, Clus_size)
Hin_node,Hse_node = HMA.nodal_measures(fc, Clus_num, Clus_size)


#%% plot some things
new_order = list(range(0,90,2))+list(range(1,90,2))[::-1]


plt.figure(1)
plt.clf()
plt.subplot(221)
plt.imshow(fc[new_order,:][:,new_order],cmap="jet")
plt.colorbar()


plt.subplot(222)
plt.scatter(Hin_node,LC_combined_FC)
slope, intercept, r, p, _ = linregress(Hin_node, LC_combined_FC)
plt.plot(Hin_node, slope * Hin_node + intercept, color='red', linestyle='--')
plt.xlabel("integration component")
plt.ylabel("FC with billateral LC")


plt.subplot(223)
plt.scatter(Hin_node,LC_right_FC)
slope, intercept, r, p, _ = linregress(Hin_node, LC_right_FC)
plt.plot(Hin_node, slope * Hin_node + intercept, color='red', linestyle='--')
plt.xlabel("integration component")
plt.ylabel("FC with right LC")

plt.subplot(224)
plt.scatter(Hin_node,LC_left_FC)
slope, intercept, r, p, _ = linregress(Hin_node, LC_left_FC)
plt.plot(Hin_node, slope * Hin_node + intercept, color='red', linestyle='--')
plt.xlabel("integration component")
plt.ylabel("FC with left LC")

plt.show()








from nilearn import image,plotting,datasets
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# bold_img_4d = nib.load('conc_data_brain_filt_reg_20081028MT.img')


lc_mask = nib.load("LCmetaMask_bilateral_MNI05_s01f_plus50.nii")
bf_mask = nib.load("BF-2005_MNI_resliced_to_1.5mm.nii")
AAL_mask = nib.load("AAL.nii") 

# Make a copy of the data to modify
modified_data = AAL_mask.get_fdata().copy()
# Set labels 91 to 116 to 0
modified_data[np.isin(modified_data, np.unique(modified_data)[-26:])] = 0
# Create a new NIfTI image with the modified data
AAL90_mask = nib.Nifti1Image(modified_data, affine=AAL_mask.affine, header=AAL_mask.header)


mni_template = datasets.load_mni152_template()


data = image.mean_img("conc_data_brain_filt_reg_20090602AM.img", copy_header=True)
fdata = data.get_fdata()
print("Min:", np.min(fdata))
print("Max:", np.max(fdata))
print("Mean:", np.mean(fdata))



#%% LC AND BF


fig = plt.figure(1)
fig.clf()
display= plotting.plot_epi(data, title="LC and BF mask over BOLD average",
                  display_mode="xz",
                  cut_coords = (4.7,-23),
                   vmin=np.quantile(fdata[fdata!=0],0.01),
                    vmax=np.quantile(fdata[fdata!=0],0.9),
                    colorbar=False,
                  figure=fig)

display.add_overlay(bf_mask, cmap="Greens",vmin=0,vmax=1.3)#, threshold=0.1, alpha=0.8)
display.add_overlay(lc_mask, cmap="Reds",vmin=0,vmax=1.15)#, threshold=0.1, alpha=0.8)
display.add_overlay(AAL90_mask, cmap="Blues",vmin=0,vmax=1,transparency=0.6)#, threshold=0.1, alpha=0.8)

plt.savefig("masks_over_BOLD_20090602AM.svg",dpi=300)
plt.show()

halt

#%%BF
fig = plt.figure(2)
fig.clf()
display= plotting.plot_epi(data, title="BF mask over BOLD average",
                  display_mode="yx",
                  cut_coords = (0,7),
                  vmin=499,
                   colorbar=False,
                  figure=fig)

display.add_overlay(bf_mask, cmap="Oranges",vmin=0,vmax=1.5)#, threshold=0.1, alpha=0.8)
display.add_overlay(AAL90_mask, cmap="Blues",vmin=0,vmax=1,transparency=0.6)#, threshold=0.1, alpha=0.8)
plt.show()

#%% LC over MNI 
fig = plt.figure(3)
fig.clf()
display = plotting.plot_anat(
    bg_img=mni_template,
    title="LC mask over T1 template",
    colorbar=False,
    display_mode="xz",
    cut_coords = (4.7,-23),
    black_bg=False,  # Set background to white
    figure=fig
)
display.add_overlay(lc_mask, cmap="Reds",vmin=0,vmax=1.2)#, threshold=0.1, alpha=0.8)
display.add_overlay(AAL90_mask, cmap="Blues",vmin=0,vmax=1,transparency=0.5)#, threshold=0.1, alpha=0.8)
plt.show()

#%% BF over MNI

fig = plt.figure(4)
fig.clf()
display = plotting.plot_anat(
    bg_img=mni_template,
    title="BF mask over T1 template",
    display_mode="yx",
    cut_coords = (0,7),
    black_bg=False,  # Set background to white
    figure=fig
)
display.add_overlay(bf_mask, cmap="Oranges",vmin=0,vmax=2)#, threshold=0.1, alpha=0.8)
display.add_overlay(AAL90_mask, cmap="Blues",vmin=0,vmax=1,transparency=0.6)#, threshold=0.1, alpha=0.8)
plt.show()

#%% plot both 

fig = plt.figure(5)
fig.clf()
display = plotting.plot_anat(
    bg_img=mni_template,
    title="BF and LC mask over T1 template",
    display_mode="xz",
    cut_coords = (4.7,-23),
    black_bg=False,  # Set background to white
    draw_cross = False,
    figure=fig,
)
display.add_overlay(bf_mask, cmap="Greens",vmin=0,vmax=1.1)#, threshold=0.1, alpha=0.8)
display.add_overlay(lc_mask, cmap="Reds",vmin=0,vmax=1.15)#, threshold=0.1, alpha=0.8)
display.add_overlay(AAL90_mask, cmap="Blues",vmin=0,vmax=1,transparency=0.6)#, threshold=0.1, alpha=0.8)
# plt.savefig("LC_and_BF_over_mni.svg",dpi=300)
plt.show()




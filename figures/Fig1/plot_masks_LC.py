from nilearn import image, plotting, datasets
import nibabel as nib

# Load BOLD data (4D)
bold_img_4d = nib.load('conc_data_brain_filt_reg_20090602AM.img')
bold_img_3d = image.mean_img(bold_img_4d)

lc_mask = nib.load("LCmetaMask_bilateral_MNI05_s01f_plus50.nii")
lc_mask_resampled = image.resample_to_img(lc_mask, bold_img_3d,
                                          force_resample=True,interpolation="nearest")

display = plotting.plot_anat(
    bg_img=bold_img_3d,
    title="LC Masks",
    display_mode='xz',  # 'x' for sagittal, 'z' for axial
    cut_coords=(0, -24),  # Adjust coordinates as needed
    black_bg=False  # Set background to white
)
display.add_overlay(lc_mask, cmap="jet_r")#, threshold=0.1, alpha=0.8)

# Show the plot
plotting.show()

#%%

from nilearn import plotting

plotting.plot_anat(bg_img=bold_img_3d, title="Visual Check: LC on BOLD")
plotting.plot_roi(lc_mask_resampled, bg_img=bold_img_3d, title="LC mask on BOLD")
plotting.show()

#%%

plotting.plot_roi(
    roi_img=lc_mask,
    bg_img=bold_img_3d,
    title="LC mask on MNI152 anatomy",
    cut_coords=(0, -36, -26),  # sagittal (x=0), coronal (y=-36), axial (z=-26)
    display_mode="ortho",  # shows all 3 views
    cmap="hot"
)
plotting.show()
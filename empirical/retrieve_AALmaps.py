import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def reord(vec):
    left = range(0,90,2)
    right = range(1,90,2)
    ids = list(left)+list(right[::-1])
    return vec[ids]

    
indices = list(range(90))
left = indices[::2]
right = indices[1::2][::-1]
sort = left+right

full_AALlabels = pd.read_csv("ROI_MNI_V4.txt",sep="\t",names = ["initials","label","num"]).iloc[:90]
##just to check alignment
full_AALlabels = full_AALlabels.iloc[sort].reset_index(drop=True)
cerebral_AAL = full_AALlabels["label"].values

infolder = "masks_for_extraction/"
outfolder = "maps/"

##############PART ONLY FOR VAT
name = "VAChT_feobv_hc18_aghourian_sum.nii"
receptor = infolder+name
img = nib.load(receptor)
atlas_filename = infolder+'AAL.nii'
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, 
                            memory='nilearn_cache', verbose=5)
r1 = masker.fit_transform(img)


##reorder to symmetrized and normalize
dist = reord(r1.flatten()[:90])
dist = (dist-dist.min())/(dist.max()-dist.min())
dist = dist/dist.mean()
np.save(outfolder+"DIST_VAChT_feobv_hc18_aghourian.npy",dist)

################NOW FOR LC PROJ

###check normalize and shuffle
data_lc = pd.read_csv(infolder+"AAL_LCTractography.csv",names = range(116),decimal=",")
dist = reord(data_lc.values.mean(axis=0)[:90])
maxi_to_take = np.sort(dist)[-5]
##areas of thalamus, can be obtained from the labels
dist[37] = maxi_to_take
dist[38] = maxi_to_take
dist[51] = maxi_to_take
dist[52] = maxi_to_take

dist = (dist-dist.min())/(dist.max()-dist.min())
dist = dist/dist.mean()

np.save(outfolder+"DIST_LC_proj.npy",dist)



####################now shuffled symmetrically for each of the vectors
shuffled_idx = np.zeros(90)
shuffled_l = np.random.permutation(np.array(range(45)))
shuffled_r = np.arange(45,90,1)
for i in range(45):
    shuffled_r[-i-1] = 89-shuffled_l[i]

shuffled_idx[:45] = shuffled_l
shuffled_idx[45:] = shuffled_r
shuffled_idx = shuffled_idx.astype(int)


shuffled_dist = dist[shuffled_idx]
shuffled_labels = cerebral_AAL[shuffled_idx]

# np.save("DIST_VAChT_feobv_hc18_aghourian.npy",dist)
np.save("SHUFFLED_SYMM_DIST_LC_proj.npy",shuffled_dist)
np.save("SHUFFLED_SYMM_LABELS_LC_proj.npy",shuffled_labels)


print(receptor, f"mean = {dist.mean():.3f}")



###############some plotting to check
l = dist[:45];shuffled_l = shuffled_dist[:45]
r = dist[45:90];shuffled_r = shuffled_dist[45:]


plt.figure(1)
plt.clf()
plt.suptitle("dist_proj_LC",weight="bold",fontsize=25)
plt.subplot(2,2,1)
plt.bar(range(45),l,label="left")
  ##izquierda
plt.legend(fontsize=20)
plt.grid()
plt.xticks([0,10,20,30,40,45],fontsize=23,weight="bold")
# plt.xlabel("SCHAEFER ROI",fontsize=25,weight="bold")
# plt.xticks([0,10,20,30,40,50],[])
plt.ylabel("normalized density",fontsize=23,weight="bold")
plt.yticks([0,0.5,1],["0","0.5","1"],fontsize=23,weight="bold")

plt.subplot(2,2,2)
plt.bar(range(45,90),r,label="right")
plt.legend(fontsize=20)
plt.grid()
# plt.xticks([50,60,70,80,90,100],fontsize=23,weight="bold")
plt.xlabel("AAL",fontsize=25,weight="bold")
# plt.ylabel("normalized density",fontsize=23,weight="bold")
plt.yticks([0,0.5,1],["0","0.5","1"],fontsize=23,weight="bold")
plt.tight_layout()
plt.show()

##shuffled
plt.subplot(2,2,3)
plt.bar(range(45),shuffled_l,label="left shuffled")
  ##izquierda
plt.legend(fontsize=20)
plt.grid()
plt.xticks([0,10,20,30,40,45],fontsize=23,weight="bold")
# plt.xlabel("SCHAEFER ROI",fontsize=25,weight="bold")
# plt.xticks([0,10,20,30,40,50],[])
plt.ylabel("normalized density",fontsize=23,weight="bold")
plt.yticks([0,0.5,1],["0","0.5","1"],fontsize=23,weight="bold")

plt.subplot(2,2,4)
plt.bar(range(45,90),shuffled_r,label="right shuffled")
plt.legend(fontsize=20)
plt.grid()
# plt.xticks([50,60,70,80,90,100],fontsize=23,weight="bold")
plt.xlabel("AAL",fontsize=25,weight="bold")
# plt.ylabel("normalized density",fontsize=23,weight="bold")
plt.yticks([0,0.5,1],["0","0.5","1"],fontsize=23,weight="bold")
plt.tight_layout()
plt.show()

#%% DK 

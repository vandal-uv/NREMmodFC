import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nb
from neuromaps.datasets import fetch_fslr
from surfplot import Plot
from surfplot.plotting import Plot
# import os
# from PIL import Image
# from svgutils.compose import Figure, SVG


mapfolder = "../../maps/"
maps = {"LC":np.load(mapfolder+"DIST_LC_proj.npy"),
        "VAT":np.load(mapfolder+"DIST_VAChT_feobv_hc18_aghourian.npy")}

# mapNA = 
# mapACh = 
labels = pd.read_csv("../../sorted_AAL_labels.txt")["label"].values

def reord(mapa):
    "we take LlrR to lrlrlrlr"
    out = np.copy(mapa)
    out[::2] = mapa[:45]
    out[1::2] = mapa[45:][::-1]
    return out
    


reord_labels = reord(labels)
print([(i,reord_labels[i]) for i in range(len(reord_labels))])

#%%ploteamos en cerebros!


def plot_brain(name,vmin,vmax):
    
    this_map = reord(maps[name])
    
    this_map = np.delete(this_map, np.array([70, 71, 72, 73, 74, 75, 76, 77]))
    these_labels = np.delete(reord_labels,np.array([70, 71, 72, 73, 74, 75, 76, 77]))
    this_map /= this_map.max()
    this_map +=0.00001
    
    lh_labels_gii = nb.load(mapfolder+'AAL.32k.L.label.gii')
    lh_labels = lh_labels_gii.darrays[0].data.astype(int)  # Get label index per vertex
    
    # Do the same for the right hemisphere
    rh_labels_gii = nb.load(mapfolder+'AAL.32k.R.label.gii')
    rh_labels = rh_labels_gii.darrays[0].data.astype(int)
    
    Ds_left = this_map[::2]
    Ds_right = this_map[1::2]
    
    # Map each vertex to its corresponding value
    lh_vertex_data = np.zeros_like(lh_labels, dtype=float)
    for i in range(41):
        lh_vertex_data[lh_labels == i+1] = Ds_left[i]    
    
    rh_vertex_data = np.zeros_like(rh_labels, dtype=float)
    for i in range(41):
        rh_vertex_data[rh_labels == i+1] = Ds_right[i]
        
    ##we normalize to plot between -1 and 1
    maxi = np.max(np.abs(np.append(lh_vertex_data,rh_vertex_data)))
    lh_vertex_data /= maxi
    rh_vertex_data /= maxi
    
    
    surfaces = fetch_fslr()
    lh, rh = surfaces['inflated']
    p = Plot(lh, rh,
              views="lateral",
              layout="row",
              mirror_views=True,
             zoom=1.25,
             size=(1200, 600))
    p.add_layer({'left': lh_vertex_data, 'right': rh_vertex_data}, 
                cmap = 'Reds',cbar=True, color_range = (vmin, vmax),
                zero_transparent=True)
        
    fig = p.build()
    plt.savefig(name+"_brain.svg",dpi=300)
    
    
plot_brain("VAT",vmin=0,vmax=0.6)

plot_brain("LC",vmin=0,vmax=0.25)

halt
#%% generate a colormap

import matplotlib.colorbar as cbar
import matplotlib.cm as cm
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

# Create a dummy mappable object with Reds colormap
cmap = cm.get_cmap('Reds')
norm = plt.Normalize(vmin=0, vmax=1)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Add horizontal colorbar
cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
cbar.set_ticks([])  # remove ticks
cbar.ax.text(0, -1.5, 'min', va='center', ha='left', fontsize=10)
cbar.ax.text(1, -1.5, 'max', va='center', ha='right', fontsize=10, transform=cbar.ax.transAxes)

for spine in cbar.ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1)
plt.savefig("Reds_colorbar.svg",dpi=300)
plt.show()

#%% plot all 90 areas to see where it's strongest
labelsais=15


plt.figure(2)
plt.clf()
plt.suptitle("all 90 areas")

ax1 = plt.subplot(211)
# ax1.set_title("left hemisphere")
ax1.bar(range(0,3*45,3),maps["VAT"][:45],color="tab:blue",label="VAT left")
ax1.set_ylabel("VAT left",color="tab:blue",weight="bold",fontsize=labelsais)
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.bar(range(1,3*45+1,3),maps["LC"][:45],color="tab:orange",label="LC proj left")
ax2.set_ylabel("LC proj left",color="tab:orange",weight="bold",fontsize=labelsais)
ax1.set_xticks(())
ax2.legend(loc="upper right")

##right_hemisphere
ax1 = plt.subplot(212)
# ax1.set_title("right hemisphere")
ax1.bar(range(0,3*45,3),maps["VAT"][45:][::-1],color="tab:blue",label="VAT right")
ax1.set_ylabel("VAT right",color="tab:blue",weight="bold",fontsize=labelsais)
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.bar(range(1,3*45+1,3),maps["LC"][45:][::-1],color="tab:orange",label="LC proj right")
ax2.set_ylabel("LC proj right",color="tab:orange",weight="bold",fontsize=labelsais)
ax1.set_xticks(0.5+np.array(range(0,3*45,3)), labels[45:][::-1],rotation=90)
ax2.legend(loc="upper right")


plt.tight_layout()
plt.show()

#%%
plt.figure(3)
plt.clf()
plt.suptitle("only 82 areas (out caudate, putamen, pallidum and thalamus)")


this_map = reord(maps["VAT"])
this_map = np.delete(this_map, np.array([70, 71, 72, 73, 74, 75, 76, 77]))
this_map /= this_map.max()
ax1 = plt.subplot(211)
# ax1.set_title("left hemisphere")
ax1.bar(range(0,3*41,3),this_map[::2],color="tab:blue",label="VAT left")
ax1.set_ylabel("VAT left",color="tab:blue",weight="bold",fontsize=labelsais)
ax1.legend(loc="upper left")


this_map = reord(maps["LC"])
this_map = np.delete(this_map, np.array([70, 71, 72, 73, 74, 75, 76, 77]))
this_map /= this_map.max()
ax2 = ax1.twinx()
ax2.bar(range(1,3*41+1,3),this_map[::2],color="tab:orange",label="LC proj left")
ax2.set_ylabel("LC proj left",color="tab:orange",weight="bold",fontsize=labelsais)
ax1.set_xticks(())
ax2.legend(loc="upper right")

##right_hemisphere
this_map = reord(maps["VAT"])
this_map = np.delete(this_map, np.array([70, 71, 72, 73, 74, 75, 76, 77]))
this_map /= this_map.max()
ax1 = plt.subplot(212)
# ax1.set_title("left hemisphere")
ax1.bar(range(0,3*41,3),this_map[1::2],color="tab:blue",label="VAT left")
ax1.set_ylabel("VAT right",color="tab:blue",weight="bold",fontsize=labelsais)
ax1.legend(loc="upper left")

this_map = reord(maps["LC"])
this_map = np.delete(this_map, np.array([70, 71, 72, 73, 74, 75, 76, 77]))
this_map /= this_map.max()
ax2 = ax1.twinx()
ax2.bar(range(1,3*41+1,3),this_map[1::2],color="tab:orange",label="LC proj left")
ax2.set_ylabel("LC proj right",color="tab:orange",weight="bold",fontsize=labelsais)

ax2.legend(loc="upper right")

labels = reord(pd.read_csv("../sorted_AAL_labels.txt")["label"].values)
labels = np.delete(labels, np.array([70, 71, 72, 73, 74, 75, 76, 77]))
ax1.set_xticks(0.5+np.array(range(0,3*41,3)), labels[::2],rotation=90)

plt.tight_layout()
plt.show()

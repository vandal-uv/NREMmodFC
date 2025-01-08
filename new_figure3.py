#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:20:50 2024

@author: flehue
"""


import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import utils
from plot_violins import violin_plot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import ttest_ind as ttest
import pickle 
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')




lis1 = ["HOMO",
        "DIST_FC_BF",
        "SHUFFLED_DIST_FC_BF",
        "DIST_A4B2_flubatine_hc30_hillmer_tal_corrected",
        "SHUFFLED_DIST_A4B2_flubatine_hc30_hillmer_tal_corrected",
        "DIST_M1_lsn_hc24_naganawa",
        "SHUFFLED_DIST_M1_lsn_hc24_naganawa"]

lis2 = ["HOMO", 
        "DIST_FC_LC",
        "SHUFFLED_DIST_FC_LC",
        "DIST_LC_proj",
        "SHUFFLED_DIST_LC_proj",
        "DIST_NAT_MRB_hc77_ding",
        "SHUFFLED_DIST_NAT_MRB_hc77_ding"]

optimals_homo = {"W":  (0.16,0.0,7.68,0.0),
                 "N1": (0.16,0.04,7.68,0.0),
                 "N2": (0.16,0.0,7.68,0.0), 
                 "N3":  (0.16,-0.04,7.68,0.04)}

optimals_map = {"W":  (0.16,-0.02,7.68,-0.02),
                 "N1": (0.16,0.18,7.68,-0.02),
                 "N2": (0.16,0.02,7.68,-0.04), 
                 "N3":  (0.16,0.02,7.68,-0.12)}

optimals_shuf = {"W":  (0.16,0.0,7.68,0.0),
                 "N1": (0.16,0.0,7.68,0.04),
                 "N2": (0.16,0.0,7.68,0.0), 
                 "N3":  (0.16,0.0,7.68,-0.04)}
   
bas = [0.16,7.68];names="Global Coupling G",r"Input-Output Slope $\sigma_E$"

states = ("W","N1","N2","N3")
var_ex = {"euccorr":"min","e":"min","ssim":"max","corr":"max"}
# var2see = "euccorr"

def extract(filepath,xv="delta_G",yv="delta_sigma",var2see="euccorr",C1=0,w=1,thx=(-0.08,0.2),thy =(-0.2,0.08)):
    data_pre = pd.read_csv(filepath)
    data_pre[xv] = np.round(data_pre[xv].values,decimals=2)
    data_pre[yv] = np.round(data_pre[yv].values,decimals=2)
    nseed = 50#len(data_pre["seed"].unique())
    
    data_pre = data_pre[(thx[0]<=data_pre[xv]) & (data_pre[xv]<=thx[1]) & (thy[0]<=data_pre[yv]) & (data_pre[yv]<=thy[1])]
    
    for st in states:
        data_pre[f"euccorr{st}"] = data_pre[f"e{st}"]/(C1+w*abs(data_pre[f"corr{st}"]))
    data = data_pre.groupby([xv,yv]).agg(np.nanmean).reset_index()
    
    
    #axis data
    x_vals = np.sort(data[xv].unique());y_vals = np.sort(data[yv].unique())
    # print(x_vals)
    lenx,leny = len(x_vals),len(y_vals)
    
    #we fill the matrices
    plotmats,coors_o,vals_o,violins_o = [],[],[],[]
    for idd,var in enumerate([var2see+s for s in states]+["mean","sync","meta"]):
        plotmat = np.zeros((leny,lenx))
        for i,d2 in enumerate(y_vals):
            plotmat[i,:] = data[data[yv]==d2].sort_values(xv)[var].values
        if var_ex[var2see] =="min":
            iy,ix = np.unravel_index(plotmat.argmin(), plotmat.shape)
        elif var_ex[var2see] =="max":
            iy,ix = np.unravel_index(plotmat.argmax(), plotmat.shape)
        xo,yo,oval = x_vals[ix],y_vals[iy],plotmat[iy,ix]
        
        violin = data_pre[(data_pre[xv]==xo) & (data_pre[yv]==yo)][var].values
        violin = np.array(list(violin)+ (nseed-len(violin))*[violin.mean()]) #rellenamos
        
        plotmats.append(plotmat);coors_o.append((ix,iy));vals_o.append((xo,yo,oval));violins_o.append(violin)
        
    #raw violin
    
    #return
    output = {"x_vals":x_vals,"y_vals":y_vals,
              "plotmats":plotmats,"coors_o":coors_o,
              "vals_o":vals_o,"violins_o":violins_o}
    # print(output.keys())
    return output
    
    


#%%

states = ("W","N1","N2","N3")
var2see = "euccorr"
xv, yv = "delta_G","delta_sigma"

# filename_homo = "data/final_refined_sweep_delta_both_WhomoNEW_fromG0.14_sigma7.7_maps_0_0_4nov_50iter.txt"
# filename_map = "data/extend_sweep_delta_both_SC_MAPS_7_3_fromG0.14_sigma7.7_maps_7_3_6nov.txt"
# filename_shuf = "data/sweep_delta_SHUFFLED_SC_MAPS_7_3_fromG0.14_sigma7.7_maps_8_4_21oct_50iter.txt"

filename_homo = "output/sweep_delta_homoW_fromG0.16_sigma7.68_maps_0_0_9dic24_50iter.txt"
filename_map = "output/sweep_deltamaps_from_supposed_homoW_fromG0.16_sigma7.68_maps_7_3_9dic24_50iter.txt"
filename_shuf = "output/sweep_deltaSHUFFLED_from_supposed_homoW_fromG0.16_sigma7.68_maps_8_4_9dic24_50iter.txt"

output_homo = extract(filename_homo)
output_map = extract(filename_map)
output_shuf = extract(filename_shuf)
print(f"homo {output_homo['vals_o'][:4]}")
print(f"map {output_map['vals_o'][:4]}")
print(f"shuf {output_shuf['vals_o'][:4]}")


output_keys = ('x_vals', 'y_vals', 'plotmats', 'coors_o', 'vals_o','violins_o')

colors = ["crimson","orange","forestgreen","navy"]
    
#%% bondad de ajuste en el optimo 

df_violin = pd.DataFrame()
for s,st in enumerate(states):
    df_violin[st] = output_homo["violins_o"][s]
    
    
    
#%% figure 
from matplotlib.patches import FancyArrowPatch


alfa = 0.6
titlesais = 20
labelsais = 18
ticsais = 15
legendsais = 18
   
#arrow shit
dd = 0.003

colors = ["crimson","orange","forestgreen","navy"]
 
plt.figure(1)
plt.clf()
plt.gcf().set_size_inches(12, 18)
# plt.suptitle("Homogeneous modulation")

ax = plt.subplot2grid((3,2),(0,0))
ax.set_title("Homogeneous Modulation\n(homo)",fontsize=titlesais,weight="bold")
plt.xlabel(r"$G$"+"-variation "+r"$\delta_G$",fontsize=labelsais)
plt.ylabel(r"$\sigma$"+"-variation "+r"$\delta_\sigma$",fontsize=labelsais)
init = np.array([optimals_homo["W"][i] for i in (1,3)])
for s,state in enumerate(states):
    Go,sigmaEo = [optimals_homo[state][i] for i in (1,3)]
    
    if s>0:
        end = np.array([Go,sigmaEo])
        width=3.2
        if s ==1:
            arrow = FancyArrowPatch(
                                (init[0], init[1]-0.005), (end[0],end[1]-0.005),
                                arrowstyle=f"-|>,head_width={2*width},head_length={2*width}",
                                linewidth=width,
                                color="tab:blue")
        else:
            arrow = FancyArrowPatch(
                                (init[0], init[1]), (end[0],end[1]),
                                arrowstyle=f"-|>,head_width={2*width},head_length={2*width}",
                                linewidth=width,
                                color="tab:blue")
        ax.add_patch(arrow)
        init = end
# (0,(5,10))
    plt.scatter(Go,sigmaEo,color=colors[s],edgecolors="black",s=200,label=state,alpha=alfa)
ax.set_xticks((-0.04,-0.02,0,0.02,0.04),(-0.04,-0.02,0,0.02,0.04),fontsize=ticsais)
ax.set_yticks((-0.04,-0.02,0,0.02,0.04),(-0.04,-0.02,0,0.02,0.04),fontsize=ticsais)
# plt.legend(fontsize=ticsais,loc= "upper right")

##########heatmap figure original
ax = plt.subplot2grid((3,2),(0,1))
ax.set_title("Heterogenous Modulation\n(map)",fontsize=titlesais,weight="bold")
plt.xlabel(r"$G$"+"-variation "+r"$\delta_G$",fontsize=labelsais)
plt.ylabel(r"$\sigma$"+"-variation "+r"$\delta_\sigma$",fontsize=labelsais)
init = np.array([optimals_map["W"][i] for i in (1,3)])
for s,state in enumerate(states):
    Go,sigmaEo = [optimals_map[state][i] for i in (1,3)]
    if s>0:
        #la flecha 
        end = np.array([Go,sigmaEo])
        if s<4:
            width=3.2
            arrow = FancyArrowPatch(
                                    (init[0], init[1]), (end[0],end[1]),
                                    arrowstyle=f"-|>,head_width={2*width},head_length={2*width}",
                                    linewidth=width,
                                    color="tab:blue"
                                )
            ax.add_patch(arrow)
        else:
            plt.arrow(init[0],init[1],(end-init)[0],(end-init)[1],
                    length_includes_head=True,linestyle="--",width=0)
        init = end
# (0,(5,10))
    plt.scatter(Go,sigmaEo,color=colors[s],edgecolors="black",s=200,label=state,alpha=alfa)
ax.set_xticks((-0.2,-0.1,0,0.1,0.2),(-0.2,-0.1,0,0.1,0.2),fontsize=ticsais)
ax.set_yticks((-0.15,-0.1,-0.05,0,0.05,0.1),(-0.15,-0.1,-0.05,0,0.05,0.1),fontsize=ticsais)
# ax.set_aspect(3)
# ax.legend(fontsize=ticsais,loc="upper right")


###########modulation panel HOMOOOOOOOOO
ax = plt.subplot2grid((3,2),(1,0))
Gos = [np.array(optimals_homo[st])[1] for st in states]
sigmaEos = [np.array(optimals_homo[st])[3] for st in states]
plt.bar(range(4),Gos,color=colors,alpha=alfa)
plt.plot(range(4),Gos,linestyle="dashed",marker="X",color="black")
plt.ylabel("+ACh <----------> -ACh",fontsize=labelsais)

ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
ax2.sharey(ax)
ax2.set_ylabel("-NA <----------> +NA",fontsize=labelsais)  # we already handled the x-label with ax1
for label in ax2.get_yticklabels():
    label.set_visible(False)
ax.set_yticks((0.04,0.02,0,-0.02,-0.04,-0.06),(0.04,0.02,0,-0.02,-0.04,-0.06),fontsize=ticsais)
plt.bar(range(5,9),sigmaEos,color=colors,alpha=alfa)
plt.plot(range(5,9),sigmaEos,linestyle="dashed",marker="X",color="black")
ax.set_xticks(list(range(4))+list(range(5,9)),2*list(states),fontsize=ticsais)
# plt.tick_params(labeltop=True, labelright=True)
sec = ax.secondary_xaxis(location=0)
sec.set_xticks([1.5,6.5], labels=['\nACh','\nNA'],fontsize=ticsais)
sec.tick_params('x', length=0)

##modulation panel MAPPPPPPPPPPPPP
ax = plt.subplot2grid((3,2),(1,1))
# plt.title("Modulation")
Gos = [output_map["vals_o"][s][0] for s in range(4)]
sigmaEos = [output_map["vals_o"][s][1] for s in range(4)]
plt.bar(range(4),Gos,color=colors,alpha=alfa)
plt.plot(range(4),Gos,linestyle="dashed",marker="X",color="black")
plt.ylabel("+ACh <----------> -ACh",fontsize=labelsais)
ax.set_yticks((-0.1,-0.05,0,0.05,0.1,0.15,0.2),(-0.1,-0.05,0,0.05,0.1,0.15,0.2),fontsize=ticsais)
ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
ax2.sharey(ax)
ax2.set_ylabel("-NA <----------> +NA",fontsize=labelsais)  # we already handled the x-label with ax1
for label in ax2.get_yticklabels():
    label.set_visible(False)
plt.bar(range(5,9),sigmaEos,color=colors,alpha=alfa)
plt.plot(range(5,9),sigmaEos,linestyle="dashed",marker="X",color="black")
ax.set_xticks(list(range(4))+list(range(5,9)),2*list(states),fontsize=ticsais)
# plt.tick_params(labeltop=True, labelright=True)
sec = ax.secondary_xaxis(location=0)
sec.set_xticks([1.5,6.5], labels=['\nACh', '\nNA'],fontsize=ticsais)
sec.tick_params('x', length=0)
# ax.spines[['right', 'top']].set_visible(False)

##violin plots
ax= plt.subplot2grid((3,1),(2,0))
positions = [0,1,2,
             4,5,6,
             8,9,10,
             12,13,14]
mods = ("homo","map","shuf","emp")
for s,st in enumerate(states):
    ##simulated
    # if s==0:
    ax.set_ylabel("Distance to empirical",fontsize=labelsais)
    # ax.set_title(st,weight="bold",fontsize=titlesais)
    violin_plot(ax, [output["violins_o"][s] for output in (output_homo,output_map,output_shuf)],
                color_names= [colors[s],colors[s],colors[s]],
                alpha_violin = 0.5,
                inds=positions[3*s:3*(s+1)])
    ax.set_ylim([8,18])
    ax.set_yticks((10,15,20,25,30,35),(10,15,20,25,30,35),fontsize=ticsais)
    ax.set_xticks(positions,list(mods[:3])*4,fontsize=labelsais-3,rotation=45)
    ax.spines[['right', 'top']].set_visible(False)
patches = [mpatches.Patch(color=colors[s], label=states[s],alpha=alfa) for s in range(4)]
ax.legend(handles=patches,fontsize=legendsais,loc=(0.25,1),ncol=4)
# ax.legend()


plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.05)
plt.show()
plt.savefig("figures/newfig3.png",dpi=300)
# 


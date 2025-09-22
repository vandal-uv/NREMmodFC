#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:20:50 2024

@author: flehue
"""


import sys
sys.path.append("../")
sys.path.append("../../")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import utils
from plot_violins import violin_plot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind as ttest
import pickle 
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')


#################THIS WERE OBTAINED IN heatmaps.py
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


def plot_significance_line(ax, x_start, x_end, y, horizontal_margin=0.1, text="*", fontsize=12, line_kwargs=None):
    """
    Plots a significance horizontal line with an asterisk in the middle.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to plot on.
        x_start (float): The start of the horizontal line (leftmost x-coordinate).
        x_end (float): The end of the horizontal line (rightmost x-coordinate).
        y (float): The y-coordinate of the horizontal line.
        horizontal_margin (float): The fraction of the line length to leave as margins on both sides.
        text (str): The text to display at the center of the line (e.g., "*").
        fontsize (int): Font size of the text.
        line_kwargs (dict): Additional keyword arguments for the line plot.
    """
    if line_kwargs is None:
        line_kwargs = {"color": "black", "linewidth": 1.5}
    
    # Calculate adjusted start and end points with margins
    line_length = x_end - x_start
    adjusted_x_start = x_start + line_length * horizontal_margin
    adjusted_x_end = x_end - line_length * horizontal_margin
    center_x = (adjusted_x_start + adjusted_x_end) / 2
    
    # Plot the horizontal line
    ax.plot([adjusted_x_start, adjusted_x_end], [y, y], **line_kwargs)
    
    # Add the text at the center of the line
    ax.text(center_x, y+1.15, text, ha='center', va='center', fontsize=fontsize, color=line_kwargs.get("color", "black"))


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

filename_homo = "../../output/sweep_delta_homoW_fromG0.16_sigma7.68_maps_0_0_9dic24_50iter.txt"
filename_map = "../../output/sweep_deltamaps_from_supposed_homoW_fromG0.16_sigma7.68_maps_1_1_9dic24_50iter.txt"
filename_shuf = "../../output/sweep_deltaSHUFFLED_from_supposed_homoW_fromG0.16_sigma7.68_maps_2_2_9dic24_50iter.txt"

output_homo = extract(filename_homo)
output_map = extract(filename_map)
output_shuf = extract(filename_shuf)
print(f"homo {output_homo['vals_o'][:4]}")
print(f"map {output_map['vals_o'][:4]}")
print(f"shuf {output_shuf['vals_o'][:4]}")


output_keys = ('x_vals', 'y_vals', 'plotmats', 'coors_o', 'vals_o','violins_o')

colors = ["crimson","orange","forestgreen","navy"]


p_vals = []
cohen_ds = []
for s,st in enumerate(states):
    dist_homo,dist_map,dist_shuf = [dic["violins_o"][s] for dic in (output_homo,output_map,output_shuf)]
    
    p1,p2 = ttest(dist_homo,dist_map)[1],ttest(dist_shuf,dist_map)[1]
    p_vals += [p1,p2]
    
    d1,d2 = utils.cohen_d(dist_homo,dist_map),utils.cohen_d(dist_shuf,dist_map)
    cohen_ds += [d1,d2]
p_vals = multipletests(p_vals, alpha=0.05, method='fdr_bh', 
                            is_sorted=False, returnsorted=False)[1]

    
    
#%% figure 
from matplotlib.patches import FancyArrowPatch


alfa = 0.6
titlesais = 20
labelsais = 18
ticsais = 15
legendsais = 15
   
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
    if s in (0,2):
        plt.text(Go-0.003,sigmaEo+0.004,"W,N2",fontsize=14,weight="bold")
    elif s ==3:
        plt.text(Go-0.003,sigmaEo-0.005,state,fontsize=14,weight="bold")
    else:
        plt.text(Go-0.003,sigmaEo+0.002,state,fontsize=14,weight="bold")
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
    plt.text(Go+0.009,sigmaEo+0.009,state,fontsize=14,weight="bold")
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
# plt.ylabel("+ACh <----------> -ACh",fontsize=labelsais)
plt.ylabel(r"$\delta G$",fontsize=labelsais)

ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
ax2.sharey(ax)
# ax2.set_ylabel("-NA <----------> +NA",fontsize=labelsais)  # we already handled the x-label with ax1
ax2.set_ylabel(r"$\delta \sigma$",fontsize=labelsais)  # we already handled the x-label with ax1
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
# plt.ylabel("+ACh <----------> -ACh",fontsize=labelsais)
plt.ylabel(r"$\delta G$",fontsize=labelsais)
ax.set_yticks((-0.1,-0.05,0,0.05,0.1,0.15,0.2),(-0.1,-0.05,0,0.05,0.1,0.15,0.2),fontsize=ticsais)
ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
ax2.sharey(ax)
# ax2.set_ylabel("-NA <----------> +NA",fontsize=labelsais)  # we already handled the x-label with ax1
ax2.set_ylabel(r"$\delta \sigma$",fontsize=labelsais)  # we already handled the x-label with ax1
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
    
    ##aqui ploteamos todas las modalidades para una etapa dada
    violin_plot(ax, [output["violins_o"][s] for output in (output_homo,output_map,output_shuf)],
                color_names= [colors[s],colors[s],colors[s]],
                alpha_violin = 0.5,
                inds=positions[3*s:3*(s+1)])
    ax.set_ylim([8,18])
    ax.set_yticks((10,15,20,25,30,35),(10,15,20,25,30,35),fontsize=ticsais)
    ax.set_xticks(positions,list(mods[:3])*4,fontsize=labelsais-3,rotation=45)
    ax.spines[['right', 'top']].set_visible(False)
patches = [mpatches.Patch(color=colors[s], label=states[s],alpha=alfa) for s in range(4)]
ax.legend(handles=patches,fontsize=legendsais,loc=(0.5,1),ncols = 4)
# ax.legend()
##significancias
hmo = 0.1
hmi = 0.1
Dsize = 12
plot_significance_line(ax,0-hmo,1-hmi,25,text = f"D={cohen_ds[0]:.2f}\n**",fontsize=Dsize) ##W
plot_significance_line(ax,1+hmi,2+hmo,25,text = f"D={cohen_ds[1]:.2f}\n**",fontsize=Dsize) ##W

plot_significance_line(ax,4-hmo,5-hmi,34,text = f"D={cohen_ds[2]:.2f}\n****",fontsize=Dsize) ##W
plot_significance_line(ax,5+hmi,6+hmo,34,text = f"D={cohen_ds[3]:.2f}\n****",fontsize=Dsize) ##W

plot_significance_line(ax,8-hmo,9-hmi,30,text = f"D={cohen_ds[4]:.2f}\n*",fontsize=Dsize) ##W
plot_significance_line(ax,9+hmi,10+hmo,30,text = f"D={cohen_ds[5]:.2f}\n*",fontsize=Dsize) ##W

plot_significance_line(ax,12-hmo,13-hmi,31,text = f"D={cohen_ds[6]:.2f}\n***",fontsize=Dsize) ##W
plot_significance_line(ax,13+hmi,14+hmo,31,text = f"D={cohen_ds[7]:.2f}\n***",fontsize=Dsize) ##W

plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.05)
plt.show()
# plt.savefig("newfig3.svg",dpi=300)
# 


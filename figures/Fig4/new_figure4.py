#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:35:37 2024

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
from scipy.stats import ttest_ind as ttest
import pickle 
warnings.filterwarnings('ignore')
# halt


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
# var_ex = {"euccorr":"min","e":"min","ssim":"max","corr":"max"}
# # var2see = "euccorr"

# def extract(filepath,xv="delta_G",yv="delta_sigma",var2see="euccorr",C1=0,w=1,thx=(-0.08,0.2),thy =(-0.2,0.1)):
#     data_pre = pd.read_csv(filepath)
#     data_pre[xv] = np.round(data_pre[xv].values,decimals=2)
#     data_pre[yv] = np.round(data_pre[yv].values,decimals=2)
#     nseed = 50#len(data_pre["seed"].unique())
    
#     data_pre = data_pre[(thx[0]<=data_pre[xv]) & (data_pre[xv]<=thx[1]) & (thy[0]<=data_pre[yv]) & (data_pre[yv]<=thy[1])]
    
#     for st in states:
#         data_pre[f"euccorr{st}"] = data_pre[f"e{st}"]/(C1+w*abs(data_pre[f"corr{st}"]))
#     data = data_pre.groupby([xv,yv]).agg(np.nanmean).reset_index()
    
    
#     #axis data
#     x_vals = np.sort(data[xv].unique());y_vals = np.sort(data[yv].unique())
#     # print(x_vals)
#     lenx,leny = len(x_vals),len(y_vals)
    
#     #we fill the matrices
#     plotmats,coors_o,vals_o,violins_o = [],[],[],[]
#     for idd,var in enumerate([var2see+s for s in states]+["mean","sync","meta"]):
#         plotmat = np.zeros((leny,lenx))
#         for i,d2 in enumerate(y_vals):
#             plotmat[i,:] = data[data[yv]==d2].sort_values(xv)[var].values
#         if var_ex[var2see] =="min":
#             iy,ix = np.unravel_index(plotmat.argmin(), plotmat.shape)
#         elif var_ex[var2see] =="max":
#             iy,ix = np.unravel_index(plotmat.argmax(), plotmat.shape)
#         xo,yo,oval = x_vals[ix],y_vals[iy],plotmat[iy,ix]
        
#         violin = data_pre[(data_pre[xv]==xo) & (data_pre[yv]==yo)][var].values
#         violin = np.array(list(violin)+ (nseed-len(violin))*[violin.mean()]) #rellenamos
        
#         plotmats.append(plotmat);coors_o.append((ix,iy));vals_o.append((xo,yo,oval));violins_o.append(violin)
        
#     #raw violin
    
#     #return
#     output = {"x_vals":x_vals,"y_vals":y_vals,
#               "plotmats":plotmats,"coors_o":coors_o,
#               "vals_o":vals_o,"violins_o":violins_o}
#     # print(output.keys())
#     return output
    
 
#%% mats 

with open('../../output/run_50seeds_output_homo_16dic.pickle', 'rb') as f:
    dic_homo = pickle.load(f)
    
with open('../../output/run_50seeds_output_map_16dic.pickle', 'rb') as f:
    dic_map = pickle.load(f)
    
with open('../../output/run_50seeds_output_shuf_16dic.pickle', 'rb') as f:
    dic_shuf = pickle.load(f)

homo_matts = {st:np.zeros((50,90,90)) for st in states}
map_matts = {st:np.zeros((50,90,90)) for st in states}
shuf_matts = {st:np.zeros((50,90,90)) for st in states}

for key in dic_homo.keys():
    if key != "metainfo":
        s,state = key
        homo_matts[state][s] = dic_homo[key]["sFC"]
        map_matts[state][s] = dic_map[key]["sFC"]
        shuf_matts[state][s] = dic_shuf[key]["sFC"]
        # dic[key] = dic[key]
        
homo_matts = {st:homo_matts[st].mean(axis=0) for st in states}
map_matts = {st:map_matts[st].mean(axis=0) for st in states}
shuf_matts = {st:shuf_matts[st].mean(axis=0) for st in states}
emp_matts = {st:np.loadtxt(f"../../empirical/mean_mat_{st}_8dic24.txt") for st in states}
    
    
#%% figure 
from matplotlib.patches import FancyArrowPatch


alfa = 0.6
titlesais = 20
labelsais = 18
ticsais = 15
legendsais = 13
   
#arrow shit
dd = 0.003

colors = ["crimson","orange","forestgreen","navy"]
 
plt.figure(1)
plt.clf()
plt.gcf().set_size_inches(12, 18)

##matrices
for s,st in enumerate(states):
    
    ##empirical
    ax= plt.subplot2grid((4,3),(s,0))
    # ax.set_title(st,weight="bold",fontsize=titlesais)
    if s ==0:
        plt.title("Empirical FCs\n",fontsize=titlesais)
    elif s ==3:
        plt.xlabel("ROIs",fontsize=titlesais,weight="bold")
    im=plt.imshow(emp_matts[st],vmin=0,vmax=1,cmap="jet")
    plt.xticks([0,44,89],[1,45,90],fontsize=ticsais);plt.yticks([0,44,89],[1,45,90],fontsize=ticsais)
    plt.ylabel(f"ROIs",fontsize=titlesais,weight="bold")
    
    ##simulated homo
    ax= plt.subplot2grid((4,3),(s,1))
    # ax.set_title(st,weight="bold",fontsize=titlesais)
    if s ==0:
        plt.title("Simulated FCs\n"+r"$\bf{(homo)}$",fontsize=titlesais)
    plt.imshow(homo_matts[st],vmin=0,vmax=1,cmap="jet")
    plt.xticks([0,44,89],[1,45,90],fontsize=ticsais);plt.yticks([0,44,89],[1,45,90],fontsize=ticsais)
    
    ##colorbar
    if s==3:
        axins1 = inset_axes(ax,
                        width="100%",  # width = 50% of parent_bbox width
                        height="6.25%",  # height : 5%
                        loc='lower center', borderpad = -5)
                        # bbox_to_anchor = [-1,-1,1,0.025])
        cax = plt.colorbar(im, cax=axins1, orientation="horizontal", ticks=[1, 2, 3])
        cax.set_ticks((0, 0.5, 1))
        ticklabs = ("0","0.5","1")#cax.ax.get_xticklabels()
        cax.ax.set_xticklabels(ticklabs, fontsize=ticsais)
        cax.set_label(r'Pearson Correlation', fontsize = labelsais)
    
    
    ##simulated map
    ax= plt.subplot2grid((4,3),(s,2))
    # ax.set_title(st,weight="bold",fontsize=titlesais)
    if s ==0:
        plt.title("Simulated FCs\n"+r"$\bf{(map)}$",fontsize=titlesais)
    plt.imshow(map_matts[st],vmin=0,vmax=1,cmap="jet")
    plt.xticks([0,44,89],[1,45,90],fontsize=ticsais);plt.yticks([0,44,89],[1,45,90],fontsize=ticsais)
    
    

# plt.figlegend(loc = 'upper left', ncol = 4, fontsize = 12)
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.05)
plt.show()
# plt.savefig("newfig4.svg",dpi=300)
# 


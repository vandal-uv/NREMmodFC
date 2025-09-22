#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:43:46 2024

@author: flehue
"""
import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import ttest_ind as ttest

import warnings
warnings.filterwarnings('ignore')

# d1 = pd.DataFrame()
# d2 = pd.DataFrame()
# for i in range(64):
    # subdata1 = pd.read_csv(f"data/sweep_deltaSHUFFLED_from_supposed_homoW_fromG0.16_sigma7.68_maps_8_4_9dic24_25iter_from0_rank{i}.txt",sep="\t")
    # subdata2 = pd.read_csv(f"data/sweep_deltaSHUFFLED_from_supposed_homoW_fromG0.16_sigma7.68_maps_8_4_9dic24_25iter_from25_rank{i}.txt",sep="\t")
    # d1 = pd.concat((d1,subdata1))
    # d2 = pd.concat((d2,subdata2))
# d = pd.concat((d1,d2))
# print(d.shape)
# d.to_csv("data/sweep_deltaSHUFFLED_from_supposed_homoW_fromG0.16_sigma7.68_maps_8_4_9dic24_50iter.txt",index=False)
# d1.to_csv("data/sweep_deltaSHUFFLED_from_supposed_homoW_fromG0.16_sigma7.68_maps_8_4_9dic24.txt",index=False)

# halt

#%%
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

filename_homo = "output/sweep_delta_homoW_fromG0.16_sigma7.68_maps_0_0_9dic24_50iter.txt"
filename_map = "output/sweep_deltamaps_from_supposed_homoW_fromG0.16_sigma7.68_maps_7_3_9dic24_50iter.txt"
filename_shuf = "output/sweep_deltaSHUFFLED_from_supposed_homoW_fromG0.16_sigma7.68_maps_8_4_9dic24_50iter.txt"


output_homo = extract(filename_homo)
output_map = extract(filename_map)
output_shuf = extract(filename_shuf)
print(f"homo {output_homo['vals_o'][:4]}")
print(f"map {output_map['vals_o'][:4]}")
print(f"shuf {output_shuf['vals_o'][:4]}")
# print(f"homo {output_homo['vals_o'][:4]}")


print([utils.cohen_d(output_map["violins_o"][i],output_homo["violins_o"][i]) for i in range(4)])

output_keys = ('x_vals', 'y_vals', 'plotmats', 'coors_o', 'vals_o','violins_o')

colors = ["crimson","orange","forestgreen","navy"]


#%% lo anterior mas el heatmap de la media.

labelsais = 17
titlesais = 17
ticsais = 13
legendsais = 13
alfa = 0.7

down = 10 ##para saltarse ticks


# poss = (1,2,4,5)

    
plt.figure(1,figsize=(10,14))
plt.clf()
for s,st in enumerate(states):
    var = var2see + st
    #############################HOMO
    cosa ="homo"
    x_vals,y_vals,plotmats,coors_o,vals_o,_ = [output_homo[key] for key in output_keys]
    (ix,iy),(xo,yo,oval) = coors_o[s],vals_o[s]
    xt_pos = range(len(x_vals))[::down];xt = x_vals[::down];yt_pos = range(len(y_vals))[::down];yt = y_vals[::down]
    
    ax = plt.subplot2grid((3,4),(0,s))
    ax.set_title(st,fontsize=titlesais)
    if s==0:
        plt.ylabel(cosa+"\n"+r"$\delta_\sigma$",fontsize=labelsais)
    pmat = plotmats[s]
    plt.imshow(pmat,cmap="jet")#,vmin=pmat.min(),vmax=np.percentile(pmat.flatten(),50))
    # plt.colorbar()
    plt.xticks(xt_pos,xt,rotation=90,fontsize=ticsais);plt.yticks(yt_pos,yt,fontsize=ticsais)
    plt.xticks(xt_pos,xt,rotation=90,fontsize=ticsais);plt.yticks(yt_pos,yt,fontsize=ticsais)
    # plt.xlabel(r"$\delta_G$",fontsize=labelsais);plt.ylabel(r"$\delta_\sigma$",fontsize=labelsais)
    plt.scatter(ix,iy,color="w",edgecolors="black",s=120,label=f"optimal\n{xv}={xo:.2f}\n{yv}={yo:.2f}")
    # plt.legend()
    ax.invert_yaxis()
    ###########################MAP
    cosa = "map"
    x_vals,y_vals,plotmats,coors_o,vals_o,_ = [output_map[key] for key in output_keys]
    (ix,iy),(xo,yo,oval) = coors_o[s],vals_o[s]
    xt_pos = range(len(x_vals))[::down];xt = x_vals[::down];yt_pos = range(len(y_vals))[::down];yt = y_vals[::down]
    ax = plt.subplot2grid((3,4),(1,s))
    if s==0:
        plt.ylabel(cosa+"\n"+r"$\delta_\sigma$",fontsize=labelsais)
    pmat = plotmats[s]
    plt.imshow(pmat,cmap="jet")#,vmin=pmat.min(),vmax=np.percentile(pmat.flatten(),50))
    # plt.colorbar()
    plt.xticks(xt_pos,xt,rotation=90,fontsize=ticsais);plt.yticks(yt_pos,yt,fontsize=ticsais)
    # plt.xlabel(r"$\delta_G$",fontsize=labelsais);plt.ylabel(r"$\delta_\sigma$",fontsize=labelsais)
    plt.scatter(ix,iy,color="w",edgecolors="black",s=120,label=f"optimal\n{xv}={xo:.2f}\n{yv}={yo:.2f}")
    # plt.legend()
    ax.invert_yaxis()
    ##################SHUF
    cosa = "shuf"
    x_vals,y_vals,plotmats,coors_o,vals_o,_ = [output_shuf[key] for key in output_keys]
    (ix,iy),(xo,yo,oval) = coors_o[s],vals_o[s]
    xt_pos = range(len(x_vals))[::down];xt = x_vals[::down];yt_pos = range(len(y_vals))[::down];yt = y_vals[::down]
    ax = plt.subplot2grid((3,4),(2,s))
    if s==0:
        plt.ylabel(cosa+"\n"+r"$\delta_\sigma$",fontsize=labelsais)
    pmat = plotmats[s]
    plt.imshow(pmat,cmap="jet")#,vmin=pmat.min(),vmax=np.percentile(pmat.flatten(),50))
    # plt.colorbar()
    plt.xticks(xt_pos,xt,rotation=90,fontsize=ticsais);plt.yticks(yt_pos,yt,fontsize=ticsais)
    plt.xlabel(r"$\delta_G$",fontsize=labelsais)#;plt.ylabel(r"$\delta_\sigma$",fontsize=labelsais)
    plt.scatter(ix,iy,color="w",edgecolors="black",s=120,label=f"optimal\n{xv}={xo:.2f}\n{yv}={yo:.2f}")
    # plt.legend()
    ax.invert_yaxis()
        
plt.tight_layout()
plt.savefig("figures/SF1.png",dpi=300)
plt.show()

#%% solo los observables

plt.figure(2,figsize=(10,14))
plt.clf()
for i,var in enumerate(["mean","sync","meta"]):
    #############################HOMO
    cosa ="homo"
    x_vals,y_vals,plotmats,coors_o,vals_o,_ = [output_homo[key] for key in output_keys]
    (ix,iy),(xo,yo,oval) = coors_o[s],vals_o[s]
    xt_pos = range(len(x_vals))[::down];xt = x_vals[::down];yt_pos = range(len(y_vals))[::down];yt = y_vals[::down]
    
    ax = plt.subplot2grid((3,3),(0,i))
    ax.set_title(var,fontsize=titlesais)
    if i==0:
        plt.ylabel(cosa+"\n"+r"$\delta_\sigma$",fontsize=labelsais)
    pmat = plotmats[s]
    plt.imshow(pmat,cmap="jet")#,vmin=pmat.min(),vmax=np.percentile(pmat.flatten(),50))
    plt.xticks(xt_pos,xt,rotation=90,fontsize=ticsais);plt.yticks(yt_pos,yt,fontsize=ticsais)
    plt.xticks(xt_pos,xt,rotation=90,fontsize=ticsais);plt.yticks(yt_pos,yt,fontsize=ticsais)
    # plt.xlabel(r"$\delta_G$",fontsize=labelsais);plt.ylabel(r"$\delta_\sigma$",fontsize=labelsais)
    for s,st in enumerate(states):
        plt.scatter(coors_o[s][0],coors_o[s][1],color=colors[s],edgecolors="white",s=120,label=f"optimal\n{xv}={xo:.2f}\n{yv}={yo:.2f}")

    # plt.legend()
    ax.invert_yaxis()
    ###########################MAP
    cosa = "map"
    x_vals,y_vals,plotmats,coors_o,vals_o,_ = [output_map[key] for key in output_keys]
    (ix,iy),(xo,yo,oval) = coors_o[s],vals_o[s]
    xt_pos = range(len(x_vals))[::down];xt = x_vals[::down];yt_pos = range(len(y_vals))[::down];yt = y_vals[::down]
    ax = plt.subplot2grid((3,3),(1,i))
    if i==0:
        plt.ylabel(cosa+"\n"+r"$\delta_\sigma$",fontsize=labelsais)
    pmat = plotmats[s]
    plt.imshow(pmat,cmap="jet")#,vmin=pmat.min(),vmax=np.percentile(pmat.flatten(),50))
    plt.xticks(xt_pos,xt,rotation=90,fontsize=ticsais);plt.yticks(yt_pos,yt,fontsize=ticsais)
    # plt.xlabel(r"$\delta_G$",fontsize=labelsais);plt.ylabel(r"$\delta_\sigma$",fontsize=labelsais)
    for s,st in enumerate(states):
        plt.scatter(coors_o[s][0],coors_o[s][1],color=colors[s],edgecolors="white",s=120,label=f"optimal\n{xv}={xo:.2f}\n{yv}={yo:.2f}")
    # plt.legend()
    ax.invert_yaxis()
    ##################SHUF
    cosa = "shuf"
    x_vals,y_vals,plotmats,coors_o,vals_o,_ = [output_shuf[key] for key in output_keys]
    (ix,iy),(xo,yo,oval) = coors_o[s],vals_o[s]
    xt_pos = range(len(x_vals))[::down];xt = x_vals[::down];yt_pos = range(len(y_vals))[::down];yt = y_vals[::down]
    ax = plt.subplot2grid((3,3),(2,i))
    if i==0:
        plt.ylabel(cosa+"\n"+r"$\delta_\sigma$",fontsize=labelsais)
    pmat = plotmats[s]
    plt.imshow(pmat,cmap="jet")#,vmin=pmat.min(),vmax=np.percentile(pmat.flatten(),50))
    plt.xticks(xt_pos,xt,rotation=90,fontsize=ticsais);plt.yticks(yt_pos,yt,fontsize=ticsais)
    plt.xlabel(r"$\delta_G$",fontsize=labelsais)#;plt.ylabel(r"$\delta_\sigma$",fontsize=labelsais)
    for s,st in enumerate(states):
        plt.scatter(coors_o[s][0],coors_o[s][1],color=colors[s],edgecolors="white",s=120,label=f"optimal\n{xv}={xo:.2f}\n{yv}={yo:.2f}")
    # plt.legend()
    ax.invert_yaxis()
        
plt.tight_layout()
plt.show()


#%%plot con los violines 

vh,vm,vs = [x["violins_o"] for x in (output_homo,output_map,output_shuf)]

ylims = {"homo":((-0.1,0.1),(-0.1,0.1)),
         "map":((-0.14,0.14),(-0.2,0.2)),
         "shuf":((-0.03,0.03),(-0.11,0.11)),}

plt.figure(3)
plt.clf()
for s,st in enumerate(states):
    
    for idd,cosa in enumerate((output_homo,output_map,output_shuf)):
        instance = ("homo","map","shuf")[idd]
        
        ax1= plt.subplot2grid((3,2),(idd,0))
        to_plot = [cosa["vals_o"][i][0] for i in range(4)]
        ax1.bar(range(4),to_plot,label=yv,color=colors,alpha=alfa)
        ax1.set_ylim(ylims[instance][0])
        ax1.set_ylabel(instance+"\n+ACh <----------> -ACh")
        ax1.plot(range(4),to_plot,"x-",color="black",linestyle="dashed")
        
        
        ax2 = ax1.twinx()
        to_plot = [cosa["vals_o"][i][1] for i in range(4)]
        ax2.bar(range(5,9),to_plot,label=xv,color=colors,alpha=alfa)
        ax2.set_xticks(list(range(4))+list(range(5,9)),states*2)
        ax2.set_ylabel("-NA <----------> +NA")
        ax2.set_ylim(ylims[instance][1])
        ax2.plot(range(5,9),to_plot,"x-",color="black",linestyle="dashed")
    df_o = pd.DataFrame.from_dict({"homo":vh[s],"map":vm[s],"shuf":vs[s]},orient="index").T ##para los nan
    ax = plt.subplot2grid((4,2),(s,1))
    ax.set_ylabel("euccorr "+st,fontsize=13.5)
    sns.violinplot(df_o)
    if s != 3:
        plt.xticks([])
plt.show()









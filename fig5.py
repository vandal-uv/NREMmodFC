#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:08:15 2024

@author: flehue
"""
import sys
sys.path.append("../integration_segregation/")
sys.path.append("../")
import numpy as np
import pickle
import pandas as pd
import HMA
import seaborn as sns
from plot_violins import violin_plot
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind as ttest, ks_2samp as ks, linregress as LR
from statsmodels.stats.multitest import multipletests as mtests
import matplotlib.lines as mlines
import utils

states = ("W","N1","N2","N3")
mods = ("homo","map","shuf","emp")
colors = ["crimson","orange","forestgreen","navy"]

mod_colors = ("tab:blue","tab:orange","tab:green")


# mod = "shuf"
# dic = {}
# for i in range(81):
#     with open(f"../data/run_50seeds_output_{mod}_16dic_rank{i}.pickle","rb") as f:
#         subdic = pickle.load(f)    
#     dic.update(subdic)
    
# # dic["metainfo"] = {st:50 for st in states}   
# with open(f"data/run_50seeds_output_{mod}_16dic.pickle","wb") as f:
#     pickle.dump(dic,f)

# halt


#%% cargamos los parametros optimos por estado 


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

#%% load data, empirico, simulado hetero y simulado homo 
with open('data/run_50seeds_output_homo_16dic.pickle', 'rb') as f:
    dic_homo = pickle.load(f)
with open('data/run_50seeds_output_map_16dic.pickle', 'rb') as f:
    dic_map = pickle.load(f)
with open('data/run_50seeds_output_shuf_16dic.pickle', 'rb') as f:
    dic_shuf = pickle.load(f)

with open('data/emp_15inds_output_16dic.pickle', 'rb') as f:
    dic_emp = pickle.load(f)

emp_matts = {st:np.loadtxt(f"../mean_mat_{st}_8dic24.txt") for st in states}

FCmap_in = np.load("../../maps/DIST_FC_LC.npy")
FCmap_se = np.load("../../maps/DIST_FC_BF.npy")
FCmap_in /= FCmap_in.max();FCmap_se /= FCmap_se.max()


Clus_num,Clus_size,H_all = HMA.Functional_HP(emp_matts["W"])
Hin,Hse = HMA.Balance(emp_matts["W"], Clus_num, Clus_size)
Hin_node,Hse_node = HMA.nodal_measures(emp_matts["W"], Clus_num, Clus_size)

#%% load mats and segregation shit


def load(dic,nseeds=50):  ## load integration and segregation for area in a df, from dictionary 
    matts = {st:np.zeros((nseeds,90,90)) for st in states}
    Hin_nodes = {st:np.zeros((nseeds,90)) for st in states}
    Hse_nodes = {st:np.zeros((nseeds,90)) for st in states}
    for key in dic.keys(): #unwrap dictionary 
        if key != "metainfo":
            s,state = key
            matts[state][s] = dic[key]["sFC"]
            dic[key] = dic[key]
            Hin_nodes[state][s] = dic[key]["Hin_node_sim"]
            Hse_nodes[state][s] = dic[key]["Hse_node_sim"]
    matts = {st:matts[st].mean(axis=0) for st in states}
    return matts,Hin_nodes,Hse_nodes


###estoy viendo puros 0 en el perfil en vals_se a veces

def load_dfs(dic_emp,dic_homo,dic_map,dic_shuf):
    _,Hin_nodes_emp,Hse_nodes_emp = load(dic_emp)
    
    _,Hin_nodes_homo,Hse_nodes_homo= load(dic_homo)
    _,Hin_nodes_map,Hse_nodes_map= load(dic_map)
    _,Hin_nodes_shuf,Hse_nodes_shuf= load(dic_shuf)
    #primero integration
    df_in = pd.DataFrame()
    df_se = pd.DataFrame()
    vals_in,sts_in,mods_in = [],[],[]
    vals_se,sts_se,mods_se = [],[],[]
    for st in states:
        vals_in+=list(Hin_nodes_homo[st].mean(axis=0))+list(Hin_nodes_map[st].mean(axis=0)) +list(Hin_nodes_shuf[st].mean(axis=0))\
                 +list(Hin_nodes_emp[st].mean(axis=0))
        sts_in+=list(360*[st])
        mods_in += list(90*["homo"]+90*["map"]+90*["shuf"]+90*["emp"])
        
        vals_se+=list(Hse_nodes_homo[st].mean(axis=0))+list(Hse_nodes_map[st].mean(axis=0)) +list(Hse_nodes_shuf[st].mean(axis=0))\
                 +list(Hse_nodes_emp[st].mean(axis=0))
        sts_se+=list(360*[st])
        mods_se += list(90*["homo"]+90*["map"]+90*["shuf"]+90*["emp"])
        
    df_in["val"] = vals_in;df_in["state"]=sts_in;df_in["mod"] = mods_in
    df_se["val"] = vals_se;df_se["state"]=sts_se;df_se["mod"] = mods_se
    return df_in,df_se
    
    
sim_matts_homo,Hin_nodes_homo,Hse_nodes_homo = load(dic_homo)
sim_matts_map,Hin_nodes_map,Hse_nodes_map = load(dic_map)
sim_matts_shuf,Hin_nodes_shuf,Hse_nodes_shuf = load(dic_shuf)
_,Hin_nodes_emp,Hse_nodes_emp = load(dic_emp,nseeds=15)





def diffs(dic): ##dictionary with regional differences of in se
    _,Hin_nodes,Hse_nodes = load(dic)
    difs_in = {};difs_se = {}
    difs_in["W_N1"] = Hin_nodes["N1"].mean(axis=0)-Hin_nodes["W"].mean(axis=0)
    difs_in["N1_N2"] = Hin_nodes["N2"].mean(axis=0)-Hin_nodes["N1"].mean(axis=0)
    difs_in["N1_N2"] = Hin_nodes["N3"].mean(axis=0)-Hin_nodes["N2"].mean(axis=0)
    
    difs_se["W_N1"] = Hse_nodes["N1"].mean(axis=0)-Hse_nodes["W"].mean(axis=0)
    difs_se["N1_N2"] = Hse_nodes["N2"].mean(axis=0)-Hse_nodes["N1"].mean(axis=0)
    difs_se["N2_N3"] = Hse_nodes["N3"].mean(axis=0)-Hse_nodes["N2"].mean(axis=0)
    return difs_in,difs_se

def corr_HMA(dic,method = "fdr_bh"):
    _,Hin_nodes_emp,Hse_nodes_emp = load(dic_emp)
    _,Hin_nodes,Hse_nodes = load(dic)
    ps_in,ps_se = [],[]
    corrs_in,corrs_se = [],[]
    for st in states:
        emp_in,sim_in = Hin_nodes_emp[st].mean(axis=0),Hin_nodes[st].mean(axis=0)
        emp_se,sim_se = Hse_nodes_emp[st].mean(axis=0),Hse_nodes[st].mean(axis=0)
        _,p_in = ttest(emp_in,sim_in)
        _,p_se = ttest(emp_se,sim_se)
        corrs_in.append(np.corrcoef(emp_in,sim_in)[0,1])
        corrs_se.append(np.corrcoef(emp_se,sim_se)[0,1])
        
        ps_in.append(p_in);ps_se.append(p_se)
    _,preout,_,_= mtests(ps_in+ps_se,method=method)
    ps_in = tuple(preout[:4]);ps_se = tuple(preout[4:])
    return ps_in,ps_se,corrs_in,corrs_se

##a plot helper function
def plot_1d_mean_std_with_line(datasets, colors, labels=None,alpha=0.7,inds = None,marker ="x"):
    """
    Plots each 1D dataset as mean ± standard deviation, with a line connecting the means.

    Parameters:
        datasets (list of np.ndarray): A list of 1D arrays or lists of data points.
        colors (list of str): A list of colors for each dataset.
        labels (list of str, optional): A list of labels for the legend. Defaults to None.
    """
    if len(datasets) != len(colors):
        raise ValueError("The number of datasets must match the number of colors.")
    if inds is None:
        inds = range(len(datasets))
    means = []
    stds = []
    
    for data in datasets:
        means.append(np.mean(data))
        stds.append(np.std(data))
    
    x = inds
    for i in range(len(datasets)):
        
        plt.errorbar(x[i], means[i], yerr=stds[i], fmt=marker, color=colors[i], capsize=5, zorder=3,alpha=alpha,markersize=10)
        
    
    if labels and len(labels) != len(datasets):
        raise ValueError("The number of labels must match the number of datasets.")
        plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    
        
#analyze the correlation of differences
difs_homo_in,difs_homo_se = diffs(dic_homo)
difs_map_in,difs_map_se = diffs(dic_map)
difs_shuf_in,difs_shuf_se = diffs(dic_shuf)
difs_emp_in,difs_emp_se = diffs(dic_emp)

##analyze the distribution of integration segregation
comp_df = pd.DataFrame(index=states)
ps_in,ps_se,corrs_in,corrs_se = corr_HMA(dic_homo)
for ll in ("corrs_in","corrs_se"):
    comp_df[ll+"_homo"] = eval(ll)
ps_in,ps_se,corrs_in,corrs_se = corr_HMA(dic_map)
for ll in ("corrs_in","corrs_se"):
    comp_df[ll+"_map"] = eval(ll)
ps_in,ps_se,corrs_in,corrs_se = corr_HMA(dic_shuf)
for ll in ("corrs_in","corrs_se"):
    comp_df[ll+"_shuf"] = eval(ll)
    
#%%correlation with empirical accross seeds
df_seeds = pd.DataFrame()
_,in_homo,se_homo = load(dic_homo)
_,in_map,se_map = load(dic_map)
_,in_shuf,se_shuf = load(dic_shuf)
df_in,df_se = load_dfs(dic_emp,dic_homo,dic_map,dic_shuf)

for s,st in enumerate(states):
    
    fun = lambda x : np.arctanh(x)
    
    x_in = df_in[(df_in["mod"]=="emp") & (df_in["state"]==st)]["val"].values
    #homo_in
    vals_in_homo = [fun(np.corrcoef(x_in,in_homo[st][seed])[0,1]) for seed in range(50)]
    #map_in
    vals_in_map = [fun(np.corrcoef(x_in,in_map[st][seed])[0,1]) for seed in range(50)]
    #shuf_in
    vals_in_shuf = [fun(np.corrcoef(x_in,in_shuf[st][seed])[0,1]) for seed in range(50)]    
    
    x_se = df_se[(df_se["mod"]=="emp") & (df_se["state"]==st)]["val"].values
    #homo_in
    vals_se_homo = [fun(np.corrcoef(x_se,se_homo[st][seed])[0,1]) for seed in range(50)]
    #map_in
    vals_se_map = [fun(np.corrcoef(x_se,se_map[st][seed])[0,1]) for seed in range(50)]
    print("nannn",np.argwhere(np.isnan(vals_se_map)))
    #shuf_in
    vals_se_shuf = [fun(np.corrcoef(x_se,se_shuf[st][seed])[0,1]) for seed in range(50)]    
    
    
    
    
    df_seeds[st+"_in_homo"] = vals_in_homo
    df_seeds[st+"_in_map"] = vals_in_map
    df_seeds[st+"_in_shuf"] = vals_in_shuf
    
    df_seeds[st+"_se_homo"] = vals_se_homo
    df_seeds[st+"_se_map"] = vals_se_map
    df_seeds[st+"_se_shuf"] = vals_se_shuf
#%% cohen's d

for s,st in enumerate(states):
    dist_homo,dist_map,dist_shuf = [df_seeds[f"{st}_in_{mod}"].values for mod in mods[:-1]]
    d1 = utils.cohen_d(dist_map,dist_homo)
    d2 = utils.cohen_d(dist_map,dist_shuf)
    d3 = utils.cohen_d(dist_homo,dist_shuf)
    print(st,"integration", f"{d1:.2f},{d2:.2f},,,,,,,,,{d3:.2f}")
    
    dist_homo,dist_map,dist_shuf = [df_seeds[f"{st}_se_{mod}"].values for mod in mods[:-1]]
    d1 = utils.cohen_d(dist_map,dist_homo)
    d2 = utils.cohen_d(dist_map,dist_shuf)
    d3 = utils.cohen_d(dist_homo,dist_shuf)
    print(st,"segregation", f"{d1:.2f},{d2:.2f},,,,,,,,,{d3:.2f}")
    
#%%violin plot of integration
#df of integration
alfa = 0.7
titlesais = 20
labelsais = 18
ticsais = 15
legendsais = 18



alfa = 0.6
plt.figure(1)
plt.clf()
plt.gcf().set_size_inches(12, 18)

###show integration component vs map in the awake state
ax = plt.subplot2grid((3,2),(0,0))
x= FCmap_in;y=Hin_node
a, b, r, p, ss = LR(x, y)
print(r,p)
ax.plot(x,a*x+b,alpha=0.8,color="tab:orange",label=r"$\rho =$"+f"{r:.4f}")
ax.scatter(x, y, s = 60, color = 'navy', alpha = 0.6)
ax.set_ylabel("(Wake) Empirical nodal Integration",fontsize=labelsais)
ax.set_yticks((0,0.1,0.2,0.3),(0,0.1,0.2,0.3),fontsize=ticsais)
ax.set_xlabel("Nodal FC strength with LC (normalized)",fontsize=labelsais)
ax.set_xticks((0,0.2,0.4,0.6,0.8,1),(0,0.2,0.4,0.6,0.8,1),fontsize=ticsais)
ax.legend(loc="upper left",fontsize=ticsais)
ax.spines[['right', 'top']].set_visible(False)

###show segregation component vs map in the awake state
ax = plt.subplot2grid((3,2),(0,1))
x= FCmap_se;y=Hse_node
a, b, r, p, ss = LR(x, y)
print(r,p)
ax.plot(x,a*x+b,alpha=0.8,color="tab:orange",label=r"$\rho =$"+f"{r:.4f}")
ax.scatter(x, y, s = 60, color = 'navy', alpha = 0.6)
ax.set_ylabel("(Wake) Empirical nodal Segregation",fontsize=labelsais)
ax.set_yticks((0,0.1,0.2,0.3,0.4),(0,0.1,0.2,0.3,0.4),fontsize=ticsais)
ax.set_xlabel("Nodal FC strength with BF (normalized)",fontsize=labelsais)
ax.set_xticks((0,0.2,0.4,0.6,0.8,1),(0,0.2,0.4,0.6,0.8,1),fontsize=ticsais)
ax.legend(loc="upper left",fontsize=ticsais)
ax.spines[['right', 'top']].set_visible(False)
# plt.tight_layout(pad=4)


###empirical integration
ax = plt.subplot2grid((3,2),(1,0))
violin_plot(ax, [df_in[(df_in["state"]==st) & (df_in["mod"]=="emp")]["val"].values for st in states],
            color_names= colors,
            alpha_violin = 0.5)
ax.set_ylabel("Empirical nodal Integration",fontsize=labelsais)
ax.set_yticks((0,0.05,0.1,0.15,0.2),(0,0.05,0.1,0.15,0.2),fontsize=ticsais)
ax.set_xticks(range(4),states,fontsize=labelsais)
ax.spines[['right', 'top']].set_visible(False)

##empirical segregation
ax = plt.subplot2grid((3,2),(1,1))
# ax.set_title("empirical segregation component\n across areas")
violin_plot(ax, [df_se[(df_se["state"]==st) & (df_se["mod"]=="emp")]["val"].values for st in states],
            color_names= colors,
            alpha_violin = 0.5)
ax.set_ylabel("Empirical nodal Segregation",fontsize=labelsais)
ax.set_yticks((0,0.05,0.1,0.15,0.2),(0,0.05,0.1,0.15,0.2),fontsize=ticsais)
ax.set_xticks(range(4),states,fontsize=labelsais)
ax.spines[['right', 'top']].set_visible(False)


########################IDEA 1, 4 grupos para cada estado
alfa=0.8;markers = ["o","^","s"]
#Integration
Inds = []
ax = plt.subplot2grid((3,2),(2,0))
inds1 = 0+5*np.arange(0,4,1)
datasets = [df_seeds[f"{st}_in_homo"].values for st in states]
plot_1d_mean_std_with_line(datasets, colors=colors,alpha=alfa,inds=inds1,marker =markers[0])

inds2 = 1+5*np.arange(0,4,1)
datasets = [df_seeds[f"{st}_in_map"].values for st in states]
plot_1d_mean_std_with_line(datasets, colors=colors,alpha=alfa,inds=inds2,marker =markers[1])

inds3 = 2+5*np.arange(0,4,1)
datasets = [df_seeds[f"{st}_in_shuf"].values for st in states]
plot_1d_mean_std_with_line(datasets, colors=colors,alpha=alfa,inds=inds3,marker =markers[2])

Inds = np.concatenate((inds1,inds2,inds3))
ax.set_ylabel("Correlation with Empirical Integration",fontsize=labelsais)
ax.set_ylabel("Similarity to Empirical Integration",fontsize=labelsais)
ax.set_yticks((-0.2,0,0.2,0.4,0.6),(-0.2,0,0.2,0.4,0.6),fontsize=ticsais)
ax.set_xticks((1,6,11,16),states,fontsize=labelsais)
ax.tick_params(axis=u'x', which=u'both',length=0)
ax.spines[['right', 'top']].set_visible(False)

mline = [mlines.Line2D([], [], color='black', marker=markers[i], ls='', label=mods[i],alpha=0.7) for i in range(3)]
lgnd= ax.legend(handles=mline,fontsize=legendsais,markerscale=1.6)
# for legend_handle in lgnd.legendHandles:
#     legend_handle._legmarker.set_markersize(9)
# sec = ax.secondary_xaxis(location=0)
# sec.set_xticks((1,5,9,13), labels=[f"\n\n{st}" for st in states],fontsize=ticsais)
# ax.spines[['right', 'top']].set_visible(False)

#Segregation
Inds = []
ax = plt.subplot2grid((3,2),(2,1))
inds1 = 0+5*np.arange(0,4,1)
datasets = [df_seeds[f"{st}_se_homo"].values for st in states]
plot_1d_mean_std_with_line(datasets, colors=colors,alpha=alfa,inds=inds1,marker =markers[0])

inds2 = 1+5*np.arange(0,4,1)
datasets = [df_seeds[f"{st}_se_map"].values for st in states]
plot_1d_mean_std_with_line(datasets, colors=colors,alpha=alfa,inds=inds2,marker =markers[1])

inds3 = 2+5*np.arange(0,4,1)
datasets = [df_seeds[f"{st}_se_shuf"].values for st in states]
plot_1d_mean_std_with_line(datasets, colors=colors,alpha=alfa,inds=inds3,marker =markers[2])

Inds = np.concatenate((inds1,inds2,inds3))
ax.set_ylabel("Correlation with Empirical Segregation",fontsize=labelsais)
ax.set_ylabel("Similarity to Empirical Segregation",fontsize=labelsais)
ax.set_yticks((0,0.2,0.4,0.6),(0,0.2,0.4,0.6),fontsize=ticsais)
ax.set_xticks((1,6,11,16),states,fontsize=labelsais)
ax.tick_params(axis=u'x', which=u'both',length=0)
ax.spines[['right', 'top']].set_visible(False)

# mline = [mlines.Line2D([], [], color='black', marker=markers[i], ls='', label=mods[i]) for i in range(3)]
ax.legend(handles=mline,fontsize=legendsais,markerscale=1.6)

# sec = ax.secondary_xaxis(location=0)
# sec.set_xticks((1,5,9,13), labels=[f"\n\n{st}" for st in states],fontsize=ticsais)
# ax.spines[['right', 'top']].set_visible(False)


#####################IDEA 2, 3 grupos para modalidad
#Integration
# Inds = []
# ax = plt.subplot2grid((3,2),(2,0))
# inds1 = np.arange(0,4,1)
# datasets = [df_seeds[f"{st}_in_homo"].values for st in states]
# plot_1d_mean_std_with_line(datasets, colors=colors,alpha=alfa,inds=inds1)

# inds2 = 5+np.arange(0,4,1)
# datasets = [df_seeds[f"{st}_in_map"].values for st in states]
# plot_1d_mean_std_with_line(datasets, colors=colors,alpha=alfa,inds=inds2)

# inds3 = 10+np.arange(0,4,1)
# datasets = [df_seeds[f"{st}_in_shuf"].values for st in states]
# plot_1d_mean_std_with_line(datasets, colors=colors,alpha=alfa,inds=inds3)

# Inds = np.concatenate((inds1,inds2,inds3))
# ax.set_ylabel("Correlation of Empirical\n& Simulated Integration",fontsize=labelsais)
# ax.set_yticks((-0.2,0,0.2,0.4,0.6),(-0.2,0,0.2,0.4,0.6),fontsize=ticsais)
# ax.set_xticks(Inds,3*states,fontsize=labelsais-4)
# # ax.legend()
# sec = ax.secondary_xaxis(location=0)
# sec.set_xticks((inds1.mean(),inds2.mean(),inds3.mean()), labels=[f"\n{mod}" for mod in mods[:-1]],fontsize=ticsais)
# ax.spines[['right', 'top']].set_visible(False)

# ##Segregation
# Inds = []
# ax = plt.subplot2grid((3,2),(2,1))
# inds1 = 0+ np.arange(0,4,1)
# datasets = [df_seeds[f"{st}_se_homo"].values for st in states]
# plot_1d_mean_std_with_line(datasets, colors=colors,alpha=alfa,inds=inds1)

# inds2 = 5+np.arange(0,4,1)
# datasets = [df_seeds[f"{st}_se_map"].values for st in states]
# plot_1d_mean_std_with_line(datasets, colors=colors,alpha=alfa,inds=inds2)

# inds3 = 10+np.arange(0,4,1)
# datasets = [df_seeds[f"{st}_se_shuf"].values for st in states]
# plot_1d_mean_std_with_line(datasets, colors=colors,alpha=alfa,inds=inds3)

# Inds = np.concatenate((inds1,inds2,inds3))
# ax.set_ylabel("Correlation of Empirical\n& Simulated Segregation",fontsize=labelsais)
# ax.set_yticks((0,0.2,0.4,0.6),(0,0.2,0.4,0.6),fontsize=ticsais)
# ax.set_xticks(Inds,3*states,fontsize=labelsais-4)
# sec = ax.secondary_xaxis(location=0)
# sec.set_xticks((inds1.mean(),inds2.mean(),inds3.mean()), labels=[f"\n{mod}" for mod in mods[:-1]],fontsize=ticsais)
# ax.spines[['right', 'top']].set_visible(False)
# ax.legend()


plt.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.05)
# plt.savefig("fig5.svg",dpi=300)

#%%


#%%violin plots of integration and segregation
alfa = 0.2
df_in,df_se = load_dfs(dic_emp,dic_homo,dic_map,dic_shuf)





alfa = 0.6
plt.figure(2)
plt.clf()

###empirical integration
ax = plt.subplot2grid((5,2),(0,0))
ax.set_title("empirical integration component\n across areas")
violin_plot(ax, [df_in[(df_in["state"]==st) & (df_in["mod"]=="emp")]["val"].values for st in states],
            color_names= colors,
            alpha_violin = 0.5)
##empirical segregation
ax = plt.subplot2grid((5,2),(0,1))
ax.set_title("empirical segregation component\n across areas")
violin_plot(ax, [df_se[(df_se["state"]==st) & (df_se["mod"]=="emp")]["val"].values for st in states],
            color_names= colors,
            alpha_violin = 0.5)


for s,st in enumerate(states):
    print(st)
    
    print("integration")
    x_in = df_in[(df_in["mod"]=="emp") & (df_in["state"]==st)]["val"].values #el empirico pal estado    
    ax = plt.subplot2grid((5,2),(1+s,0))
    ax.set_title(st)
    for m,mod in enumerate(mods[:-1]):
        y_in = df_in[(df_in["mod"]==mod) & (df_in["state"]==st)]["val"].values
        
        kss = ks(x_in,y_in)
        corr = np.corrcoef(x_in,y_in)[0,1]
        e = np.linalg.norm(x_in-y_in)
        
        a, b, r, p, ss = LR(x_in, y_in)
        
        print(mod,f"ks={kss[0]:.4f},corr={corr:.4f},r={r:.4f},p={p:.4f},e={e:f}")
        
        if mod=="map":
            plt.plot(x_in,a*x_in+b,color="tab:orange",label=f"{mod}, r={r:.4f}")
            plt.scatter(x_in, y_in, s = 60, color = 'navy', alpha = 0.5)
            # plt.scatter(x,y,label=f"{mod}, r={r:.4f}",color=mod_colors[m])
        elif mod =="homo":
            plt.plot(x_in,a*x_in+b,alpha=0.8,color="grey",label=f"{mod}, r={r:.4f}")
            plt.scatter(x_in, y_in, s = 60, color = 'grey', alpha = 0.2)
        else:
            plt.plot(x_in,a*x_in+b,alpha=0.8,color="grey",linestyle="dashed",label=f"{mod}, r={r:.4f}")
            plt.scatter(x_in, y_in, s = 60, color = 'grey', alpha = 0.2)
    ax.legend(loc="lower right")
    
    ##segregation
    print("segregation")
    x_se = df_se[(df_se["mod"]=="emp") & (df_se["state"]==st)]["val"].values #el empirico pal estado    
    ax = plt.subplot2grid((5,2),(1+s,1))
    ax.set_title(st)
    for m,mod in enumerate(mods[:-1]):
        y_se = df_se[(df_se["mod"]==mod) & (df_se["state"]==st)]["val"].values
        
        kss = ks(x_se,y_se)
        corr = np.corrcoef(x_se,y_se)[0,1]
        e = np.linalg.norm(x_se-y_se)
        
        a, b, r, p, ss = LR(x_se, y_se)
        
        print(mod,f"ks={kss[0]:.4f},corr={corr:.4f},r={r:.4f},p={p:.4f},e={e:f}")
        
        if mod=="map":
            plt.plot(x_se,a*x_se+b,color="tab:orange",label=f"{mod}, r={r:.4f}")
            plt.scatter(x_se, y_se, s = 60, color = 'navy', alpha = 0.5)
            # plt.scatter(x,y,label=f"{mod}, r={r:.4f}",color=mod_colors[m])
        elif mod =="homo":
            plt.plot(x_se,a*x_se+b,alpha=0.8,color="grey",label=f"{mod}, r={r:.4f}")
            plt.scatter(x_se, y_se, s = 60, color = 'grey', alpha = 0.2)
        else:
            plt.plot(x_se,a*x_se+b,alpha=0.8,color="grey",linestyle="dashed",label=f"{mod}, r={r:.4f}")
            plt.scatter(x_se, y_se, s = 60, color = 'grey', alpha = 0.2)
    ax.legend(loc="lower right")
    
plt.tight_layout()
plt.show()



        
        
#%% display  FC matrices
vmin,vmax = 0,1


plt.figure(3)
plt.clf()
# plt.suptitle("homo,hetero,empirical")
for s,st in enumerate(states):
    plt.subplot(4,4,4*s+1)
    if st=="W":
        plt.title("sim homo")
    plt.imshow(sim_matts_homo[st],vmin=vmin,vmax=vmax,cmap="jet")
    plt.subplot(4,4,4*s+2)
    if st=="W":
        plt.title("sim map")
    plt.imshow(sim_matts_map[st],vmin=vmin,vmax=vmax,cmap="jet")
    plt.subplot(4,4,4*s+3)
    if st=="W":
        plt.title("sim shuf")
    plt.imshow(sim_matts_shuf[st],vmin=vmin,vmax=vmax,cmap="jet")
    plt.subplot(4,4,4*s+4)
    if st=="W":
        plt.title("emp")
    plt.imshow(emp_matts[st],vmin=vmin,vmax=vmax,cmap="jet")
    plt.colorbar()
plt.tight_layout()
plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
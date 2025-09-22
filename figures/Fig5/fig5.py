#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:08:15 2024

@author: flehue
"""
import sys
sys.path.append("../integration_segregation/")
sys.path.append("../../")
import numpy as np
import pickle
import pandas as pd
from scipy import signal
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
with open('../../output/run_50seeds_output_homo_16dic.pickle', 'rb') as f:
    dic_homo = pickle.load(f)
with open('../../output/run_50seeds_output_map_16dic.pickle', 'rb') as f:
    dic_map = pickle.load(f)
with open('../../output/run_50seeds_output_shuf_16dic.pickle', 'rb') as f:
    dic_shuf = pickle.load(f)

with open('../../output/emp_15inds_output_16dic.pickle', 'rb') as f:
    dic_emp = pickle.load(f)

emp_matts = {st:np.loadtxt(f"../../empirical/mean_mat_{st}_8dic24.txt") for st in states}
Clus_num,Clus_size,H_all = HMA.Functional_HP(emp_matts["W"])
Hin,Hse = HMA.Balance(emp_matts["W"], Clus_num, Clus_size)
Hin_node,Hse_node = HMA.nodal_measures(emp_matts["W"], Clus_num, Clus_size)

# halt
#%%
TR = 2.08 

a0, b0 = signal.bessel(3, 2 * TR * np.array([0.01, 0.1]), btype = 'bandpass')

with open("../../data/all_individuals_IDs.pickle","rb") as f:
    entries = pickle.load(f)["included"]

FCmap_in = np.zeros(90)
FCmap_se = np.zeros(90)
for e,entry in enumerate(entries):
    print(e)
    signals = np.load(f"../../empirical/BOLD_empirical_AAL90/BOLD_complete_nonfilt_S{entry}.npy")
    BOLD_filt = signal.filtfilt(a0, b0, signals, axis = 0)[:,0:90]
    
    LC = np.load(f"../../empirical/BOLD_empirical_AAL90/LC_bilateral_norm_nonfilt_S{entry}.npy")
    LC_filt = signal.filtfilt(a0, b0, LC, axis = 0).flatten()
    
    BF = np.load(f"../../empirical/BOLD_empirical_AAL90/BF_bilateral_norm_nonfilt_S{entry}.npy")
    BF_filt = signal.filtfilt(a0, b0, BF, axis = 0).flatten()
    
    
    vector = np.load(f"../../data/BOLD_empirical_AAL90/align_stages_S{entry}.npy")
    idx_WAKE = vector == 0

    #Extracting the time series for each sleep stage, then compute the FC matrices
    BOLD_WAKE = BOLD_filt[idx_WAKE,:]
    LC_WAKE = LC_filt[idx_WAKE]
    BF_WAKE = BF_filt[idx_WAKE]
    
    FCmap_in += np.array([np.corrcoef(BOLD_WAKE[:,i],LC_WAKE)[0,1] for i in range(90)])
    FCmap_se += np.array([np.corrcoef(BOLD_WAKE[:,i],BF_WAKE)[0,1] for i in range(90)])
    
    
    
FCmap_in /= len(entries)
FCmap_in = FCmap_in[ list(range(0,90,2)) + list(range(1,90,2))[::-1]     ]

FCmap_se /= len(entries)
FCmap_se = FCmap_se[ list(range(0,90,2)) + list(range(1,90,2))[::-1]     ]

# FCmap_in = np.load("../../maps/DIST_FC_LC.npy")
# FCmap_se = np.load("../../maps/DIST_FC_BF.npy")
FCmap_in /= FCmap_in.max();FCmap_se /= FCmap_se.max()



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
ticsais = 14
legendsais = 18
pad_title = -14



alfa = 0.6
plt.figure(1)
plt.clf()
plt.gcf().set_size_inches(12, 18)


###empirical integration
ax = plt.subplot2grid((3,2),(0,0))
ax.set_title("Empirical nodal Integration",fontsize=titlesais)
violin_plot(ax, [df_in[(df_in["state"]==st) & (df_in["mod"]=="emp")]["val"].values for st in states],
            color_names= colors,
            alpha_violin = 0.5)
ax.set_ylabel("Integration component",fontsize=labelsais)
ax.set_yticks((0,0.05,0.1,0.15),(0,0.05,0.1,0.15),fontsize=ticsais)
ax.set_xticks(range(4),states,fontsize=labelsais)
ax.spines[['right', 'top']].set_visible(False)

##empirical segregation
ax = plt.subplot2grid((3,2),(0,1))
ax.set_title("Empirical nodal Segregation",fontsize=titlesais)
violin_plot(ax, [df_se[(df_se["state"]==st) & (df_se["mod"]=="emp")]["val"].values for st in states],
            color_names= colors,
            alpha_violin = 0.5)
ax.set_ylabel("Segregation component",fontsize=labelsais)
ax.set_yticks((0,0.05,0.1,0.15),(0,0.05,0.1,0.15),fontsize=ticsais)
ax.set_xticks(range(4),states,fontsize=labelsais)
ax.spines[['right', 'top']].set_visible(False)

#################show integration component vs map in the awake state
ax = plt.subplot2grid((3,2),(1,0))
ax.set_title("LC-FC vs Integration (Wake)",y = 0.98,fontsize=titlesais)
x= FCmap_in;y=Hin_node
a, b, r, p, ss = LR(x, y)
print(r,p)
ax.plot(x,a*x+b,alpha=0.8,color="tab:orange",label=r"$\rho =$"+f"{r:.4f}")
ax.scatter(x, y, s = 60, color = 'navy', alpha = 0.6)
ax.set_ylabel("Integration Component",fontsize=labelsais)
ax.set_yticks((0,0.1,0.2,0.3),(0,0.1,0.2,0.3),fontsize=ticsais)
ax.set_xlabel("Nodal FC strength with LC (normalized)",fontsize=labelsais)
ax.set_xticks((0,0.2,0.4,0.6,0.8,1),(0,0.2,0.4,0.6,0.8,1),fontsize=ticsais)
ax.legend(loc=(0.04,0.78),fontsize=ticsais)
ax.spines[['right', 'top']].set_visible(False)
###inset axis 
subax = ax.inset_axes([0.5,0.03,0.5,0.25])
subax.set_title("          LC-FC vs segregation",fontsize=titlesais-7)
x = FCmap_in;y=Hse_node
a, b, r, p, ss = LR(x, y)
print(r,p)
subax.plot(x,a*x+b,alpha=0.8,color="tab:orange",label=r"$\rho =$"+f"{r:.4f}")
subax.scatter(x, y, s = 40, color = 'navy', alpha = 0.4)
my_pal = {"W": "crimson","N1": "orange","N2": "forestgreen","N3": "navy"}
subax.set_xticks([],[])
subax.set_yticks([],[])
subax.patch.set_alpha(0.5)
ax.set_aspect(2.5)



###show segregation component vs map in the awake state
ax = plt.subplot2grid((3,2),(1,1))
ax.set_title("BF-FC vs segregation (Wake)",y = 0.98,fontsize=titlesais)
x= FCmap_se;y=Hse_node
a, b, r, p, ss = LR(x, y)
print(r,p)
ax.plot(x,a*x+b,alpha=0.8,color="tab:orange",label=r"$\rho =$"+f"{r:.4f}")
ax.scatter(x, y, s = 60, color = 'navy', alpha = 0.6)
ax.set_ylabel("Segregation Component",fontsize=labelsais)
ax.set_yticks((0,0.1,0.2,0.3),(0,0.1,0.2,0.3),fontsize=ticsais)
ax.set_ylim((0,.35))
ax.set_xlabel("Nodal FC strength with BF (normalized)",fontsize=labelsais)
ax.set_xticks((0,0.2,0.4,0.6,0.8,1),(0,0.2,0.4,0.6,0.8,1),fontsize=ticsais)
ax.legend(loc=(0.04,0.78),fontsize=ticsais)
ax.spines[['right', 'top']].set_visible(False)
###inset axis 
subax = ax.inset_axes([0.5,0.03,0.5,0.25])
subax.set_title("          BF-FC vs integration",fontsize=titlesais-7)
x = FCmap_se;y=Hin_node
a, b, r, p, ss = LR(x, y)
print(r,p)
subax.plot(x,a*x+b,alpha=0.8,color="tab:orange",label=r"$\rho =$"+f"{r:.4f}")
subax.scatter(x, y, s = 40, color = 'navy', alpha = 0.4)
my_pal = {"W": "crimson","N1": "orange","N2": "forestgreen","N3": "navy"}
subax.set_xticks([],[])
subax.set_yticks([],[])
subax.patch.set_alpha(0.5)
ax.set_aspect(2.5)


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

plt.tight_layout()
# plt.gcf().subplots_adjust(bottom=)
plt.savefig("fig5_recalculated_correlations.svg",dpi=300)
        
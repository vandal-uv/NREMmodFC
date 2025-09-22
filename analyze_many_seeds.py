# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 22:46:34 2024

@author: flehu
"""
import sys
sys.path.append("../integration_segregation/")
import numpy as np
import pickle
import pandas as pd
import HMA
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind as ttest, ks_2samp as ks, linregress as LR
from statsmodels.stats.multitest import multipletests as mtests


states = ("W","N1","N2","N3")
mods = ("homo","map","shuf","emp")
# mod_colors = ("tab:blue","tab:orange","tab:green")
# mod = "map"
# with open(f"../data/rerun_50seeds_output_{mod}_19nov_rank0.pickle","rb") as f:
#     dic = pickle.load(f)
# for i in range(1,81):
#     with open(f"../data/rerun_50seeds_output_{mod}_19nov_rank{i}.pickle","rb") as f:
#         subdic = pickle.load(f)    
#     dic.update(subdic)
    
# dic["metainfo"] = {st:50 for st in states}   
# with open(f"data/run_50seeds_output_{mod}_nov.pickle","wb") as f:
#     pickle.dump(dic,f)

# halt


#%% cargamos los parametros optimos por estado 


optimals_homo = {"W":  (0.14,0.0,7.7,0.0),
                 "N1": (0.14,0.03,7.7,0.0),
                 "N2": (0.14,0.03,7.7,-0.03), 
                 "N3":  (0.14,-0.03,7.7,-0.03)}

optimals_map = {"W":  (0.14,-0.01,7.7,0.07),
                  "N1": (0.14,0.09,7.7,-0.01),
                  "N2": (0.14,0.06,7.7,-0.07), 
                  "N3":  (0.14,0.01,7.7,-0.13)}

optimals_shuf = {"W":  (0.14,0.0,7.7,0.0),
                 "N1": (0.14,0.02,7.7,0.06),
                 "N2": (0.14,0.02,7.7,-0.02), 
                 "N3":  (0.14,-0.01,7.7,-0.1)}


#%% load data, empirico, simulado hetero y simulado homo 
with open('data/run_50seeds_output_homo_nov.pickle', 'rb') as f:
    dic_homo = pickle.load(f)
with open('data/run_50seeds_output_map_nov.pickle', 'rb') as f:
    dic_map = pickle.load(f)
with open('data/run_50seeds_output_shuf_nov.pickle', 'rb') as f:
    dic_shuf = pickle.load(f)

with open('data/output_EMPIRICAL_20agosto24.pickle', 'rb') as f:
    dic_emp = pickle.load(f)
#%% load mats and segregation shit


def load(dic):  ## load integration and segregation for area in a df, from dictionary 
    matts = {st:np.zeros((dic["metainfo"][st],90,90)) for st in states}
    Hin_nodes = {st:np.zeros((dic["metainfo"][st],90)) for st in states}
    Hse_nodes = {st:np.zeros((dic["metainfo"][st],90)) for st in states}
    for key in dic.keys(): #unwrap dictionary 
        if key != "metainfo":
            s,state = key
            matts[state][s] = dic[key]["sFC"]
            dic[key] = dic[key]
            Hin_nodes[state][s] = dic[key]["Hin_node_sim"]
            Hse_nodes[state][s] = dic[key]["Hse_node_sim"]
    matts = {st:matts[st].mean(axis=0) for st in states}
    return matts,Hin_nodes,Hse_nodes

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
_,Hin_nodes_emp,Hse_nodes_emp = load(dic_emp)

emp_matts = {st:np.loadtxt(f"../../../analyze_empirical/mean_arctanhrho_filtered_{st}.txt") for st in states}


#%%diferencias de integracion y segregacion entre estados 

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


#%% display  FC matrices
vmin,vmax = 0,1


plt.figure(1)
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

#%%violin plot of integration
#df of integration
alfa = 0.2
df_in,df_se = load_dfs(dic_emp,dic_homo,dic_map,dic_shuf)





alfa = 0.6
plt.figure(2)
plt.clf()
plt.subplot2grid((3,2),(0,0))
plt.title("integration")
sns.violinplot(data=df_in,x="state",y="val",hue="mod",alpha=alfa)
plt.subplot2grid((3,2),(0,1))
plt.title("segregation")
sns.violinplot(data=df_se,x="state",y="val",hue="mod",alpha=alfa)

for s,st in enumerate(states):
    print(st)
    x = df_in[(df_in["mod"]=="emp") & (df_in["state"]==st)]["val"].values #el empirico pal estado
    plt.subplot(3,2,3+s)
    plt.title(st)
    
    
    for m,mod in enumerate(mods[:-1]):
        y = df_in[(df_in["mod"]==mod) & (df_in["state"]==st)]["val"].values
        
        kss = ks(x,y)
        corr = np.corrcoef(x,y)[0,1]
        e = np.linalg.norm(x-y)
        
        a, b, r, p, s = LR(x, y)
        
        print(mod,f"ks={kss[0]:.4f},corr={corr:.4f},r={r:.4f},p={p:.4f},e={e:f}")
        
        
        
        
        
        if mod=="map":
            plt.plot(x,a*x+b,color="tab:orange",label=f"{mod}, r={r:.4f}")
            plt.scatter(x, y, s = 60, color = 'navy', alpha = 0.5)
            # plt.scatter(x,y,label=f"{mod}, r={r:.4f}",color=mod_colors[m])
        elif mod =="homo":
            plt.plot(x,a*x+b,alpha=0.8,color="grey",label=f"{mod}, r={r:.4f}")
            plt.scatter(x, y, s = 60, color = 'grey', alpha = 0.2)
        else:
            plt.plot(x,a*x+b,alpha=0.8,color="grey",linestyle="dashed",label=f"{mod}, r={r:.4f}")
            plt.scatter(x, y, s = 60, color = 'grey', alpha = 0.2)
            # plt.scatter(x,y,label=f"{mod}, r={r:.4f}",color=mod_colors[m],alpha=alfa)
        
    plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

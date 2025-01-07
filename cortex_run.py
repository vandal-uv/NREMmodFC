# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 17:26:29 2022

@author: flehu
"""
import sys
sys.path.append("integration_segregation/",)
import numpy as np
import netwWilsonCowanPlastic as wc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import BOLDModel as BD
from scipy import signal 
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from time import time as tm
import utils
import pickle
import HMA 

original_struct = np.loadtxt("SC_opti_25julio.txt")
struct = utils.sub_weight(original_struct, 1)

plt.figure(101)
plt.imshow(struct,cmap="jet")
plt.show()
AALlabels = pd.read_csv("sorted_AAL_labels.txt")
RSN = np.loadtxt("RSN_AAL_Enzo.txt")
RSN_labels = ["Vis","ES","Aud","SM","DM","EC"]
states = ("W","N1","N2","N3")

# optimal = {"W":(0.4,0.167,6.6),"N1":(0.3,0.214,6.875),"N2":(0.2,0.191,7)}
# optimal_homo = {"W":(0,0.135,7.775),"N1":(0,0.175,7.7),"N2":(0,0.167,7.7),"N3":(0,0.429,5.075)}
state = "W"
modality = "map"

##formato (G,delta_G,sigmaE,delta_sigmaE)
optimals_homo = {"W":  (0.14,0.0,7.7,0.0),
                 "N1": (0.14,0.03,7.7,0.0),
                 "N2": (0.14,0.03,7.7,-0.03), 
                 "N3":  (0.14,-0.03,7.7,-0.03)}

optimals_map = {"W":  (0.14,-0.01,7.7,0.19),
                 "N1": (0.14,0.09,7.7,-0.01),
                 "N2": (0.14,0.06,7.7,-0.07), 
                 "N3":  (0.14,0.01,7.7,-0.13)}

optimals_shuf = {"W":  (0.14,0.0,7.7,0.0),
                 "N1": (0.14,0.02,7.7,0.06),
                 "N2": (0.14,0.02,7.7,-0.02), 
                 "N3":  (0.14,-0.01,7.7,-0.1)}

modality = "shuf"

if modality == "homo":
    ach_dist = np.ones(90)
    na_dist = np.ones(90)
elif modality =="map":
    ach_dist = np.load("maps/DIST_VAChT_feobv_hc18_aghourian.npy")
    na_dist = np.load("maps/DIST_LC_proj.npy")

elif modality == "shuf":
    ach_dist = np.load("maps/SHUFFLED_DIST_VAChT_feobv_hc18_aghourian.npy")
    na_dist = np.load("maps/SHUFFLED_DIST_LC_proj.npy")

G_val,deltaG,sigmaE,deltasigmaE = eval(f"optimals_{modality}['{state}']")

local_G = G_val + ach_dist*deltaG
local_sigmaE = sigmaE + na_dist*deltasigmaE

wc.sigmaE = local_sigmaE 
wc.G = local_G

# halt
#%%


wc.CM = struct
# wc.G = G_val #0.485 minimo obtenido el 2 de mayo
rhoE_val = 0.14
wc.rhoE = rhoE_val
wc.P = 0.4###input a las excitatorias





#%%
wc.tTrans1=1  # simulation to remove transient (runs twice)
wc.tTrans2=100
wc.timeTrans1=np.arange(0,wc.tTrans1,wc.dtSim) # time vector f or transient Sim
wc.timeTrans2=np.arange(0,wc.tTrans2,wc.dtSim)
tstop = 600
wc.tstop = tstop
wc.timeSim=np.arange(0,tstop,wc.dtSim) ##shorter
wc.time=np.arange(0,tstop,wc.dt)

nnodes = 90
wc.nnodes = nnodes
now = tm()
wc.wilsonCowan.recompile()
print("running!")
tray = wc.run(verbose=True)
print(f"time = {tm()-now:.3f} s ")
#%% aquí generamos la simulación BOLD y le aplicamos un lag

E_t = tray[:,0,:]


#en la simulacion hay 500 puntos por segundo, entonces si corro 250 puntos sería medio segundos
BOLD_downsamp = 10
BOLD = wc.simBOLD(E_t,nnodes=90,BOLD_downsamp=BOLD_downsamp) ###cuidado con el downsample del BOLD, el que usaba antes es 1000
BOLDtime = np.arange(0,tstop,wc.dt*BOLD_downsamp)[int(2000/BOLD_downsamp):]
sFC = np.corrcoef(BOLD.T)
# BOLD = shift_sub(BOLD,lag)
##hemodynamic
#%%metricas de comparacion
sim_mean = sFC.mean()

empFCs = {st:np.loadtxt(f"../analyze_empirical/mean_arctanhrho_filtered_{st}.txt") for st in states}


empFC = eval(f"empFCs[{'state'}]")
flat_empFC = np.concatenate([empFC[i,i+1:] for i in range(90)])

#corr,euc,ssim,new_metric
metrics = {st:utils.get_all_metrics(sFC,empFCs[st],data_range=1) for st in states}
C1 = 0.3; w = 1
euccorrs = {st:(metrics[st][1]/(C1+w*metrics[st][0])) for st in states}



BOLD_sync,BOLD_meta = utils.kuramoto(BOLD)
EEG_sync,EEG_meta = utils.kuramoto(E_t)

print(f"sFC mean = {sim_mean:.4}, empFC mean = {np.mean(empFC):.4f}")

print("ssim for all stages:", [metrics[st][2] for st in states])
print("euccorr for all stages:", [euccorrs[st] for st in states])
print(f"simBOLD sync = {BOLD_sync:.4f}")
print(f"simBOLD meta = {BOLD_meta:.4f}")
print(f"simEEG sync = {EEG_sync:.4f}")
print(f"simEEG meta = {EEG_meta:.4f}")


#%% integracion segregacion 

#simulada
Clus_num_sim,Clus_size_sim,H_all_sim = HMA.Functional_HP(sFC)
Hin_sim,Hse_sim = HMA.Balance(sFC, Clus_num_sim, Clus_size_sim)
# HMA_balance_sim = Hin_sim-Hse_sim
Hin_node_sim,Hse_node_sim = HMA.nodal_measures(sFC, Clus_num_sim, Clus_size_sim)

#empirica
Clus_num_emp,Clus_size_emp,H_all_emp = HMA.Functional_HP(empFC)
Hin_emp,Hse_emp = HMA.Balance(empFC, Clus_num_emp, Clus_size_emp)
# HMA_balance_emp = Hin_emp-Hse_emp
Hin_node_emp,Hse_node_emp = HMA.nodal_measures(empFC, Clus_num_emp, Clus_size_emp)

balance_df = pd.DataFrame(index = ["sim","emp"])
balance_df["integration"] = [Hin_sim,Hin_emp]
balance_df["segregation"] = [Hse_sim,Hse_emp]
balance_df["(int-seg)"] = balance_df["integration"]-balance_df["segregation"]
print(state,"\n",balance_df)



#%% plot simulated
plt.figure(80)
plt.clf()
plt.title(f"Functional Connectivity",weight="bold",fontsize=20)
im=plt.imshow(sFC,cmap="jet",aspect=1,vmin=0,vmax=1)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=20)
plt.xticks(fontsize=20);plt.yticks(fontsize=20)
plt.tight_layout()
# plt.savefig("images/homogeneous_N2.svg")
plt.show()

# np.save(f"data/heterogeneous_{state}_simFC.npy",sFC)


#%% see empirial FC just in case
plt.figure(81)
plt.clf()
plt.suptitle("states")
for s,st in enumerate(states):
    plt.subplot(2,2,s+1)
    plt.title(st,weight="bold",fontsize=15)
    plt.imshow(empFCs[st],cmap="jet",aspect=1,vmin=0,vmax=1)
    plt.colorbar()
plt.tight_layout()
plt.show()
#%% MIRAR LA FRECUENCIA DE EEG
ROI = np.random.choice(range(90),10)

dt = wc.dt
timemask = (0<=wc.time) & (wc.time<40)

EEG = E_t
freqs, PSDs = signal.welch(EEG, 1000//wc.downsamp, 'hann', 4000//wc.downsamp, 2000//wc.downsamp, axis = 0, scaling = 'density')

sampling_rate = 1/wc.dt
freqs, PSDs = signal.welch(E_t, fs = sampling_rate, window = 'hann',
                                    nperseg = sampling_rate*2, noverlap = sampling_rate*1,
                                    axis = 0)
PSD = np.mean(PSDs, axis = 1)

plt.figure(2)
plt.clf()
plt.subplot(3,1,1)
plt.title("EEG signal?")
plt.plot(wc.time[timemask], E_t[:,ROI][timemask])
maxfreq = freqs[np.argmax(PSD)]
print(f"peakfreq={maxfreq}")

plt.subplot(3,1,2)
plt.plot(freqs,PSD,color="black",linewidth=5)
for roi in ROI:
    plt.plot(freqs,PSDs[:,roi])
plt.vlines(maxfreq,0.00001,0.1,color="red",label=f"max={maxfreq}")
plt.legend()
plt.yscale('log')
plt.xscale('log')

plt.subplot(3,1,3)
plt.title("fMRI")
for roi in ROI:
    plt.plot(BOLDtime,BOLD[:,roi])
plt.tight_layout()
plt.show()
#%%

plt.figure(78)
plt.clf()
plt.title("fMRI")
for roi in ROI:
    plt.plot(BOLDtime,BOLD[:,roi],linewidth=2)
plt.yticks([-0.001,-0.0005,0,0.0005,0.001],fontsize=30)
plt.xticks([20,25,30,35,40],fontsize=30)
plt.xlim([20,40])
plt.tight_layout()
plt.show()

#%%

Z = (E_t-E_t.mean(axis=0))/E_t.std(axis=0)

plt.figure(55)
plt.clf()
plt.title("Z signal jejejjejej")
plt.plot(wc.time, Z[:,ROI])
plt.show()

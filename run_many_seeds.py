# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:23:57 2022

@author: flehu
"""
import sys
sys.path.append("integration_segregation/")
sys.path.append("../")
import HMA
import numpy as np
import netwWilsonCowanPlastic as wc
import os
from scipy import signal
import BOLDModel as BD
from scipy import signal 
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import utils
import gc
import itertools
import pickle

hilbert = signal.hilbert

rank=int(os.environ['SLURM_ARRAY_TASK_ID'])
threads=int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1

states = ("W","N1","N2","N3")
# state = "N1"

##formato (G,delta_G,sigmaE,delta_sigmaE)
########obtained from heatmaps.py
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

modality = "map"

map_folder = "empirical/maps/"
if modality == "homo":
    ach_dist = np.ones(90)
    na_dist = np.ones(90)
elif modality =="map":
    ach_dist = np.load(map_folder+"DIST_VAChT_feobv_hc18_aghourian.npy")
    na_dist = np.load(map_folder+"DIST_LC_proj.npy")

elif modality == "shuf":
    ach_dist = np.load(map_folder+"SHUFFLED_SYMM_DIST_VAChT_feobv_hc18_aghourian.npy")
    na_dist = np.load(map_folder+"SHUFFLED_SYMM_DIST_LC_proj.npy")

ach_dist = ach_dist/ach_dist.mean() ##seteamos media a 1
na_dist = na_dist/na_dist.mean()
##semillas
n_iterations = 50
n_init = 0

mapnames1 = ["HOMO",
            "DIST_VAChT_feobv_hc18_aghourian",
            "SHUFFLED_SYMM_DIST_VAChT_feobv_hc18_aghourian"]

mapnames2 = ["HOMO",
            "DIST_LC_proj",
            "SHUFFLED_SYMM_DIST_LC_proj"]

struct = np.loadtxt("SC_opti_25julio.txt")
# struct = utils.sub_weight(original_struct, 1) ##aquí está conectado el tálamo
empFCW,empFCN1,empFCN2,empFCN3 = [np.loadtxt(f"../../analyze_empirical/mean_arctanhrho_filtered_{s}.txt") for s in states]
wc.P = 0.4
wc.rhoE = 0.18
wc.CM = struct


wc.tTrans1=1  # simulation to remove transient
wc.tTrans2=400
wc.timeTrans1=np.arange(0,wc.tTrans1,wc.dtSim) # time vector f or transient Sim
wc.timeTrans2=np.arange(0,wc.tTrans2,wc.dtSim)
tstop = 600
wc.tstop = tstop
wc.timeSim=np.arange(0,tstop,wc.dtSim) ##shorter
wc.time=np.arange(0,tstop,wc.dt)

time = wc.timeSim[::wc.downsamp]


#%% simulation


seeds = np.array(range(n_init,n_init+n_iterations)) 
sims = list(itertools.product(seeds,states))

save_dic = {}

for sim,cuatrio in enumerate(sims):
    if sim%threads == rank:
        seed,state = cuatrio
        wc.sid = seed
        
        G_val,deltaG,sigmaE_val,deltasigmaE = eval(f"optimals_{modality}['{state}']")
        

        local_G = G_val + ach_dist*deltaG
        local_sigmaE = sigmaE_val+na_dist*deltasigmaE

        wc.sigmaE = local_sigmaE 
        wc.G = local_G
        
        
        wc.nnodes = 90
        wc.run.recompile()

        
        tray = wc.run()#[:,:,utils.cortex]
        E_t = tray[:,0,:]
        BOLD = wc.simBOLD(E_t,nnodes=90)
        sFC = np.corrcoef(BOLD.T)
        
        
        Clus_num_sim,Clus_size_sim,H_all_sim = HMA.Functional_HP(sFC)
        Hin_sim,Hse_sim = HMA.Balance(sFC, Clus_num_sim, Clus_size_sim)
        # HMA_balance_sim = Hin_sim-Hse_sim
        Hin_node_sim,Hse_node_sim = HMA.nodal_measures(sFC, Clus_num_sim, Clus_size_sim)
        save_dic[(seed,state)] = {"Hin_sim":Hin_sim,"Hse_sim":Hse_sim,
                                     "Hin_node_sim":Hin_node_sim,"Hse_node_sim":Hse_node_sim,
                                     "sFC":sFC}

        del tray,E_t,BOLD,sFC
        
        gc.collect()
    gc.collect()
gc.collect()

outfile = f'output/temp/run_50seeds_output_{modality}_16dic_rank{rank}.pickle'
with open(outfile, 'wb') as f: #cargamos FCs
    pickle.dump(save_dic,f)

##this was done also with empirical matrices, but using 15 individuals instead of 50 seeds
##this is done in generate_plots_fig1
##then it is collapsed in "output/run_50seeds_output_{modality}_16dic.pickle"
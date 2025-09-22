    # -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:23:57 2022

@author: flehu
"""
import sys
sys.path.append("../")
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

hilbert = signal.hilbert

rank=int(os.environ['SLURM_ARRAY_TASK_ID'])
threads=int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1

##semillas
n_iterations = 25
n_init = 25

this_G = 0.16
this_sigma = 7.68

map_id1,map_id2 = 2,2 ##math the ids of the previous lists, 2,2 is shuffled
out= f"output/temp/sweep_deltaSHUFFLED_from_homoW_fromG{this_G}_sigma{this_sigma}_maps_{map_id1}_{map_id2}_9dic24_{n_iterations}iter_from{n_init}_rank{rank}.txt"
##to be collapsed later in "output/sweep_deltaSHUFFLED_from_homoW_fromG0.16_sigma7.68_maps_2_2_9dic24_50iter.txt"

struct = np.loadtxt("SC_opti_25julio.txt")


states = ("W","N1","N2","N3")
empFCW,empFCN1,empFCN2,empFCN3 = [np.loadtxt(f"mean_mat_{s}_8dic24.txt") for s in states]
map_loc = "empirical/maps/"

mapnames1 = ["HOMO",
            "DIST_VAChT_feobv_hc18_aghourian",
            "SHUFFLED_SYMM_DIST_VAChT_feobv_hc18_aghourian"]

mapnames2 = ["HOMO",
            "DIST_LC_proj",
            "SHUFFLED_SYMM_DIST_LC_proj"]

namemap1 = mapnames1[map_id1]
if namemap1 == "HOMO":
    this_map1 = np.ones(90) ##caso homogeneo simplemente aumenta G
else:
    this_map1 = np.load(map_loc+namemap1+".npy")
this_map1 = this_map1/this_map1.mean() #we normalize mean to 1


namemap2 = mapnames2[map_id2]
if namemap2 == "HOMO":
    this_map2 = np.ones(90) ##caso homogeneo simplemente aumenta G
else:
    this_map2 = np.load(map_loc+namemap2+".npy")
this_map2 = this_map2/this_map2.mean()

#############AQUI PONEMOS EL BASAL, INCLUSO CON MAPA



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

#note: this is a zoomed in version of the ranges mentioned in the manuscript
delta_G_vals = np.linspace(-0.1,0.3,20,endpoint=False)
delta_sigma_vals = np.linspace(-0.2,0.2,20,endpoint=False)
# delta_sigma_vals = np.linspace(-0.2,0.2,20,endpoint=False)

seeds = np.array(range(n_init,n_init+n_iterations)) 
sims = list(itertools.product(seeds,delta_G_vals,delta_sigma_vals))

for sim,cuatrio in enumerate(sims):
    if sim%threads == rank:
        seed,this_G_delta,this_sigma_delta = cuatrio
        wc.sid = seed

        local_G = this_G + this_G_delta*this_map1
        wc.G = local_G
        
        local_sigma = this_sigma + this_sigma_delta*this_map2
        wc.sigmaE = local_sigma

        wc.nnodes = 90
        wc.run.recompile()

        
        tray = wc.run()#[:,:,utils.cortex]
        E_t = tray[:,0,:]
        BOLD = wc.simBOLD(E_t,nnodes=90)
        sFC = np.corrcoef(BOLD.T)
        
        W_metrics  = utils.get_all_metrics(sFC,empFCW, data_range = 1)
        N1_metrics = utils.get_all_metrics(sFC,empFCN1, data_range = 1)
        N2_metrics = utils.get_all_metrics(sFC,empFCN2, data_range = 1)
        N3_metrics = utils.get_all_metrics(sFC,empFCN3, data_range = 1)
        
        #measures
        
        freqs,fftPow=signal.welch(E_t.T,fs=1/wc.dt,nperseg=4000)
        meanpow = fftPow.mean(axis=0)
        maxmeanpow = np.where(meanpow==meanpow.max())[0][0]
        peakfreq = freqs[maxmeanpow]

        sync,meta = utils.kuramoto(BOLD) ##notar que aquí sí condsideramos las otras áreas
        mean = np.mean(sFC)

        #free space
        del tray,E_t,BOLD,sFC
        # gc.collect()
            
            
            
        corrW,eucW,ssimW,_ = W_metrics
        corrN1,eucN1,ssimN1,_ = N1_metrics
        corrN2,eucN2,ssimN2,_ = N2_metrics
        corrN3,eucN3,ssimN3,_ = N3_metrics
        
        
        
        if not os.path.isfile(out):
            with open(out,'w') as f:
                f.write("rank\tseed\tdelta_G\tdelta_sigma\tssimW\tssimN1\tssimN2\tssimN3\tcorrW\tcorrN1\tcorrN2\tcorrN3\teW\teN1\teN2\teN3\tsync\tmeta\tmean\tpeakfreq\n")
        with open(out,'a') as f:
            f.write(f"{rank}\t{seed}\t{this_G_delta:.4f}\t{this_sigma_delta:.4f}\t{ssimW:.4f}\t{ssimN1:.4f}\t{ssimN2:.4f}\t{ssimN3:.4f}\t{corrW:.4f}\t{corrN1:.4f}\t{corrN2:.4f}\t{corrN3:.4f}\t{eucW:.4f}\t{eucN1:.4f}\t{eucN2:.4f}\t{eucN3:.4f}\t{sync:.4f}\t{meta:.4f}\t{mean:.4f}\t{peakfreq:.4f}\n")
        
        gc.collect()
    gc.collect()
gc.collect()

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
n_iterations = 50
n_init = 0

this_sigma,this_G = 7.68,0.16 ##W optimal, obtained after exploration
out = "output/temp/sweep_delta_homoW_fromG0.16_sigma7.68_maps_0_0_9dic24_{}iter_from{}_rank{}".format(n_iterations,n_init,rank)
##note, we then concatenate everything together in the file "empirical/sweep_delta_homoW_fromG0.16_sigma7.68_maps_0_0_9dic24_50iter.txt"

struct = np.loadtxt("SC_opti_25julio.txt")

states = ("W","N1","N2","N3")
empFCW,empFCN1,empFCN2,empFCN3 = [np.loadtxt(f"empirical/mean_mat_{s}_8dic24.txt") for s in states]

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
delta_G_vals = np.linspace(-0.1,0.5,20,endpoint=False)
delta_sigma_vals = np.linspace(-1,1,20,endpoint=False)

seeds = np.array(range(n_init,n_init+n_iterations)) 
sims = list(itertools.product(seeds,delta_G_vals,delta_sigma_vals))

for sim,cuatrio in enumerate(sims):
    if sim%threads == rank:
        seed,this_G_delta,this_sigma_delta = cuatrio
        wc.sid = seed

        local_G = this_G + this_G_delta
        wc.G = local_G
        
        local_sigma = this_sigma + this_sigma_delta
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
        this_sync,this_meta = utils.kuramoto(BOLD) ##notar que aquí sí condsideramos las otras áreas
        this_mean = np.mean(sFC)
        this_peakfreq = freqs[maxmeanpow]
        measures = this_sync,this_meta,this_mean,this_peakfreq

        #free space
        del tray,E_t,BOLD,sFC
        # gc.collect()
            
            
            
        corrW,eucW,ssimW,new_metricW = W_metrics
        corrN1,eucN1,ssimN1,new_metricN1 = N1_metrics
        corrN2,eucN2,ssimN2,new_metricN2 = N2_metrics
        corrN3,eucN3,ssimN3,new_metricN3 = N3_metrics
        sync,meta,mean,peakfreq = measures
        
        
        
        if not os.path.isfile(out):
            with open(out,'w') as f:
                f.write("rank\tseed\tdelta_G\tdelta_sigma\tssimW\tssimN1\tssimN2\tssimN3\tcorrW\tcorrN1\tcorrN2\tcorrN3\teW\teN1\teN2\teN3\tsync\tmeta\tmean\tpeakfreq\n")
        with open(out,'a') as f:
            f.write(f"{rank}\t{seed}\t{this_G_delta:.4f}\t{this_sigma_delta:.4f}\t{ssimW:.4f}\t{ssimN1:.4f}\t{ssimN2:.4f}\t{ssimN3:.4f}\t{corrW:.4f}\t{corrN1:.4f}\t{corrN2:.4f}\t{corrN3:.4f}\t{eucW:.4f}\t{eucN1:.4f}\t{eucN2:.4f}\t{eucN3:.4f}\t{sync:.4f}\t{meta:.4f}\t{mean:.4f}\t{peakfreq:.4f}\n")
        
        gc.collect()
    gc.collect()
gc.collect()

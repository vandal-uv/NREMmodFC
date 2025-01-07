# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 13:03:01 2022

@author: flehu
"""

import numpy as np
#import netwWilsonCowanPlastic as wc
import os
from scipy import signal
#import BOLDModel as BD
from scipy import signal 
from skimage.metrics import structural_similarity as ssim
import pandas as pd
hilbert = signal.hilbert

def cohen_d(x,y): ##diferencia entre dos distribuciones en terminos de tamaño de efecto
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def flat_FC(FC):
    lenfc = len(FC)
    return np.concatenate([FC[i,i+1:] for i in range(lenfc)])

def new_metric(flat1,flat2):
    this_corr_back = 1-np.corrcoef(flat1,flat2)[0,1]
    mean_dif_sq = (flat1.mean()-flat2.mean())**2
    return this_corr_back+mean_dif_sq


def kuramoto(sign):
    analytic = hilbert(sign,axis=0)
    angle = np.angle(analytic)
    kuramoton = np.abs(np.mean(np.exp(1j*angle),axis=1))
    sync = kuramoton.mean()
    meta = kuramoton.std()
    return sync,meta

def get_all_metrics(sFC,empFC,data_range=1):
    lenny = len(sFC)
    flat_empFC = np.concatenate([empFC[i,i+1:] for i in range(lenny)])
    flat_sFC = np.concatenate([sFC[i,i+1:] for i in range(lenny)])
    this_corr = np.corrcoef(flat_sFC,flat_empFC)[0,1]
    this_euc = np.linalg.norm(flat_empFC-flat_sFC)
    this_ssim = ssim(sFC,empFC,data_range = data_range)
    this_new_metric = new_metric(flat_sFC,flat_empFC)
    return (this_corr,this_euc,this_ssim,this_new_metric)

thal = [38,51]


subL, subR = [35,36,37,38],[51,52,53,54]
sub = subL+subR
cortex = [i for i in range(90) if i not in sub]
def sub_weight(struct,prop=1): ########RECORDAR QUE SON PROPORCION DE LA ORIGINAL
    
    
    out = np.copy(struct)
    out[:,sub] = prop*struct[:,sub] ###todo lo que llega a la corteza    
    out[sub,:] = prop*struct[sub,:] ##prueba
    for i in sub:
        for j in sub:
            out[i,j] = struct[i,j]
    return out


def cortex_mat(mat90):
    subL, subR = [35,36,37,38],[51,52,53,54]
    sub = subL+subR
    cortex = [i for i in range(90) if i not in sub]
    return mat90[cortex,:][:,cortex]


#############ENTRADAS POR RSN
#RSN_index = []
#RSNs = np.loadtxt("../RSN_AAL_Enzo.txt")
#RSN_labels = ["Vis","ES","Aud","SM","DM","EC"]
#for i in range(6):
#    these_nodes = RSNs[:,i]==1
#    RSN_index.append(np.array(range(90))[these_nodes])
##########

def RSN_profile_FC(FC):
    ##asumimos que la señal de BOLD viene filtrada
    profile = []
    for k,rsn in enumerate(RSN_index):
        subFC = FC[rsn,:][:,rsn]
        flat_subFC = flat_FC(subFC)
        mean = flat_subFC.mean()
        profile.append(mean)
    return np.array(profile)


def find_extreme(df,targetval,ex="min", cols=None):
    if ex =="min":
        exval = df[targetval].min()
    else:
        exval = df[targetval].max()
    
    if cols:
        return df[df[targetval] == exval].iloc[0][cols]
    return df[df[targetval] == exval].iloc[0]

def xy2plotcor(x,y,xvals,yvals):
    lennyx,minx,maxx = len(xvals),min(xvals),max(xvals)
    mx = (lennyx-1)/(maxx-minx)
    nx = -minx*mx
    lennyy,miny,maxy = len(yvals),min(yvals),max(yvals)
    my = (lennyy-1)/(maxy-miny)
    ny = -miny*my
    xcor = x*mx+nx
    ycor= y*my+ny
    return xcor,ycor

def fill_missing(df,col1,col2,what=np.nan):
    df2 = df
    idx1,idx2 = list(df.columns).index(col1),list(df.columns).index(col2) ##lugares de las columnas
    vals1,vals2 = df[col1],df[col2] #valores que quiero
    
    present = df[[col1,col2]].values ##los que estan 
    here = [(present[i,0],present[i,1]) for i in range(len(df))]
    
    for x in vals1:
        for y in vals2:
            if (x,y) not in here:
                lis = what*np.ones(df.shape[1]) ##llenamos con basura todo
                lis[idx1],lis[idx2] = x,y ##menos las columnas de interes
                df2 = df2.append(pd.DataFrame(lis[:,None].T,columns=df.columns))
    return df2

thal = [38,51]
not_thal = [i for i in range(90) if i not in thal]
#cada fila es lo que entra a la corteza
def scale_mat(struct,G,subG): ##aqui subG puede ser un vector
    out = np.copy(struct)
    for area in thal:
        out[:,area] = subG*struct[:,area]
    for area in not_thal:
        out[:,area] = G*struct[:,area]
    return out

def envelope(series,dt=0.002,freqs = [8,13],fcut=None): #series viene en (tiempo,rois)
    fmin,fmax = freqs
    a0,b0 = signal.bessel(3, [2 * dt * fmin, 2 * dt * fmax], btype = 'bandpass')
    Vfilt = signal.filtfilt(a0, b0, series, axis = 0)
    envelopes = np.abs(signal.hilbert(Vfilt,axis=0))
    if fcut:
        a0,b0 = signal.bessel(3, [2 * dt * fcut], btype = 'low') #low-pass filtering
        Efilt = signal.filtfilt(a0, b0, envelopes, axis = 0) #filtered envelopes
        return Efilt
    return envelopes
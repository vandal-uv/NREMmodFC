# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:36:24 2020


Hopf whole-brain model adapted from [1]. Used to simulate fMRI BOLD-like signals.

Deco, G., Cruzat, J., Cabral, J., Tagliazucchi, E., Laufs, H., Logothetis, N. K., 
& Kringelbach, M. L. (2019). Awakening: Predicting external stimulation to force 
transitions between different brain states. Proceedings of the National Academy 
of Sciences, 116(36), 18088-18097.

@author: Carlos Coronel
"""

import numpy as np
import networkx as nx
from numba import njit,jit,float64
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

#Model parameters
a = 0 #External input (bifurcation parameter)
w = 0.05 * 2 * np.pi #Oscillatory frequency
#Note: frequencies could be different between brain regions and also obtained using the real data
beta = 0.032 #noise scaling factor

#Simulation parameters
dt = 1E-1 #Integration step
teq = 60 #Equilibrium time
tmax = 1200 #Signals' length
downsamp = 1 #This reduces the number of points of the signals by X

#Network parameters
nnodes = 90 #number of nodes
ones_vector = np.ones(nnodes).reshape((1,nnodes))
#Structural connectivity matrix
M = nx.to_numpy_array(nx.watts_strogatz_graph(nnodes,8,0.075))
norm = np.mean(np.sum(M,0)) #global normalization
G = 0.25 #Global coupling
seed = 0 #Random seed

# @jit(float64[:,:](float64[:,:],float64[:,:],float64),nopython=True)
#Hopf multi-column model
@njit
def Hopf_model(x,y,t):
    
    
    deltaX = (x @ ones_vector).T - (x @ ones_vector)
    deltaY = (y @ ones_vector).T - (y @ ones_vector)
    
    IsynX = G * M / norm * deltaX @ ones_vector.T
    IsynY = G * M / norm * deltaY @ ones_vector.T
    
    x_dot = (a - x**2 - y**2) * x - w * y + IsynX 
    y_dot = (a - x**2 - y**2) * y + w * x + IsynY
    
    return(np.hstack((x_dot,y_dot)))

# @jit(float64[:,:](float64[:,:],float64[:,:],float64),nopython=True)
#Stochastic part of the model
@njit
def Noise(x,y,t):
       
    x_dot = np.random.normal(0,1,((nnodes,1))) * beta
    y_dot = np.random.normal(0,1,((nnodes,1))) * beta
    
    return(np.hstack((x_dot,y_dot)))

# @jit(float64(float64),nopython=True)
#This function is just for setting the random seed
@njit
def set_seed(seed):
    np.random.seed(seed)
    return(seed)


def Sim(verbose = True):
    """
    Run a network simulation with the current parameter values.
    
    Note that the time unit in this model is seconds.

    Parameters
    ----------
    verbose : Boolean, optional
        If True, some intermediate messages are shown.
        The default is False.

    Raises
    ------
    ValueError
        An error raises if the dimensions of C and the number of nodes
        do not match.

    Returns
    -------
    x : ndarray
        Time trajectory for the x variable of each node.
    y : ndarray
        Time trajectory for the y variable of each node.    
    time_vector : numpy array (vector)
        Values of time.
        
    """
    global teq,dt,tmax,M,seed,nnodes,downsamp,seed
         
    if M.shape[0]!=M.shape[1] or M.shape[0]!=nnodes:
        raise ValueError("check M dimensions (",M.shape,") and number of nodes (",nnodes,")")
    
    if M.dtype is not np.dtype('float64'):
        try:
            M=M.astype(np.float64)
        except:
            raise TypeError("M must be of numeric type, preferred float")    
    
    set_seed(seed) #Set the random seed    
   
    Neq = int(teq / dt / downsamp) #Number of points to discard
    Nmax = int(tmax / dt / downsamp) #Number of points of the signals
    tsim = teq + tmax #total simulation time
    Nsim = int(tsim / dt) #total simulation points without downsampling
    
    #Initial conditions
    ic = np.ones((nnodes,2)) * np.random.uniform(0.01,1,((nnodes,2)))
    results = np.zeros((Nmax + Neq,nnodes,2))
    results[0,:,:] = np.copy(ic)
    results_temp = np.copy(ic) #Temporal vector to update y values 
       
    #Time vector
    time_vector = np.linspace(0, tmax, Nmax)

    Hopf_model.recompile()
    Noise.recompile()

    if verbose == True:
        for i in range(1,Nsim):
            results_temp += Hopf_model(results_temp[:,[0]],results_temp[:,[1]],i) * dt +\
                            Noise(results_temp[:,[0]],results_temp[:,[1]],i) * np.sqrt(dt)
            #This line is for store values each 'downsamp' points
            if (i % downsamp) == 0:
                results[i//downsamp,:,:] = results_temp
            if (i % (10 / dt)) == 0:
                print('Elapsed time: %i seconds'%(i * dt)) #this is for impatient people
    else:
        for i in range(1,Nsim):
            results_temp += Hopf_model(results_temp[:,[0]],results_temp[:,[1]],i) * dt +\
                            Noise(results_temp[:,[0]],results_temp[:,[1]],i) * np.sqrt(dt)
            #This line is for store values each 'downsamp' points
            if (i % downsamp) == 0:
                results[i//downsamp,:,:] = results_temp
    
    results = results[Neq:,:,:]
    
    return(results, time_vector)


def ParamsNode():
    pardict={}
    for var in ('a','beta','w'):
        pardict[var]=eval(var)
        
    return pardict

def ParamsNet():
    pardict={}
    for var in ('nnodes','G','norm'):
        pardict[var]=eval(var)
        
    return pardict

def ParamsSim():
    pardict={}
    for var in ('tmax','teq','dt','downsamp','seed'):
        pardict[var]=eval(var)
        
    return pardict










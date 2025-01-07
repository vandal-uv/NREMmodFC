# -*- coding: utf-8 -*-
"""
Network of Wilson-Cowan oscillators with homeostatic plasticity
Plus some basic analysis
@author: porio
"""

import numpy as np
from numba import jit,float64, vectorize,njit
import BOLDModel as BD
from numba.core.errors import NumbaPerformanceWarning
import warnings
from scipy import signal
import gc
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


###  MODEL PARAMETERS  ####

#Node parameters
# Any of them can be redefined as a vector of length nnodes
#excitatory connections
a_ee=3.5; a_ie_0=2.5
#inhibitory connections
a_ei=3.75; a_ii=0
#tau
tauE=0.010; tauI=0.020  # Units: seconds
#external input
P = 0.4 # 0.4
Q = 0
# inhibitory plasticity
rhoE=0.14 # target mean value for E
tau_ip=2  #time constant for plasticity

rE,rI=0.5,0.5
mu = 1;
sigmaE = 4#0.25
sigmaI=4#0.25 ##valor original es 0.25


### Time units are seconds  ###
tTrans1=600  # simulation to remove transient (runs twice)
tTrans2=600
tstop=600 # length of actual simulation
dt=0.002    #interval for points storage (sampling interval)
dtSim=0.0001   #interval for simulation (ODE integration)
downsamp=int(dt/dtSim)

timeTrans1=np.arange(0,tTrans1,dtSim) # time vector for transient Sim
timeTrans2=np.arange(0,tTrans2,dtSim)
timeSim=np.arange(0,tstop,dtSim)  # time vector for actual Sim
time=np.arange(0,tstop,dt)   # time vector for data storage


# Noise factor
D=0.002 #0.002
sqdtD=D/np.sqrt(dtSim)
sid = 12
np.random.seed(sid)

#network parameters
G=0.7
# CM = np.loadtxt("../structural/struct_antidiagonal_set_to_overallmax.txt")
CM = np.random.uniform(size=(90,90))
nnodes = len(CM)
# CM=np.load("netHier1.npy")

N=len(CM)  # make sure the number of nodes is equal to the size of CM

### MODEL FUNCTIONS  ###

@vectorize([float64(float64,float64,float64)],nopython=True)
def S(x,sigma,mu):
    return (1/(1+np.exp(-(x-mu)*sigma)))

# @jit(float64[:,:](float64,float64[:,:]),nopython=True)
@njit
def wilsonCowan(t,X,sigmaE,mu,tau_ip,G):
    E,I,a_ie = X
    noise=np.random.normal(0,sqdtD,size=N)
    return np.vstack(((-E + (1-rE*E)*S(a_ee*E - a_ie*I + G*np.dot(CM,E) + P + noise,sigmaE,mu))/tauE,
                     (-I + (1-rI*I)*S(a_ei*E - a_ii*I ,sigmaI,mu))/tauI,
                     (I*(E-rhoE))/tau_ip))

### HERE STARTS THE SIMULATION  ###
@njit
def run(verbose=False):

    # Initial conditions
    E0=0.1
    I0=0.1
    
    
    """
    Run two deterministic simulations of timeTrans. 
    First with tau_ip=0.05, Second with tau_ip= 1
    """
    
    Var=np.array([E0,I0,a_ie_0]).reshape(3,1)*np.ones((1,N))
    
    tau_ip=0.05
    # wilsonCowan.recompile()
    # if verbose:
    #     print(f"Starting first adaptation ({tTrans1} s)")
    
    # Varinit=np.zeros((len(timeTrans),3,N))
    for i,t in enumerate(timeTrans1):
        # Varinit[i]=Var
        Var+=dtSim*wilsonCowan(t,Var,sigmaE,mu,tau_ip,G)
      
    tau_ip=1
    # wilsonCowan.recompile()
    # if verbose:
    #     print(f"Starting second adaptation ({tTrans2 } s)")
    for i,t in enumerate(timeTrans2):
        Var+=dtSim*wilsonCowan(t,Var,sigmaE,mu,tau_ip,G)
    
    tau_ip=2
    
    downsamp=int(dt/dtSim)  # ratio between simulation dt and storage dt
             
    Y_t=np.zeros((len(time),3,N))  #Vector para guardar datos
    
    # if verbose:
    #     print(f"Simulating {tstop} s dt={dtSim}, Total {len(timeSim)} steps")
    
    # wilsonCowan.recompile()
    for i,t in enumerate(timeSim):
        if i%downsamp==0:   # if the current time step is a multiple of the storage rate,
            Y_t[i//downsamp]=Var   # we store the current variables
            
        # if t%50==0 and verbose:         # Every 50 simluated seconds we print some message
        #     print("%g of %g s"%(t,tstop))
            
        Var += dtSim*wilsonCowan(t,Var,sigmaE,mu,tau_ip,G)
    # gc.collect()
    return Y_t


def simBOLD(E_t,nnodes=90,BOLD_downsamp=1000):
    """
    Takes E_t and returns filtered BOLD
    """
    BOLD_signals = BD.Sim(E_t, nnodes, dt * downsamp)
    Neq = 2000
    BOLD_signals = BOLD_signals[Neq:,:]
    
    BOLD_dt = dt * downsamp 
    BOLD = BOLD_signals
    
    ##TR=2, freqs in [0.01,0.1]
    a,b = signal.bessel(2,[2 * 0.01 * BOLD_dt, 2 * 0.1 * BOLD_dt], btype = 'bandpass')
    BOLD = signal.filtfilt(a,b,BOLD,axis=0)
    # BOLD_2 = signal.filtfilt(a,b,BOLD_2,axis=0)
    # BOLD_downsamp = 1000 # sFC empFC corr = 0.3023
    BOLD = BOLD[::BOLD_downsamp]
    gc.collect()
    return BOLD
    

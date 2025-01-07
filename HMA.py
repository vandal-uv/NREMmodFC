# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:00:58 2021

Hierarchical modular partition of FC networks, Integration and Segregation related metrics
using eigenmode analysis [1,2]. The codes were adapted from 
https://github.com/TobousRong/Hierarchical-module-analysis


[1] Wang, R., Lin, P., Liu, M., Wu, Y., Zhou, T., & Zhou, C. (2019). 
Hierarchical connectome modes and critical state jointly maximize 
human brain functional diversity. Physical review letters, 123(3), 
038301.

[2] Wang, R., Liu, M., Cheng, X., Wu, Y., Hildebrandt, A., & Zhou, C. (2021). 
Segregation, integration, and balance of large-scale resting brain networks 
configure different cognitive abilities. Proceedings of the National 
Academy of Sciences, 118(23).

[3] Wang, R., Fan, Y., Wu, Y., & Zhou, C. (2021). Heterogeneous aging trajectories 
within resting-state brain networks predict distinct ADHD symptoms in adults. 
arXiv preprint arXiv:2107.13219.

@author: Carlos Coronel
"""

import numpy as np


def Functional_HP(FC):
    '''
    This function uses an eigenmode-based analysis to detect the hierarchical
    modules in FC networks.
    
    Parameters
    ----------
    FC : numpy array.
         functional connectivity matrix.
         
    Returns
    -------
    Clus_num : list.
               number of modules found at each eigenmode level.
    Clus_size : list of arrays.
               number of nodes belonging to each module at each eigenmode level. 
    H_all : nested list of arrays
            it contains all the assignments to each module within hierarchies.
    '''
    
    N = FC.shape[0] #number of ROIs
    H_all= [] #all modules assignments 
    
    #This method requires the complete positive connectivity in FC matrix, 
    #that generates the global integration in the first level. 
    FC[FC < 0] = 0
    FC = (FC + FC.T) / 2
    
    #singular value decomposition, where 'u' and 'v' are the eigenvectors, and 
    #'s' the singular values (eigenvalues).
    u, s, v = np.linalg.svd(FC)
    s[s<0] = 0 #For avoiding negative eigenvalues
    
    #Number of nodes in each module at 1st level.
    H1_1 = np.argwhere(u[:,1] < 0)[:,0]
    H1_2 = np.argwhere(u[:,1] >= 0)[:,0]
    
    H1 = [] #modules of mode 1
    H1.append(H1_1)
    H1.append(H1_2)
    H_all.append(H1)
    
    Clus_num= [1] #The first level has one module and corresponds to the global integration.  
    Clus_size = [[N]] #The first level has one module with N total number of ROIs (nodes).
    for mode in range(1,N-1):
        
        exec('H%i = []'%(mode + 1))
        
        x = np.argwhere(u[:,mode + 1] >= 0)[:,0]
        y = np.argwhere(u[:,mode + 1] < 0)[:,0]
        H = []
        for j in range(0, 2 * Clus_num[mode-1]):
            #assume the number of cluster in j-1 level is 2^(mode-1)
            H.append(eval('H' + '%s_%s'%(mode,j+1))) 
        
        idx = np.array([len(index) for index in H]) #length of each cluster in H
        #Delete the cluster with 0 size
        H = [H[full] for full in range(0,len(idx)) if idx[full] != 0] 
        idx = [idx[full] for full in range(0,len(idx)) if idx[full] != 0] 
        
        Clus_size.append(idx)
        Clus_num.append(len(H)) #number of cluster
        k = 0
        for j in range(0, 2 * Clus_num[mode], 2): #modular number
            Positive_Node = np.intersect1d(H[k], x)    
            Negative_Node = np.intersect1d(H[k], y)    
            k = k + 1
            exec('H' + '%s_%s'%((mode + 1), (j + 2)) + '=' + 'Positive_Node')
            exec('H' + '%s_%s'%((mode + 1), (j + 1)) + '=' + 'Negative_Node')
            exec('H%i.append(H%s_%s)'%((mode + 1),(mode + 1), (j + 2)))
            exec('H%i.append(H%s_%s)'%((mode + 1),(mode + 1), (j + 1)))  
        exec('H_all.append(H%i)'%(mode + 1))
    
    return([Clus_num,Clus_size,H_all])
        


def Balance(FC, Clus_num, Clus_size):
    '''
    This function calculates the integration and segregation components.
    
    Parameters
    ----------
    FC : numpy array.
         functional connectivity matrix.
    Clus_num : list.
               number of modules found at each eigenmode level.
    Clus_size: list of arrays.
               number of nodes belonging to each module at each eigenmode level.          
         
    Returns
    -------
    Hin : float.
          Integration component.
    Hse : float.
          Segregation component.   
    '''
    

    N = FC.shape[0] #number of ROIs

    #This method requires the complete positive connectivity in FC matrix, 
    #that generates the global integration in the first level. 
    FC[FC < 0] = 0
    FC = (FC + FC.T) / 2

    #singular value decomposition, where 'u' and 'v' are the eigenvectors, and 
    #'s' the singular values (eigenvalues).    
    u, s, v = np.linalg.svd(FC)
    s[s<0] = 0
    s = s ** 2 #using the squared Lambda
    
    p = np.zeros(N-1)
    #modular size correction
    for i in range(0,len(Clus_num) - 1):
        p[i] = np.sum(np.abs(np.array(Clus_size[i]) - N / Clus_num[i])) / N 
    
    HF = s[0:(N-1)] * np.array(Clus_num) *(1-p);
    Hin = np.sum(HF[0]) / N**2 #integration component
    Hse = np.sum(HF[1:(N-1)]) / N**2 #segregation component
    
    return([Hin,Hse])



def nodal_measures(FC, Clus_num, Clus_size):
    
    '''
    This function calculates the nodal (regional) integration and segregation components.
    
    Parameters
    ----------
    FC : numpy array.
         functional connectivity matrix.
    Clus_num : list.
               number of modules found at each eigenmode level.
    Clus_size: list of arrays.
               number of nodes belonging to each module at each eigenmode level.          
         
    Returns
    -------
    Hin_nodal : list, float.
                Integration component of the N total number of nodes.
    Hse_nodal : list, float.
                Segregation component of the N total number of nodes.   
    '''    
    
    N = FC.shape[0] #number of ROIs

    #This method requires the complete positive connectivity in FC matrix, 
    #that generates the global integration in the first level. 
    FC[FC < 0] = 0
    FC = (FC + FC.T) / 2

    #singular value decomposition, where 'u' and 'v' are the eigenvectors, and 
    #'s' the singular values (eigenvalues).    
    u, s, v = np.linalg.svd(FC)
    s[s<0] = 0
    s = s ** 2 #using the squared Lambda
    
    p = np.zeros(N-1)
    #modular size correction
    for i in range(0,len(Clus_num) - 1):
        p[i] = np.sum(np.abs(np.array(Clus_size[i]) - N / Clus_num[i])) / N 
    
    HF = s[0:(N-1)] * np.array(Clus_num) *(1-p)
    
    #Nodal (regional) integration and segregation components
    Hin_nodal = HF[0] / N * u[:,0]**2
    Hse_nodal = np.zeros(N)
    for i in range(1,N-1):
        Hse_nodal += HF[i] / N * u[:,i]**2
    
    return([Hin_nodal,Hse_nodal])
    
    
    
    
    
        
        
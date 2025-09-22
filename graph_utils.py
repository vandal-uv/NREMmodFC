#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:09:16 2019

[1] Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: 
uses and interpretations. Neuroimage, 52(3), 1059-1069.

[2] Lancichinetti, A., & Fortunato, S. (2012). Consensus clustering in complex 
networks. Scientific reports, 2, 336.

https://github.com/tsalo/brainconn

@author: Carlos Coronel
"""

import numpy as np
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
#import sys
#sys.path.insert(1, '/home/ccoronel/filer/Otros/')
#sys.path.insert(1, 'Z:/Otros/')
#import connectivity

# import networkx as nx
# import community

def contradiagonalize(matriz):
    matriz2 = np.copy(matriz)
    max_val = matriz.max()
    N = matriz.shape[0]
    for i in range(N):
        matriz2[i,-(i+1)] = max_val
    return matriz2



def cuberoot(x):
    """
    Correctly handle the cube root for negative weights, instead of uselessly
    crashing as in python or returning the wrong root as in matlab
    """
    return(np.sign(x) * np.abs(x)**(1 / 3))

def binarize(W, copy = True):
    """
    Binarizes an input weighted connection matrix.  If copy is not set, this
    function will *modify W in place.*
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.
    Returns
    -------
    W : NxN :obj:`numpy.ndarray`
        binary connectivity matrix
    """
    if copy:
        W = W.copy()
    W[W != 0] = 1
    # W = W.astype(int)  # causes tests to fail, but should be included
    return(W)


def invert(W, copy = True):
    """
    Inverts elementwise the weights in an input connection matrix.
    In other words, change the from the matrix of internode strengths to the
    matrix of internode distances.
    If copy is not set, this function will *modify W in place.*
    Parameters
    ----------
    W : :obj:`numpy.ndarray`
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.
    Returns
    -------
    W : :obj:`numpy.ndarray`
        inverted connectivity matrix
    """
    if copy:
        W = W.copy()
    E = np.where(W)
    W[E] = 1. / W[E]
    return(W)

def get_uptri(x):

    nnodes = x.shape[0]
    # vector = np.concatenate([x[:i,i] for i in range(nnodes)]).flatten()
    npairs = (nnodes**2 - nnodes) // 2
    vector = np.zeros(npairs)
    
    idx = 0
    for row in range(0, nnodes - 1):
        for col in range(row + 1, nnodes):
            vector[idx] = x[row, col]
            idx += 1
    
    return(vector)

def matrix_recon(x):
    npairs = len(x)
    nnodes = int((1 + np.sqrt(1 + 8 * npairs)) // 2)
    
    matrix = np.zeros((nnodes, nnodes))
    idx = 0
    for row in range(0, nnodes - 1):
        for col in range(row + 1, nnodes):
            matrix[row, col] = x[idx]
            idx = idx + 1
    matrix = matrix + matrix.T
   
    return(matrix)   


def Trapezoidal(x, y, h):
    N = len(x)
    iters = N // h - 1
    residual = N - N % h - 1
    AUC = np.zeros(y.shape[1])
    for i in np.arange(0, iters):
        a, b = i * h, (i + 1) * h
        AUC = AUC + (x[b] - x[a]) * (y[a,:] + y[b,:]) / 2
    AUC = AUC + (x[-1] - x[residual]) * (y[-1,:] + y[residual,:]) / 2
        
    return(AUC)


def thresholding(x, threshold = 0.20, zero_diag = True, direct = 'undirected'):
    nnodes = x.shape[0]
    if zero_diag == True:
        np.fill_diagonal(x, 0)
    
    if direct == 'directed':
        nlinks = nnodes**2 - nnodes
        x_vector = x.reshape((1, nnodes**2))
        to_get_links = int(nlinks * threshold)
        sorting = np.argsort(x_vector)[0,::-1]
        selection = sorting[0:to_get_links]
        to_delete = np.delete(np.arange(0, nnodes**2, 1), selection)
        x_thresholded = np.copy(x_vector[0,:])
        x_thresholded[to_delete] = 0
        x_thresholded = x_thresholded.reshape((nnodes, nnodes))
    elif direct == 'undirected':
        nlinks = (nnodes**2 - nnodes) // 2
        x_vector = get_uptri(x)
        to_get_links = int(nlinks * threshold)
        sorting = np.argsort(x_vector)[::-1]
        selection = sorting[0:to_get_links]
        to_delete = np.delete(np.arange(0, nlinks, 1), selection)
        x_vector[to_delete] = 0
        x_thresholded = matrix_recon(x_vector)
    else:
        print('Invalid type of matrix -> direct options: undirected or directed')
    
    return(x_thresholded)


def probabilistic_thresholding2(y, surr_number = 50, alpha = 0.05):
    
    nnodes = y.shape[1]
    L = y.shape[0]
    npairs = (nnodes**2 - nnodes) // 2
    matrix_surr = np.zeros((npairs, surr_number))
    
    FC_real = np.corrcoef(y.T)
    FC_real = FC_real[np.triu_indices(n = nnodes, k = 1)]
    
    for i in range(surr_number):
        
        np.random.seed(i + 1)
        
        y_ord = np.sort(y, axis = 0)
        ranks = np.argsort(y, axis = 0)
        rev_ranks = np.argsort(ranks, axis = 0)
        
        noise = np.random.normal(0,1,((L,nnodes)))
        noise_ord = np.sort(noise, axis = 0)
        
        noise_reord = np.zeros_like(noise_ord)
        for j in range(0,nnodes):
            noise_reord[:,j] = noise_ord[rev_ranks[:,j],j]
        
        noiseF = np.fft.fft(noise_reord, axis = 0)
        random_vector = np.random.uniform(0, 2 * np.pi, ((L//2, nnodes)))
        random_vector = np.row_stack((random_vector, random_vector[::-1,:]))
        phase_surrogate = noiseF * np.exp(1j * random_vector)
        surrogate = np.real(np.fft.ifft(phase_surrogate, axis = 0))
        surrogate_ranks = np.argsort(surrogate, axis = 0)
        rev_ranks = np.argsort(surrogate_ranks, axis = 0)
        
        surr_signal = np.zeros_like(y)
        for j in range(0,nnodes):
            surr_signal[:,j] = y_ord[rev_ranks[:,j],j]    
         
        FC_surr = np.corrcoef(surrogate.T)
        matrix_surr[:,i] = FC_surr[np.triu_indices(n = nnodes, k = 1)]
    
    p_vector = np.zeros(npairs)
    for i in range(npairs):
        mu, sigma = stats.norm.fit(matrix_surr[i,:])
        p_vector[i] = 1 - stats.norm.cdf(FC_real[i], mu, sigma)
    
    
    reject, p_adjust, alphacSidak, alphacBonf = multipletests(p_vector, alpha = alpha, method='fdr_bh')
    
    FC_adjusted = FC_real * ((p_adjust < alpha) * 1)
    FC_adjusted = matrix_recon(FC_adjusted)
    
    return(FC_adjusted)



def probabilistic_thresholding(y, surr_number = 50, alpha = 0.05, conn = 'corr'):

    y = y - np.mean(y, axis = 0)
    yf = np.fft.fft(y, axis = 0)

    nnodes = y.shape[1]
    npairs = (nnodes**2 - nnodes) // 2
    matrix_surr = np.zeros((npairs, surr_number))
    
    if conn == 'corr':
        FC_real = np.corrcoef(y.T)
#    elif conn == 'wpli':
#        FC_real = connectivity.PLI2(y.T, method = 'wpli')
    else:
        raise KeyError('invalid connectivity metric')
    FC_real = FC_real[np.triu_indices(n = nnodes, k = 1)]
    
    for i in range(surr_number):
        np.random.seed(i + 1)
        random_vector = np.random.uniform(0, 2 * np.pi, ((yf.shape[0] // 2), yf.shape[1]))
        random_vector = np.row_stack((random_vector, random_vector[::-1,:]))
        yfR = yf * np.exp(1j * random_vector)
        surrogate = np.fft.ifft(yfR, axis = 0)
        surrogate = surrogate.real
        
        if conn == 'corr':
            FC_surr = np.corrcoef(surrogate.T)
#        elif conn == 'wpli':
#            FC_surr = connectivity.PLI2(surrogate.T, method = 'wpli')
        else:
            raise KeyError('invalid connectivity metric')        
        matrix_surr[:,i] = FC_surr[np.triu_indices(n = nnodes, k = 1)]
    
    p_vector = np.zeros(npairs)
    for i in range(npairs):
        mu, sigma = stats.norm.fit(matrix_surr[i,:])
        p_vector[i] = 1 - stats.norm.cdf(FC_real[i], mu, sigma)
    
    
    reject, p_adjust, alphacSidak, alphacBonf = multipletests(p_vector, alpha = alpha, method='fdr_bh')
    
    FC_adjusted = FC_real * ((p_adjust < alpha) * 1)
    FC_adjusted = matrix_recon(FC_adjusted)
    p_matrix = matrix_recon(p_adjust)
    
    return([FC_adjusted, p_matrix])
def rare_dist(v1,v2):
    v1_n = v1/np.linalg.norm(v1)
    v2_n = v2/np.linalg.norm(v2)
    return np.linalg.norm(v1_n-v2_n)
    
def participation_coef(W, ci, degree = 'undirected'):
    """
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        binary/weighted directed/undirected connection matrix
    ci : Nx1 :obj:`numpy.ndarray`
        community affiliation vector
    degree : {'undirected', 'in', 'out'}, optional
        Flag to describe nature of graph. 'undirected': For undirected graphs,
        'in': Uses the in-degree, 'out': Uses the out-degree
    Returns
    -------
    P : Nx1 :obj:`numpy.ndarray`
        participation coefficient
    """
    if degree == 'in':
        W = W.T

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out) degree
    Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation
    Kc2 = np.zeros((n,))  # community-specific neighbors

    for i in range(1, int(np.max(ci)) + 1):
        Kc2 += np.square(np.sum(W * (Gc == i), axis=1))

    P = np.ones((n,)) - Kc2 / np.square(Ko)
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0

    return(P)
 
 
def ls2ci(ls, zeroindexed=False):
    """
    Convert from a 2D python list of modules to a community index vector.
    The list is a pure python list, not requiring numpy.
    Parameters
    ----------
    ls : listof(list)
        pure python list with lowest value zero-indexed
        (regardless of value of zeroindexed parameter)
    zeroindexed : bool
        If True, ci uses zero-indexing (lowest value is 0). Defaults to False.
    Returns
    -------
    ci : Nx1 :obj:`numpy.ndarray`
        community index vector
    """
    if ls is None or np.size(ls) == 0:
        return ()  # list is empty
    nr_indices = sum(map(len, ls))
    ci = np.zeros((nr_indices,), dtype=int)
    z = int(not zeroindexed)
    for i, x in enumerate(ls):
        for j, y in enumerate(ls[i]):
            ci[ls[i][j]] = i + z
    return ci
    
    
def modularity_dir(A, gamma=1, kci=None):
    """
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.
    Parameters
    ----------
    A : NxN :obj:`numpy.ndarray`
        directed weighted/binary connection matrix
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    kci : Nx1 :obj:`numpy.ndarray` | None
        starting community structure. If specified, calculates the Q-metric
        on the community structure giving, without doing any optimzation.
        Otherwise, if not specified, uses a spectral modularity maximization
        algorithm.
    Returns
    -------
    ci : Nx1 :obj:`numpy.ndarray`
        optimized community structure
    Q : float
        maximized modularity metric
    Notes
    -----
    This algorithm is deterministic. The matlab function bearing this
    name incorrectly disclaims that the outcome depends on heuristics
    involving a random seed. The louvain method does depend on a random seed,
    but this function uses a deterministic modularity maximization algorithm.
    """
    from scipy import linalg
    n = len(A)  # number of vertices
    ki = np.sum(A, axis=0)  # in degree
    ko = np.sum(A, axis=1)  # out degree
    m = np.sum(ki)  # number of edges
    b = A - gamma * np.outer(ko, ki) / m
    B = b + b.T  # directed modularity matrix

    init_mod = np.arange(n)  # initial one big module
    modules = []  # output modules list

    def recur(module):
        n = len(module)
        modmat = B[module][:, module]

        vals, vecs = linalg.eig(modmat)  # biggest eigendecomposition
        rlvals = np.real(vals)
        max_eigvec = np.squeeze(vecs[:, np.where(rlvals == np.max(rlvals))])
        if max_eigvec.ndim > 1:  # if multiple max eigenvalues, pick one
            max_eigvec = max_eigvec[:, 0]
        # initial module assignments
        mod_asgn = np.squeeze((max_eigvec >= 0) * 2 - 1)
        q = np.dot(mod_asgn, np.dot(modmat, mod_asgn))  # modularity change

        if q > 0:  # change in modularity was positive
            qmax = q
            np.fill_diagonal(modmat, 0)
            it = np.ma.masked_array(np.ones((n,)), False)
            mod_asgn_iter = mod_asgn.copy()
            while np.any(it):  # do some iterative fine tuning
                # this line is linear algebra voodoo
                q_iter = qmax - 4 * mod_asgn_iter * \
                    (np.dot(modmat, mod_asgn_iter))
                qmax = np.max(q_iter * it)
                imax = np.argmax(q_iter * it)
                # imax, = np.where(q_iter == qmax)
                # if len(imax) > 0:
                #     imax = imax[0]
                #     print(imax)
                # does switching increase modularity?
                mod_asgn_iter[imax] *= -1
                it[imax] = np.ma.masked
                if qmax > q:
                    q = qmax
                    mod_asgn = mod_asgn_iter
            if np.abs(np.sum(mod_asgn)) == n:  # iteration yielded null module
                modules.append(np.array(module).tolist())
            else:
                mod1 = module[np.where(mod_asgn == 1)]
                mod2 = module[np.where(mod_asgn == -1)]

                recur(mod1)
                recur(mod2)
        else:  # change in modularity was negative or 0
            modules.append(np.array(module).tolist())

    # adjustment to one-based indexing occurs in ls2ci
    if kci is None:
        recur(init_mod)
        ci = ls2ci(modules)
    else:
        ci = kci
    s = np.tile(ci, (n, 1))
    q = np.sum(np.logical_not(s - s.T) * B / (2 * m))
    return ci, q


def modularity_und(A, gamma=1, kci=None):
    """
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.
    Parameters
    ----------
    A : NxN :obj:`numpy.ndarray`
        undirected weighted/binary connection matrix
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    kci : Nx1 :obj:`numpy.ndarray` | None
        starting community structure. If specified, calculates the Q-metric
        on the community structure giving, without doing any optimzation.
        Otherwise, if not specified, uses a spectral modularity maximization
        algorithm.
    Returns
    -------
    ci : Nx1 :obj:`numpy.ndarray`
        optimized community structure
    Q : float
        maximized modularity metric
    Notes
    -----
    This algorithm is deterministic. The matlab function bearing this
    name incorrectly disclaims that the outcome depends on heuristics
    involving a random seed. The louvain method does depend on a random seed,
    but this function uses a deterministic modularity maximization algorithm.
    """
    from scipy import linalg
    n = len(A)  # number of vertices
    k = np.sum(A, axis=0)  # degree
    m = np.sum(k)  # number of edges (each undirected edge
    # is counted twice)
    B = A - gamma * np.outer(k, k) / m  # initial modularity matrix

    init_mod = np.arange(n)  # initial one big module
    modules = []  # output modules list

    def recur(module):
        n = len(module)
        modmat = B[module][:, module]
        modmat -= np.diag(np.sum(modmat, axis=0))

        vals, vecs = linalg.eigh(modmat)  # biggest eigendecomposition
        rlvals = np.real(vals)
        max_eigvec = np.squeeze(vecs[:, np.where(rlvals == np.max(rlvals))])
        if max_eigvec.ndim > 1:  # if multiple max eigenvalues, pick one
            max_eigvec = max_eigvec[:, 0]
        # initial module assignments
        mod_asgn = np.squeeze((max_eigvec >= 0) * 2 - 1)
        q = np.dot(mod_asgn, np.dot(modmat, mod_asgn))  # modularity change

        if q > 0:  # change in modularity was positive
            qmax = q
            np.fill_diagonal(modmat, 0)
            it = np.ma.masked_array(np.ones((n,)), False)
            mod_asgn_iter = mod_asgn.copy()
            while np.any(it):  # do some iterative fine tuning
                # this line is linear algebra voodoo
                q_iter = qmax - 4 * mod_asgn_iter * \
                    (np.dot(modmat, mod_asgn_iter))
                qmax = np.max(q_iter * it)
                imax = np.argmax(q_iter * it)
                # imax, = np.where(q_iter == qmax)
                # if len(imax) > 1:
                #     imax = imax[0]
                # does switching increase modularity?
                mod_asgn_iter[imax] *= -1
                it[imax] = np.ma.masked
                if qmax > q:
                    q = qmax
                    mod_asgn = mod_asgn_iter
            if np.abs(np.sum(mod_asgn)) == n:  # iteration yielded null module
                modules.append(np.array(module).tolist())
                return
            else:
                mod1 = module[np.where(mod_asgn == 1)]
                mod2 = module[np.where(mod_asgn == -1)]

                recur(mod1)
                recur(mod2)
        else:  # change in modularity was negative or 0
            modules.append(np.array(module).tolist())

    # adjustment to one-based indexing occurs in ls2ci
    if kci is None:
        recur(init_mod)
        ci = ls2ci(modules)
    else:
        ci = kci
    s = np.tile(ci, (n, 1))
    q = np.sum(np.logical_not(s - s.T) * B / m)
    return ci, q

    
def efficiency_bin(G, local=False):
    """
    The global efficiency is the average of inverse shortest path length,
    and is inversely related to the characteristic path length.
    The local efficiency is the global efficiency computed on the
    neighborhood of the node, and is related to the clustering coefficient.
    Parameters
    ----------
    G : NxN :obj:`numpy.ndarray`
        binary undirected connection matrix
    local : bool
        If True, computes local efficiency instead of global efficiency.
        Default value = False.
    Returns
    -------
    Eglob : float
        global efficiency, only if local=False
    Eloc : Nx1 :obj:`numpy.ndarray`
        local efficiency, only if local=True
    """
    def distance_inv(g):
        D = np.eye(len(g))
        n = 1
        nPATH = g.copy()
        L = (nPATH != 0)

        while np.any(L):
            D += n * L
            n += 1
            nPATH = np.dot(nPATH, g)
            L = (nPATH != 0) * (D == 0)
        D[np.logical_not(D)] = np.inf
        D = 1 / D
        np.fill_diagonal(D, 0)
        return D

    G = binarize(G)
    n = len(G)  # number of nodes
    if local:
        E = np.zeros((n,))  # local efficiency

        for u in range(n):
            # find pairs of neighbors
            V, = np.where(np.logical_or(G[u, :], G[u, :].T))
            # inverse distance matrix
            e = distance_inv(G[np.ix_(V, V)])
            # symmetrized inverse distance matrix
            se = e + e.T

            # symmetrized adjacency vector
            sa = G[u, V] + G[V, u].T
            numer = np.sum(np.outer(sa.T, sa) * se) / 2
            if numer != 0:
                denom = np.sum(sa)**2 - np.sum(sa * sa)
                E[u] = numer / denom  # local efficiency

    else:
        e = distance_inv(G)
        E = np.sum(e) / (n * n - n)  # global efficiency
    return E


def efficiency_wei(Gw, local = False):
    """
    The global efficiency is the average of inverse shortest path length,
    and is inversely related to the characteristic path length.
    The local efficiency is the global efficiency computed on the
    neighborhood of the node, and is related to the clustering coefficient.
    Parameters
    ----------
    Gw : NxN :obj:`numpy.ndarray`
        undirected weighted connection matrix
        (all weights in W must be between 0 and 1)
    local : bool
        If True, computes local efficiency instead of global efficiency.
        Default value = False.
    Returns
    -------
    Eglob : float
        global efficiency, only if local=False
    Eloc : Nx1 :obj:`numpy.ndarray`
        local efficiency, only if local=True
    Notes
    -----
       The  efficiency is computed using an auxiliary connection-length
    matrix L, defined as L_ij = 1/W_ij for all nonzero L_ij; This has an
    intuitive interpretation, as higher connection weights intuitively
    correspond to shorter lengths.
       The weighted local efficiency broadly parallels the weighted
    clustering coefficient of Onnela et al. (2005) and distinguishes the
    influence of different paths based on connection weights of the
    corresponding neighbors to the node in question. In other words, a path
    between two neighbors with strong connections to the node in question
    contributes more to the local efficiency than a path between two weakly
    connected neighbors. Note that this weighted variant of the local
    efficiency is hence not a strict generalization of the binary variant.
    Algorithm:  Dijkstra's algorithm
    """
    def distance_inv_wei(G):
        n = len(G)
        D = np.zeros((n, n))  # distance matrix
        D[np.logical_not(np.eye(n))] = np.inf

        for u in range(n):
            # distance permanence (true is temporary)
            S = np.ones((n,), dtype=bool)
            G1 = G.copy()
            V = [u]
            while True:
                S[V] = 0  # distance u->V is now permanent
                G1[:, V] = 0  # no in-edges as already shortest
                for v in V:
                    W, = np.where(G1[v, :])  # neighbors of smallest nodes
                    td = np.array(
                        [D[u, W].flatten(), (D[u, v] + G1[v, W]).flatten()])
                    D[u, W] = np.min(td, axis=0)

                if D[u, S].size == 0:  # all nodes reached
                    break
                minD = np.min(D[u, S])
                if np.isinf(minD):  # some nodes cannot be reached
                    break
                V, = np.where(D[u, :] == minD)

        np.fill_diagonal(D, 1)
        D = 1 / D
        np.fill_diagonal(D, 0)
        return D

    n = len(Gw)
    Gl = invert(Gw, copy=True)  # connection length matrix
    A = np.array((Gw != 0), dtype=int)
    if local:
        E = np.zeros((n,))  # local efficiency
        for u in range(n):
            # find pairs of neighbors
            V, = np.where(np.logical_or(Gw[u, :], Gw[:, u].T))
            # symmetrized vector of weights
            sw = cuberoot(Gw[u, V]) + cuberoot(Gw[V, u].T)
            # inverse distance matrix
            e = distance_inv_wei(Gl[np.ix_(V, V)])
            # symmetrized inverse distance matrix
            se = cuberoot(e) + cuberoot(e.T)

            numer = np.sum(np.outer(sw.T, sw) * se) / 2
            if numer != 0:
                # symmetrized adjacency vector
                sa = A[u, V] + A[V, u].T
                denom = np.sum(sa)**2 - np.sum(sa * sa)
                # print numer,denom
                E[u] = numer / denom  # local efficiency

    else:
        e = distance_inv_wei(Gl)
        E = np.sum(e) / (n * n - n)
    return(E)   

  
class BCTParamError(RuntimeError):
    pass
   
    
def modularity_louvain_und_sign(W, gamma=1, qtype='sta', seed=None):
    """
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes in a way that maximizes the number of
    within-group edges, and minimizes the number of between-group edges.
    The modularity is a statistic that quantifies the degree to which the
    network may be subdivided into such clearly delineated groups.
    The Louvain algorithm is a fast and accurate community detection
    algorithm (at the time of writing).
    Use this function as opposed to modularity_louvain_und() only if the
    network contains a mix of positive and negative weights.  If the network
    contains all positive weights, the output will be equivalent to that of
    modularity_louvain_und().
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        undirected weighted/binary connection matrix with positive and
        negative weights
    qtype : str
        modularity type. Can be 'sta' (default), 'pos', 'smp', 'gja', 'neg'.
        See Rubinov and Sporns (2011) for a description.
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.
    Returns
    -------
    ci : Nx1 :obj:`numpy.ndarray`
        refined community affiliation vector
    Q : float
        optimized modularity metric
    Notes
    -----
    Ci and Q may vary from run to run, due to heuristics in the
    algorithm. Consequently, it may be worth to compare multiple runs.
    """
    np.random.seed(seed)

    n = len(W)  # number of nodes

    W0 = W * (W > 0)  # positive weights matrix
    W1 = -W * (W < 0)  # negative weights matrix
    s0 = np.sum(W0)  # weight of positive links
    s1 = np.sum(W1)  # weight of negative links

    if qtype == 'smp':
        d0 = 1 / s0
        d1 = 1 / s1  # dQ=dQ0/s0-sQ1/s1
    elif qtype == 'gja':
        d0 = 1 / (s0 + s1)
        d1 = d0  # dQ=(dQ0-dQ1)/(s0+s1)
    elif qtype == 'sta':
        d0 = 1 / s0
        d1 = 1 / (s0 + s1)  # dQ=dQ0/s0-dQ1/(s0+s1)
    elif qtype == 'pos':
        d0 = 1 / s0
        d1 = 0  # dQ=dQ0/s0
    elif qtype == 'neg':
        d0 = 0
        d1 = 1 / s1  # dQ=-dQ1/s1
    else:
        raise KeyError('modularity type unknown')

    if not s0:  # adjust for absent positive weights
        s0 = 1
        d0 = 0
    if not s1:  # adjust for absent negative weights
        s1 = 1
        d1 = 0

    h = 1  # hierarchy index
    nh = n  # number of nodes in hierarchy
    ci = [None, np.arange(n) + 1]  # hierarchical module assignments
    q = [-1, 0]  # hierarchical modularity values
    while q[h] - q[h - 1] > 1e-10:
        if h > 300:
            raise BCTParamError('Modularity Infinite Loop Style A.  Please '
                                'contact the developer with this error.')
        kn0 = np.sum(W0, axis=0)  # positive node degree
        kn1 = np.sum(W1, axis=0)  # negative node degree
        km0 = kn0.copy()  # positive module degree
        km1 = kn1.copy()  # negative module degree
        knm0 = W0.copy()  # positive node-to-module degree
        knm1 = W1.copy()  # negative node-to-module degree

        m = np.arange(nh) + 1  # initial module assignments
        flag = True  # flag for within hierarchy search
        it = 0
        while flag:
            it += 1
            if it > 1000:
                raise BCTParamError('Infinite Loop was detected and stopped. '
                                    'This was probably caused by passing in a '
                                    'directed matrix.')
            flag = False
            # loop over nodes in random order
            for u in np.random.permutation(nh):
                ma = m[u] - 1
                # positive dQ
                dQ0 = ((knm0[u, :] + W0[u, u] - knm0[u, ma]) -
                       gamma * kn0[u] * (km0 + kn0[u] - km0[ma]) / s0)
                # negative dQ
                dQ1 = ((knm1[u, :] + W1[u, u] - knm1[u, ma]) -
                       gamma * kn1[u] * (km1 + kn1[u] - km1[ma]) / s1)

                dQ = d0 * dQ0 - d1 * dQ1  # rescaled changes in modularity
                dQ[ma] = 0  # no changes for same module

                max_dQ = np.max(dQ)  # maximal increase in modularity
                if max_dQ > 1e-10:  # if maximal increase is positive
                    flag = True
                    mb = np.argmax(dQ)

                    # change positive node-to-module degrees
                    knm0[:, mb] += W0[:, u]
                    knm0[:, ma] -= W0[:, u]
                    # change negative node-to-module degrees
                    knm1[:, mb] += W1[:, u]
                    knm1[:, ma] -= W1[:, u]
                    km0[mb] += kn0[u]  # change positive module degrees
                    km0[ma] -= kn0[u]
                    km1[mb] += kn1[u]  # change negative module degrees
                    km1[ma] -= kn1[u]

                    m[u] = mb + 1  # reassign module

        h += 1
        ci.append(np.zeros((n,)))
        _, m = np.unique(m, return_inverse=True)
        m += 1

        for u in range(nh):  # loop through initial module assignments
            ci[h][np.where(ci[h - 1] == u + 1)] = m[u]  # assign new modules

        nh = np.max(m)  # number of new nodes
        wn0 = np.zeros((nh, nh))  # new positive weights matrix
        wn1 = np.zeros((nh, nh))

        for u in range(nh):
            for v in range(u, nh):
                wn0[u, v] = np.sum(W0[np.ix_(m == u + 1, m == v + 1)])
                wn1[u, v] = np.sum(W1[np.ix_(m == u + 1, m == v + 1)])
                wn0[v, u] = wn0[u, v]
                wn1[v, u] = wn1[u, v]

        W0 = wn0
        W1 = wn1

        q.append(0)
        # compute modularity
        q0 = np.trace(W0) - np.sum(np.dot(W0, W0)) / s0
        q1 = np.trace(W1) - np.sum(np.dot(W1, W1)) / s1
        q[h] = d0 * q0 - d1 * q1

    _, ci_ret = np.unique(ci[-1], return_inverse=True)
    ci_ret += 1

    return ci_ret, q[-1]
    

def dummyvar(cis, return_sparse=False):
    """
    This is an efficient implementation of matlab's "dummyvar" command
    using sparse matrices.
    input: partitions, NxM array-like containing M partitions of N nodes
        into <=N distinct communities
    output: dummyvar, an NxR matrix containing R column variables (indicator
        variables) with N entries, where R is the total number of communities
        summed across each of the M partitions.
        i.e.
        r = sum((max(len(unique(partitions[i]))) for i in range(m)))
    """
    # num_rows is not affected by partition indexes
    n = np.size(cis, axis=0)
    m = np.size(cis, axis=1)
    r = np.sum((np.max(len(np.unique(cis[:, i])))) for i in range(m))
    nnz = np.prod(cis.shape)

    ix = np.argsort(cis, axis=0)
    # s_cis=np.sort(cis,axis=0)
    # FIXME use the sorted indices to sort by row efficiently
    s_cis = cis[ix][:, range(m), range(m)]

    mask = np.hstack((((True,),) * m, (s_cis[:-1, :] != s_cis[1:, :]).T))
    indptr, = np.where(mask.flat)
    indptr = np.append(indptr, nnz)

    import scipy.sparse as sp
    dv = sp.csc_matrix((np.repeat((1,), nnz), ix.T.flat, indptr), shape=(n, r))
    return dv.toarray()


def agreement(ci, buffsz=1000):
    """
    Takes as input a set of vertex partitions CI of
    dimensions [vertex x partition]. Each column in CI contains the
    assignments of each vertex to a class/community/module. This function
    aggregates the partitions in CI into a square [vertex x vertex]
    agreement matrix D, whose elements indicate the number of times any two
    vertices were assigned to the same class.
    In the case that the number of nodes and partitions in CI is large
    (greater than ~1000 nodes or greater than ~1000 partitions), the script
    can be made faster by computing D in pieces. The optional input BUFFSZ
    determines the size of each piece. Trial and error has found that
    BUFFSZ ~ 150 works well.
    Parameters
    ----------
    ci : NxM :obj:`numpy.ndarray`
        set of M (possibly degenerate) partitions of N nodes
    buffsz : int | None
        sets buffer size. If not specified, defaults to 1000
    Returns
    -------
    D : NxN :obj:`numpy.ndarray`
        agreement matrix
    """
    ci = np.array(ci)
    n_nodes, n_partitions = ci.shape

    if n_partitions <= buffsz:  # Case 1: Use all partitions at once
        ind = dummyvar(ci)
        D = np.dot(ind, ind.T)
    else:  # Case 2: Add together results from subsets of partitions
        a = np.arange(0, n_partitions, buffsz)
        b = np.arange(buffsz, n_partitions, buffsz)
        if len(a) != len(b):
            b = np.append(b, n_partitions)
        D = np.zeros((n_nodes, n_nodes))
        for i, j in zip(a, b):
            y = ci[:, i:j]
            ind = dummyvar(y)
            D += np.dot(ind, ind.T)

    np.fill_diagonal(D, 0)
    return D


def get_agreement_matrix(W, reps = 100, tau = 0.5, gamma = 1, ci = None, B = 'modularity', seeds = None):
    nnodes = W.shape[0]
    n_p = reps #number of partitions
    D_matrix = np.zeros((nnodes,nnodes)) #Agreement matrix
    
    if seeds == None:
        seed_vector = np.arange(0, n_p, 1)
    elif len(seed_vector) == n_p:
        seed_vector = seeds
    else:
        print('seeds must be an array of length equal to reps, or None')
    
    for rand in range(0,n_p):
        partition = community_louvain(W, gamma, ci, B, seed_vector[rand])[0]
        for row in range(0,nnodes):
            for col in range(0,nnodes):
                D_matrix[row,col] += (partition[row] == partition[col]) * 1
    D_matrix /= n_p
    D_matrix[D_matrix < tau] = 0        
    np.fill_diagonal(D_matrix, 0)
    
    return(D_matrix)

def community_louvain(W, gamma= 1, ci=None, B='modularity', seed=None):
    """
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes which maximizes the number of within-group
    edges and minimizes the number of between-group edges.
    This function is a fast an accurate multi-iterative generalization of the
    louvain community detection algorithm. This function subsumes and improves
    upon modularity_[louvain,finetune]_[und,dir]() and additionally allows to
    optimize other objective functions (includes built-in Potts Model i
    Hamiltonian, allows for custom objective-function matrices).
    Parameters
    ----------
    W : NxN np.array
        directed/undirected weighted/binary adjacency matrix
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
        ignored if an objective function matrix is specified.
    ci : Nx1 np.arraylike
        initial community affiliation vector. default value=None
    B : str | NxN np.arraylike
        string describing objective function type, or provides a custom
        NxN objective-function matrix. builtin values
            'modularity' uses Q-metric as objective function
            'potts' uses Potts model Hamiltonian.
            'negative_sym' symmetric treatment of negative weights
            'negative_asym' asymmetric treatment of negative weights
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.
    Returns
    -------
    ci : Nx1 np.array
        final community structure
    q : float
        optimized q-statistic (modularity only)
    """
    np.random.seed(seed)

    n = len(W)
    s = np.sum(W)

    if np.min(W) < -1e-10:
        raise BCTParamError('adjmat must not contain negative weights')

    if ci is None:
        ci = np.arange(n) + 1
    else:
        if len(ci) != n:
            raise BCTParamError('initial ci vector size must equal N')
        _, ci = np.unique(ci, return_inverse=True)
        ci += 1
    Mb = ci.copy()

    if B in ('negative_sym', 'negative_asym'):
        W0 = W * (W > 0)
        s0 = np.sum(W0)
        B0 = W0 - gamma * np.outer(np.sum(W0, axis=1), np.sum(W, axis=0)) / s0

        W1 = W * (W < 0)
        s1 = np.sum(W1)
        if s1:
            B1 = (W1 - gamma * np.outer(np.sum(W1, axis=1), np.sum(W1, axis=0))
                  / s1)
        else:
            B1 = 0

    elif np.min(W) < -1e-10:
        raise BCTParamError('Input connection matrix contains negative '
                            'weights but objective function dealing with '
                            'negative weights was not selected')

    if B == 'potts' and np.any(np.logical_not(np.logical_or(W == 0, W == 1))):
        raise BCTParamError('Potts hamiltonian requires binary input matrix')

    if B == 'modularity':
        B = W - gamma * np.outer(np.sum(W, axis=1), np.sum(W, axis=0)) / s
    elif B == 'potts':
        B = W - gamma * np.logical_not(W)
    elif B == 'negative_sym':
        B = B0 / (s0 + s1) - B1 / (s0 + s1)
    elif B == 'negative_asym':
        B = B0 / s0 - B1 / (s0 + s1)
    else:
        try:
            B = np.array(B)
        except BCTParamError:
            print('unknown objective function type')

        if B.shape != W.shape:
            raise BCTParamError('objective function matrix does not match '
                                'size of adjacency matrix')
        if not np.allclose(B, B.T):
            print('Warning: objective function matrix not symmetric, '
                  'symmetrizing')
            B = (B + B.T) / 2

    Hnm = np.zeros((n, n))
    for m in range(1, n + 1):
        Hnm[:, m - 1] = np.sum(B[:, ci == m], axis=1)  # node to module degree
    H = np.sum(Hnm, axis=1)  # node degree
    Hm = np.sum(Hnm, axis=0)  # module degree

    q0 = -np.inf
    # compute modularity
    q = np.sum(B[np.tile(ci, (n, 1)) == np.tile(ci, (n, 1)).T]) / s

    first_iteration = True

    while q - q0 > 1e-10:
        it = 0
        flag = True
        while flag:
            it += 1
            if it > 1000:
                raise BCTParamError('Modularity infinite loop style G. '
                                    'Please contact the developer.')
            flag = False
            for u in np.random.permutation(n):
                ma = Mb[u] - 1
                dQ = Hnm[u, :] - Hnm[u, ma] + B[u, u]  # algorithm condition
                dQ[ma] = 0

                max_dq = np.max(dQ)
                if max_dq > 1e-10:
                    flag = True
                    mb = np.argmax(dQ)

                    Hnm[:, mb] += B[:, u]
                    Hnm[:, ma] -= B[:, u]  # change node-to-module strengths

                    Hm[mb] += H[u]
                    Hm[ma] -= H[u]  # change module strengths

                    Mb[u] = mb + 1

        _, Mb = np.unique(Mb, return_inverse=True)
        Mb += 1

        M0 = ci.copy()
        if first_iteration:
            ci = Mb.copy()
            first_iteration = False
        else:
            for u in range(1, n + 1):
                ci[M0 == u] = Mb[u - 1]  # assign new modules

        n = np.max(Mb)
        b1 = np.zeros((n, n))
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                # pool weights of nodes in same module
                bm = np.sum(B[np.ix_(Mb == i, Mb == j)])
                b1[i - 1, j - 1] = bm
                b1[j - 1, i - 1] = bm
        B = b1.copy()

        Mb = np.arange(1, n + 1)
        Hnm = B.copy()
        H = np.sum(B, axis=0)
        Hm = H.copy()

        q0 = q
        q = np.trace(B) / s  # compute modularity

    return ci, q

def consensus_und(D, tau, reps = 1000):
    """
    This algorithm seeks a consensus partition of the
    agreement matrix D. The algorithm used here is almost identical to the
    one introduced in Lancichinetti & Fortunato (2012): The agreement
    matrix D is thresholded at a level TAU to remove an weak elements. The
    resulting matrix is then partitions REPS number of times using the
    Louvain algorithm (in principle, any clustering algorithm that can
    handle weighted matrixes is a suitable alternative to the Louvain
    algorithm and can be substituted in its place). This clustering
    produces a set of partitions from which a new agreement is built. If
    the partitions have not converged to a single representative partition,
    the above process repeats itself, starting with the newly built
    agreement matrix.
    NOTE: In this implementation, the elements of the agreement matrix must
    be converted into probabilities.
    NOTE: This implementation is slightly different from the original
    algorithm proposed by Lanchichinetti & Fortunato. In its original
    version, if the thresholding produces singleton communities, those
    nodes are reconnected to the network. Here, we leave any singleton
    communities disconnected.
    Parameters
    ----------
    D : NxN :obj:`numpy.ndarray`
        agreement matrix with entries between 0 and 1 denoting the probability
        of finding node i in the same cluster as node j
    tau : float
        threshold which controls the resolution of the reclustering
    reps : int
        number of times the clustering algorithm is reapplied. default value
        is 1000.
    Returns
    -------
    ciu : Nx1 :obj:`numpy.ndarray`
        consensus partition
    """
    def unique_partitions(cis):
        # relabels the partitions to recognize different numbers on same
        # topology

        n, r = np.shape(cis)  # ci represents one vector for each rep
        ci_tmp = np.zeros(n)

        for i in range(r):
            for j, u in enumerate(sorted(
                    np.unique(cis[:, i], return_index=True)[1])):
                ci_tmp[np.where(cis[:, i] == cis[u, i])] = j
            cis[:, i] = ci_tmp
            # so far no partitions have been deleted from ci

        # now squash any of the partitions that are completely identical
        # do not delete them from ci which needs to stay same size, so make
        # copy
        ciu = []
        cis = cis.copy()
        c = np.arange(r)
        # count=0
        while (c != 0).sum() > 0:
            ciu.append(cis[:, 0])
            dup = np.where(np.sum(np.abs(cis.T - cis[:, 0]), axis=1) == 0)
            cis = np.delete(cis, dup, axis=1)
            c = np.delete(c, dup)
        return np.transpose(ciu)

    n = len(D)
    flag = True
    while flag:
        flag = False
        dt = D * (D >= tau)
        np.fill_diagonal(dt, 0)

        if np.size(np.where(dt == 0)) == 0:
            ciu = np.arange(1, n + 1)
        else:
            cis = np.zeros((n, reps))
            for i in np.arange(reps):
                cis[:, i], _ = modularity_louvain_und_sign(dt)
            ciu = unique_partitions(cis)
            nu = np.size(ciu, axis=1)
            if nu > 1:
                flag = True
                D = agreement(cis) / reps

    return np.squeeze(ciu + 1)  
    

def clustering_coef_bd(A):
    """
    The clustering coefficient is the fraction of triangles around a node
    (equiv. the fraction of nodes neighbors that are neighbors of each other).
    Parameters
    ----------
    A : NxN :obj:`numpy.ndarray`
        binary directed connection matrix
    Returns
    -------
    C : Nx1 :obj:`numpy.ndarray`
        clustering coefficient vector
    Notes
    -----
    Methodological note: In directed graphs, 3 nodes generate up to 8
    triangles (2*2*2 edges). The number of existing triangles is the main
    diagonal of S^3/2. The number of all (in or out) neighbour pairs is
    K(K-1)/2. Each neighbour pair may generate two triangles. "False pairs"
    are i<->j edge pairs (these do not generate triangles). The number of
    false pairs is the main diagonal of A^2.
    Thus the maximum possible number of triangles =
           = (2 edges)*([ALL PAIRS] - [FALSE PAIRS])
           = 2 * (K(K-1)/2 - diag(A^2))
           = K(K-1) - 2(diag(A^2))
    """
    S = A + A.T  # symmetrized input graph
    K = np.sum(S, axis=1)  # total degree (in+out) 
    cyc3 = np.diag(np.dot(S, np.dot(S, S))) / 2  # number of 3-cycles
    if np.sum(((cyc3 == 0) * 1)) > 0:
        K[np.where(cyc3 == 0)] = np.inf  # if no 3-cycles exist, make C=0
    # number of all possible 3 cycles
    CYC3 = K * (K - 1) - 2 * np.diag(np.dot(A, A))
    C = cyc3 / CYC3
    return C


def clustering_coef_bu(G):
    """
    The clustering coefficient is the fraction of triangles around a node
    (equiv. the fraction of nodes neighbors that are neighbors of each other).
    Parameters
    ----------
    G : NxN :obj:`numpy.ndarray`
        binary undirected connection matrix
    Returns
    -------
    C : Nx1 :obj:`numpy.ndarray`
        clustering coefficient vector
    """
    n = len(G)
    C = np.zeros((n,))

    for u in range(n):
        V, = np.where(G[u, :])
        k = len(V)
        if k >= 2:  # degree must be at least 2
            S = G[np.ix_(V, V)]
            C[u] = np.sum(S) / (k * k - k)

    return C


def clustering_coef_wd(W):
    """
    The weighted clustering coefficient is the average "intensity" of
    triangles around a node.
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        weighted directed connection matrix
    Returns
    -------
    C : Nx1 :obj:`numpy.ndarray`
        clustering coefficient vector
    Notes
    -----
    Methodological note (also see clustering_coef_bd)
    The weighted modification is as follows:
    - The numerator: adjacency matrix is replaced with weights matrix ^ 1/3
    - The denominator: no changes from the binary version
    The above reduces to symmetric and/or binary versions of the clustering
    coefficient for respective graphs.
    """
    A = np.logical_not(W == 0).astype(float)  # adjacency matrix
    S = cuberoot(W) + cuberoot(W.T)  # symmetrized weights matrix ^1/3
    K = np.sum(A + A.T, axis=1)  # total degree (in+out)
    cyc3 = np.diag(np.dot(S, np.dot(S, S))) / 2  # number of 3-cycles
    if np.sum(((cyc3 == 0) * 1)) > 0:    
        K[np.where(cyc3 == 0)] = np.inf  # if no 3-cycles exist, make C=0
    # number of all possible 3 cycles
    CYC3 = K * (K - 1) - 2 * np.diag(np.dot(A, A))
    C = cyc3 / CYC3  # clustering coefficient
    return C


def clustering_coef_wu(W):
    """
    The weighted clustering coefficient is the average "intensity" of
    triangles around a node.
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        weighted undirected connection matrix
    Returns
    -------
    C : Nx1 :obj:`numpy.ndarray`
        clustering coefficient vector
    """
    K = np.array(np.sum(np.logical_not(W == 0), axis=1), dtype=float)
    ws = cuberoot(W)
    cyc3 = np.diag(np.dot(ws, np.dot(ws, ws)))
    K[np.where(cyc3 == 0)] = np.inf  # if no 3-cycles exist, set C=0
    C = cyc3 / (K * (K - 1))
    return C


def clustering_coef_wu_sign(W, coef_type='default'):
    """
    Returns the weighted clustering coefficient generalized or separated
    for positive and negative weights.
    Three Algorithms are supported; herefore referred to as default, zhang,
    and constantini.
    1. Default (Onnela et al.), as in the traditional clustering coefficient
       computation. Computed separately for positive and negative weights.
    2. Zhang & Horvath. Similar to Onnela formula except weight information
       incorporated in denominator. Reduces sensitivity of the measure to
       weights directly connected to the node of interest. Computed
       separately for positive and negative weights.
    3. Constantini & Perugini generalization of Zhang & Horvath formula.
       Takes both positive and negative weights into account simultaneously.
       Particularly sensitive to non-redundancy in path information based on
       sign. Returns only one value.
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        weighted undirected connection matrix
    corr_type : {'default', 'zhang', 'constantini'}
        Allowed values are 'default', 'zhang', 'constantini'
    Returns
    -------
    Cpos : Nx1 :obj:`numpy.ndarray`
        Clustering coefficient vector for positive weights
    Cneg : Nx1 :obj:`numpy.ndarray`
        Clustering coefficient vector for negative weights, unless
        coef_type == 'constantini'.
    References:
        Onnela et al. (2005) Phys Rev E 71:065103
        Zhang & Horvath (2005) Stat Appl Genet Mol Biol 41:1544-6115
        Costantini & Perugini (2014) PLOS ONE 9:e88669
    """
    n = len(W)
    np.fill_diagonal(W, 0)

    if coef_type == 'default':
        W_pos = W * (W > 0)
        K_pos = np.array(np.sum(np.logical_not(W_pos == 0), axis=1),
                         dtype=float)
        ws_pos = cuberoot(W_pos)
        cyc3_pos = np.diag(np.dot(ws_pos, np.dot(ws_pos, ws_pos)))
        K_pos[np.where(cyc3_pos == 0)] = np.inf
        C_pos = cyc3_pos / (K_pos * (K_pos - 1))

        W_neg = -W * (W < 0)
        K_neg = np.array(np.sum(np.logical_not(W_neg == 0), axis=1),
                         dtype=float)
        ws_neg = cuberoot(W_neg)
        cyc3_neg = np.diag(np.dot(ws_neg, np.dot(ws_neg, ws_neg)))
        K_neg[np.where(cyc3_neg == 0)] = np.inf
        C_neg = cyc3_neg / (K_neg * (K_neg - 1))

        return C_pos, C_neg

    elif coef_type in ('zhang', 'Zhang'):
        W_pos = W * (W > 0)
        cyc3_pos = np.zeros((n,))
        cyc2_pos = np.zeros((n,))

        W_neg = -W * (W < 0)
        cyc3_neg = np.zeros((n,))
        cyc2_neg = np.zeros((n,))

        for i in range(n):
            for j in range(n):
                for q in range(n):
                    cyc3_pos[i] += W_pos[j, i] * W_pos[i, q] * W_pos[j, q]
                    cyc3_neg[i] += W_neg[j, i] * W_neg[i, q] * W_neg[j, q]
                    if j != q:
                        cyc2_pos[i] += W_pos[j, i] * W_pos[i, q]
                        cyc2_neg[i] += W_neg[j, i] * W_neg[i, q]

        cyc2_pos[np.where(cyc3_pos == 0)] = np.inf
        C_pos = cyc3_pos / cyc2_pos

        cyc2_neg[np.where(cyc3_neg == 0)] = np.inf
        C_neg = cyc3_neg / cyc2_neg

        return C_pos, C_neg

    elif coef_type in ('constantini', 'Constantini'):
        cyc3 = np.zeros((n,))
        cyc2 = np.zeros((n,))

        for i in range(n):
            for j in range(n):
                for q in range(n):
                    cyc3[i] += W[j, i] * W[i, q] * W[j, q]
                    if j != q:
                        cyc2[i] += W[j, i] * W[i, q]

        cyc2[np.where(cyc3 == 0)] = np.inf
        C = cyc3 / cyc2
        return C

 
    
def distance_bin(G):
    """
    The distance matrix contains lengths of shortest paths between all
    pairs of nodes. An entry (u,v) represents the length of shortest path
    from node u to node v. The average shortest path length is the
    characteristic path length of the network.
    Parameters
    ----------
    G : NxN :obj:`numpy.ndarray`
        binary directed/undirected connection matrix
    Returns
    -------
    D : NxN
        distance matrix
    Notes
    -----
    Lengths between disconnected nodes are set to Inf.
    Lengths on the main diagonal are set to 0.
    Algorithm: Algebraic shortest paths.
    """
    G = binarize(G, copy=True)
    D = np.eye(len(G))
    n = 1
    nPATH = G.copy()  # n path matrix
    L = (nPATH != 0)  # shortest n-path matrix

    while np.any(L):
        D += n * L
        n += 1
        nPATH = np.dot(nPATH, G)
        L = (nPATH != 0) * (D == 0)

    D[D == 0] = np.inf  # disconnected nodes are assigned d=inf
    np.fill_diagonal(D, 0)
    return D


def distance_wei(G):
    """
    The distance matrix contains lengths of shortest paths between all
    pairs of nodes. An entry (u,v) represents the length of shortest path
    from node u to node v. The average shortest path length is the
    characteristic path length of the network.
    Parameters
    ----------
    G : NxN :obj:`numpy.ndarray`
        Directed/undirected connection-length matrix.
        NB L is not the adjacency matrix. See below.
    Returns
    -------
    D : NxN :obj:`numpy.ndarray`
        distance (shortest weighted path) matrix
    B : NxN :obj:`numpy.ndarray`
        matrix of number of edges in shortest weighted path
    Notes
    -----
       The input matrix must be a connection-length matrix, typically
    obtained via a mapping from weight to length. For instance, in a
    weighted correlation network higher correlations are more naturally
    interpreted as shorter distances and the input matrix should
    consequently be some inverse of the connectivity matrix.
       The number of edges in shortest weighted paths may in general
    exceed the number of edges in shortest binary paths (i.e. shortest
    paths computed on the binarized connectivity matrix), because shortest
    weighted paths have the minimal weighted distance, but not necessarily
    the minimal number of edges.
       Lengths between disconnected nodes are set to Inf.
       Lengths on the main diagonal are set to 0.
    Algorithm: Dijkstra's algorithm.
    """
    n = len(G)
    D = np.zeros((n, n))  # distance matrix
    D[np.logical_not(np.eye(n))] = np.inf
    B = np.zeros((n, n))  # number of edges matrix

    for u in range(n):
        # distance permanence (true is temporary)
        S = np.ones((n,), dtype=bool)
        G1 = G.copy()
        V = [u]
        while True:
            S[V] = 0  # distance u->V is now permanent
            G1[:, V] = 0  # no in-edges as already shortest
            for v in V:
                W, = np.where(G1[v, :])  # neighbors of shortest nodes

                td = np.array(
                    [D[u, W].flatten(), (D[u, v] + G1[v, W]).flatten()])
                d = np.min(td, axis=0)
                wi = np.argmin(td, axis=0)

                D[u, W] = d  # smallest of old/new path lengths
                ind = W[np.where(wi == 1)]  # indices of lengthened paths
                # increment nr_edges for lengthened paths
                B[u, ind] = B[u, v] + 1

            if D[u, S].size == 0:  # all nodes reached
                break
            minD = np.min(D[u, S])
            if np.isinf(minD):  # some nodes cannot be reached
                break

            V, = np.where(D[u, :] == minD)

    return D, B
    


def charpath(D, include_diagonal=False, include_infinite=True):
    """
    The characteristic path length is the average shortest path length in
    the network. The global efficiency is the average inverse shortest path
    length in the network.
    Parameters
    ----------
    D : NxN :obj:`numpy.ndarray`
        distance matrix
    include_diagonal : bool
        If True, include the weights on the diagonal. Default value is False.
    include_infinite : bool
        If True, include infinite distances in calculation
    Returns
    -------
    lambda : float
        characteristic path length
    efficiency : float
        global efficiency
    ecc : Nx1 :obj:`numpy.ndarray`
        eccentricity at each vertex
    radius : float
        radius of graph
    diameter : float
        diameter of graph
    Notes
    -----
    The input distance matrix may be obtained with any of the distance
    functions, e.g. distance_bin, distance_wei.
    Characteristic path length is calculated as the global mean of
    the distance matrix D, excludings any 'Infs' but including distances on
    the main diagonal.
    """
    D = D.copy()

    if not include_diagonal:
        np.fill_diagonal(D, np.nan)

    if not include_infinite:
        D[np.isinf(D)] = np.nan

    Dv = D[np.logical_not(np.isnan(D))].ravel()

    # mean of finite entries of D[G]
    lambda_ = np.mean(Dv)

    # efficiency: mean of inverse entries of D[G]
    efficiency = np.mean(1 / Dv)

    # eccentricity for each vertex (ignore inf)
    ecc = np.array(np.ma.masked_where(np.isnan(D), D).max(axis=1))

    # radius of graph
    radius = np.min(ecc)  # but what about zeros?

    # diameter of graph
    diameter = np.max(ecc)

    return lambda_, efficiency, ecc, radius, diameter    
  

def transitivity_bd(A):
    """
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).
    Parameters
    ----------
    A : NxN :obj:`numpy.ndarray`
        binary directed connection matrix
    Returns
    -------
    T : float
        transitivity scalar
    Notes
    -----
    Methodological note: In directed graphs, 3 nodes generate up to 8
    triangles (2*2*2 edges). The number of existing triangles is the main
    diagonal of S^3/2. The number of all (in or out) neighbour pairs is
    K(K-1)/2. Each neighbour pair may generate two triangles. "False pairs"
    are i<->j edge pairs (these do not generate triangles). The number of
    false pairs is the main diagonal of A^2. Thus the maximum possible
    number of triangles = (2 edges)*([ALL PAIRS] - [FALSE PAIRS])
                        = 2 * (K(K-1)/2 - diag(A^2))
                        = K(K-1) - 2(diag(A^2))
    """
    S = A + A.T  # symmetrized input graph
    K = np.sum(S, axis=1)  # total degree (in+out)
    cyc3 = np.diag(np.dot(S, np.dot(S, S))) / 2  # number of 3-cycles
    CYC3 = K * (K - 1) - 2 * np.diag(np.dot(A, A))  # number of all 3-cycles
    return np.sum(cyc3) / np.sum(CYC3)


def transitivity_bu(A):
    """
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).
    Parameters
    ----------
    A : NxN :obj:`numpy.ndarray`
        binary undirected connection matrix
    Returns
    -------
    T : float
        transitivity scalar
    """
    tri3 = np.trace(np.dot(A, np.dot(A, A)))
    tri2 = np.sum(np.dot(A, A)) - np.trace(np.dot(A, A))
    return tri3 / tri2


def transitivity_wd(W):
    """
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        weighted directed connection matrix
    Returns
    -------
    T : int
        transitivity scalar
    Methodological note (also see note for clustering_coef_bd)
    The weighted modification is as follows:
    - The numerator: adjacency matrix is replaced with weights matrix ^ 1/3
    - The denominator: no changes from the binary version
    The above reduces to symmetric and/or binary versions of the clustering
    coefficient for respective graphs.
    """
    A = np.logical_not(W == 0).astype(float)  # adjacency matrix
    S = cuberoot(W) + cuberoot(W.T)  # symmetrized weights matrix ^1/3
    K = np.sum(A + A.T, axis=1)  # total degree (in+out)
    cyc3 = np.diag(np.dot(S, np.dot(S, S))) / 2  # number of 3-cycles
    if np.sum(((cyc3 == 0) * 1)) > 0:  
        K[np.where(cyc3 == 0)] = np.inf  # if no 3-cycles exist, make T=0
    # number of all possible 3-cycles
    CYC3 = K * (K - 1) - 2 * np.diag(np.dot(A, A))
    return np.sum(cyc3) / np.sum(CYC3)  # transitivity


def transitivity_wu(W):
    """
    Transitivity is the ratio of 'triangles to triplets' in the network.
    (A classical version of the clustering coefficient).
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        weighted undirected connection matrix
    Returns
    -------
    T : int
        transitivity scalar
    """
    K = np.sum(np.logical_not(W == 0), axis=1)
    ws = cuberoot(W)
    cyc3 = np.diag(np.dot(ws, np.dot(ws, ws)))
    return np.sum(cyc3, axis=0) / np.sum(K * (K - 1), axis=0) 
    
 
def degrees_dir(CIJ):
    """
    Node degree is the number of links connected to the node. The indegree
    is the number of inward links and the outdegree is the number of
    outward links.
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        directed binary/weighted connection matrix
    Returns
    -------
    in_degree : Nx1 :obj:`numpy.ndarray`
        node in-degree
    out_degree : Nx1 :obj:`numpy.ndarray`
        node out-degree
    deg : Nx1 :obj:`numpy.ndarray`
        node degree (in-degree + out-degree)
    Notes
    -----
    Inputs are assumed to be on the columns of the CIJ matrix.
           Weight information is discarded.
    """
    CIJ = binarize(CIJ, copy=True)  # ensure CIJ is binary
    in_degree = np.sum(CIJ, axis=0)  # indegree = column sum of CIJ
    out_degree = np.sum(CIJ, axis=1)  # outdegree = row sum of CIJ
    deg = in_degree + out_degree  # degree = indegree+outdegree
    return in_degree, out_degree, deg


def degrees_und(CIJ):
    """
    Node degree is the number of links connected to the node.
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        undirected binary/weighted connection matrix
    Returns
    -------
    deg : Nx1 :obj:`numpy.ndarray`
        node degree
    Notes
    -----
    Weight information is discarded.
    """
    CIJ = binarize(CIJ, copy=True)  # ensure CIJ is binary
    return np.sum(CIJ, axis=0)    


def strengths_dir(CIJ):
    """
    Node strength is the sum of weights of links connected to the node. The
    instrength is the sum of inward link weights and the outstrength is the
    sum of outward link weights.
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        directed weighted connection matrix
    Returns
    -------
    is : Nx1 :obj:`numpy.ndarray`
        node in-strength
    os : Nx1 :obj:`numpy.ndarray`
        node out-strength
    str : Nx1 :obj:`numpy.ndarray`
        node strength (in-strength + out-strength)
    Notes
    -----
    Inputs are assumed to be on the columns of the CIJ matrix.
    """
    istr = np.sum(CIJ, axis=0)
    ostr = np.sum(CIJ, axis=1)
    return istr, ostr


def strengths_und(CIJ):
    """
    Node strength is the sum of weights of links connected to the node.
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        undirected weighted connection matrix
    Returns
    -------
    str : Nx1 :obj:`numpy.ndarray`
        node strengths
    """
    return np.sum(CIJ, axis=0)


def strengths_und_sign(W):
    """
    Node strength is the sum of weights of links connected to the node.
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        undirected connection matrix with positive and negative weights
    Returns
    -------
    Spos : Nx1 :obj:`numpy.ndarray`
        nodal strength of positive weights
    Sneg : Nx1 :obj:`numpy.ndarray`
        nodal strength of positive weights
    vpos : float
        total positive weight
    vneg : float
        total negative weight
    """
    W = W.copy()
    np.fill_diagonal(W, 0)  # clear diagonal
    Spos = np.sum(W * (W > 0), axis=0)  # positive strengths
    Sneg = np.sum(W * (W < 0), axis=0)  # negative strengths

    vpos = np.sum(W[W > 0])  # positive weight
    vneg = np.sum(W[W < 0])  # negative weight
    return Spos, Sneg, vpos, vneg

def assortativity_bin(CIJ, flag=0):
    """
    The assortativity coefficient is a correlation coefficient between the
    degrees of all nodes on two opposite ends of a link. A positive
    assortativity coefficient indicates that nodes tend to link to other
    nodes with the same or similar degree.
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        binary directed/undirected connection matrix
    flag : int
        0 : undirected graph; degree/degree correlation
        1 : directed graph; out-degree/in-degree correlation
        2 : directed graph; in-degree/out-degree correlation
        3 : directed graph; out-degree/out-degree correlation
        4 : directed graph; in-degree/in-degreen correlation
    Returns
    -------
    r : float
        assortativity coefficient
    Notes
    -----
    The function accepts weighted networks, but all connection
    weights are ignored. The main diagonal should be empty. For flag 1
    the function computes the directed assortativity described in Rubinov
    and Sporns (2010) NeuroImage.
    """
    if flag == 0:  # undirected version
        deg = degrees_und(CIJ)
        i, j = np.where(np.triu(CIJ, 1) > 0)
        K = len(i)
        degi = deg[i]
        degj = deg[j]
    else:  # directed version
        id, od, deg = degrees_dir(CIJ)
        i, j = np.where(CIJ > 0)
        K = len(i)

        if flag == 1:
            degi = od[i]
            degj = id[j]
        elif flag == 2:
            degi = id[i]
            degj = od[j]
        elif flag == 3:
            degi = od[i]
            degj = od[j]
        elif flag == 4:
            degi = id[i]
            degj = id[j]
        else:
            raise ValueError('Flag must be 0-4')

    # compute assortativity
    term1 = np.sum(degi * degj) / K
    term2 = np.square(np.sum(.5 * (degi + degj)) / K)
    term3 = np.sum(.5 * (degi * degi + degj * degj)) / K
    r = (term1 - term2) / (term3 - term2)
    return r


def assortativity_wei(CIJ, flag=0):
    """
    The assortativity coefficient is a correlation coefficient between the
    strengths (weighted degrees) of all nodes on two opposite ends of a link.
    A positive assortativity coefficient indicates that nodes tend to link to
    other nodes with the same or similar strength.
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        weighted directed/undirected connection matrix
    flag : int
        0 : undirected graph; strength/strength correlation
        1 : directed graph; out-strength/in-strength correlation
        2 : directed graph; in-strength/out-strength correlation
        3 : directed graph; out-strength/out-strength correlation
        4 : directed graph; in-strength/in-strengthn correlation
    Returns
    -------
    r : float
        assortativity coefficient
    Notes
    -----
    The main diagonal should be empty. For flag 1
       the function computes the directed assortativity described in Rubinov
       and Sporns (2010) NeuroImage.
    """
    if flag == 0:  # undirected version
        str = strengths_und(CIJ)
        i, j = np.where(np.triu(CIJ, 1) > 0)
        K = len(i)
        stri = str[i]
        strj = str[j]
    else:
        ist, ost = strengths_dir(CIJ)  # directed version
        i, j = np.where(CIJ > 0)
        K = len(i)

        if flag == 1:
            stri = ost[i]
            strj = ist[j]
        elif flag == 2:
            stri = ist[i]
            strj = ost[j]
        elif flag == 3:
            stri = ost[i]
            strj = ost[j]
        elif flag == 4:
            stri = ist[i]
            strj = ost[j]
        else:
            raise ValueError('Flag must be 0-4')

    # compute assortativity
    term1 = np.sum(stri * strj) / K
    term2 = np.square(np.sum(.5 * (stri + strj)) / K)
    term3 = np.sum(.5 * (stri * stri + strj * strj)) / K
    r = (term1 - term2) / (term3 - term2)
    return r
    
    
def betweenness_bin(G):
    """
    Node betweenness centrality is the fraction of all shortest paths in
    the network that contain a given node. Nodes with high values of
    betweenness centrality participate in a large number of shortest paths.
    Parameters
    ----------
    A : NxN :obj:`numpy.ndarray`
        binary directed/undirected connection matrix
    BC : Nx1 :obj:`numpy.ndarray`
        node betweenness centrality vector
    Notes
    -----
    Betweenness centrality may be normalised to the range [0,1] as
    BC/[(N-1)(N-2)], where N is the number of nodes in the network.
    """
    G = np.array(G, dtype=float)  # force G to have float type so it can be
    # compared to float np.inf

    n = len(G)  # number of nodes
    eye = np.eye(n)  # identity matrix
    d = 1  # path length
    NPd = G.copy()  # number of paths of length |d|
    NSPd = G.copy()  # number of shortest paths of length |d|
    NSP = G.copy()  # number of shortest paths of any length
    L = G.copy()  # length of shortest paths

    NSP[np.where(eye)] = 1
    L[np.where(eye)] = 1

    # calculate NSP and L
    while np.any(NSPd):
        d += 1
        NPd = np.dot(NPd, G)
        NSPd = NPd * (L == 0)
        NSP += NSPd
        L = L + d * (NSPd != 0)

    L[L == 0] = np.inf  # L for disconnected vertices is inf
    L[np.where(eye)] = 0
    NSP[NSP == 0] = 1  # NSP for disconnected vertices is 1

    DP = np.zeros((n, n))  # vertex on vertex dependency
    diam = d - 1

    # calculate DP
    for d in range(diam, 1, -1):
        DPd1 = np.dot(((L == d) * (1 + DP) / NSP), G.T) * \
            ((L == (d - 1)) * NSP)
        DP += DPd1

    return np.sum(DP, axis=0)


def betweenness_wei(G):
    """
    Node betweenness centrality is the fraction of all shortest paths in
    the network that contain a given node. Nodes with high values of
    betweenness centrality participate in a large number of shortest paths.
    Parameters
    ----------
    L : NxN :obj:`numpy.ndarray`
        directed/undirected weighted connection matrix
    Returns
    -------
    BC : Nx1 :obj:`numpy.ndarray`
        node betweenness centrality vector
    Notes
    -----
    The input matrix must be a connection-length matrix, typically
    obtained via a mapping from weight to length. For instance, in a
    weighted correlation network higher correlations are more naturally
    interpreted as shorter distances and the input matrix should
    consequently be some inverse of the connectivity matrix.
    Betweenness centrality may be normalised to the range [0,1] as
    BC/[(N-1)(N-2)], where N is the number of nodes in the network.
    """
    n = len(G)
    BC = np.zeros((n,))  # vertex betweenness

    for u in range(n):
        D = np.tile(np.inf, (n,))
        D[u] = 0  # distance from u
        NP = np.zeros((n,))
        NP[u] = 1  # number of paths from u
        S = np.ones((n,), dtype=bool)  # distance permanence
        P = np.zeros((n, n))  # predecessors
        Q = np.zeros((n,), dtype=int)  # indices
        q = n - 1  # order of non-increasing distance

        G1 = G.copy()
        V = [u]
        while True:
            S[V] = 0  # distance u->V is now permanent
            G1[:, V] = 0  # no in-edges as already shortest
            for v in V:
                Q[q] = v
                q -= 1
                W, = np.where(G1[v, :])  # neighbors of v
                for w in W:
                    Duw = D[v] + G1[v, w]  # path length to be tested
                    if Duw < D[w]:  # if new u->w shorter than old
                        D[w] = Duw
                        NP[w] = NP[v]  # NP(u->w) = NP of new path
                        P[w, :] = 0
                        P[w, v] = 1  # v is the only predecessor
                    elif Duw == D[w]:  # if new u->w equal to old
                        NP[w] += NP[v]  # NP(u->w) sum of old and new
                        P[w, v] = 1  # v is also predecessor

            if D[S].size == 0:
                break  # all nodes were reached
            if np.isinf(np.min(D[S])):  # some nodes cannot be reached
                Q[:q + 1], = np.where(np.isinf(D))  # these are first in line
                break
            V, = np.where(D == np.min(D[S]))

        DP = np.zeros((n,))
        for w in Q[:n - 1]:
            BC[w] += DP[w]
            for v in np.where(P[w, :])[0]:
                DP[v] += (1 + DP[w]) * NP[v] / NP[w]

    return BC   
    

def flow_coef_bd(CIJ):
    """
    Computes the flow coefficient for each node and averaged over the
    network, as described in Honey et al. (2007) PNAS. The flow coefficient
    is similar to betweenness centrality, but works on a local
    neighborhood. It is mathematically related to the clustering
    coefficient  (cc) at each node as, fc+cc <= 1.
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        binary directed connection matrix
    Returns
    -------
    fc : Nx1 :obj:`numpy.ndarray`
        flow coefficient for each node
    FC : float
        average flow coefficient over the network
    total_flo : int
        number of paths that "flow" across the central node
    """
    N = len(CIJ)

    fc = np.zeros((N,))
    total_flo = np.zeros((N,))
    max_flo = np.zeros((N,))

    # loop over nodes
    for v in range(N):
        # find neighbors - note: both incoming and outgoing connections
        nb, = np.where(CIJ[v, :] + CIJ[:, v].T)
        fc[v] = 0
        if np.where(nb)[0].size:
            CIJflo = -CIJ[np.ix_(nb, nb)]
            for i in range(len(nb)):
                for j in range(len(nb)):
                    if CIJ[nb[i], v] and CIJ[v, nb[j]]:
                        CIJflo[i, j] += 1
            total_flo[v] = np.sum(
                (CIJflo == 1) * np.logical_not(np.eye(len(nb))))
            max_flo[v] = len(nb) * len(nb) - len(nb)
            fc[v] = total_flo[v] / max_flo[v]

    fc[np.isnan(fc)] = 0
    FC = np.mean(fc)

    return fc, FC, total_flo   
    

def module_degree_zscore(W, ci, flag=0):
    """
    The within-module degree z-score is a within-module version of degree
    centrality.
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        binary/weighted directed/undirected connection matrix
    ci : Nx1 np.array_like
        community affiliation vector
    flag : int
        Graph type. 0: undirected graph (default)
                    1: directed graph in degree
                    2: directed graph out degree
                    3: directed graph in and out degree
    Returns
    -------
    Z : Nx1 :obj:`numpy.ndarray`
        within-module degree Z-score
    """
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    if flag == 2:
        W = W.copy()
        W = W.T
    elif flag == 3:
        W = W.copy()
        W = W + W.T

    n = len(W)
    Z = np.zeros((n,))  # number of vertices
    for i in range(1, int(np.max(ci) + 1)):
        Koi = np.sum(W[np.ix_(ci == i, ci == i)], axis=1)
        Z[np.where(ci == i)] = (Koi - np.mean(Koi)) / np.std(Koi)

    Z[np.where(np.isnan(Z))] = 0
    return Z   
    
def pick_four_unique_nodes_quickly(n):
    """
    This is equivalent to np.random.choice(n, 4, replace=False)
    Another fellow suggested np.random.random(n).argpartition(4) which is
    clever but still substantially slower.
    """
    k = np.random.randint(n**4)
    a = k % n
    b = k // n % n
    c = k // n ** 2 % n
    d = k // n ** 3 % n
    if (a != b and a != c and a != d and b != c and b != d and c != d):
        return (a, b, c, d)
    else:
        # the probability of finding a wrong configuration is extremely low
        # unless for extremely small n. if n is extremely small the
        # computational demand is not a problem.

        # In my profiling it only took 0.4 seconds to include the uniqueness
        # check in 1 million runs of this function so I think it is OK.
        return pick_four_unique_nodes_quickly(n)


def randmio_und_signed(R, itr):
    """
    This function randomizes an undirected weighted network with positive
    and negative weights, while simultaneously preserving the degree
    distribution of positive and negative weights. The function does not
    preserve the strength distribution in weighted networks.
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        undirected binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.
    Returns
    -------
    R : NxN :obj:`numpy.ndarray`
        randomized network
    """
    R = R.copy()
    n = len(R)

    itr *= int(n * (n - 1) / 2)

    max_attempts = int(np.round(n / 2))
    eff = 0

    for it in range(int(itr)):
        att = 0
        while att <= max_attempts:

            a, b, c, d = pick_four_unique_nodes_quickly(n)

            r0_ab = R[a, b]
            r0_cd = R[c, d]
            r0_ad = R[a, d]
            r0_cb = R[c, b]

            # rewiring condition
            if (np.sign(r0_ab) == np.sign(r0_cd) and
                    np.sign(r0_ad) == np.sign(r0_cb) and
                    np.sign(r0_ab) != np.sign(r0_ad)):

                R[a, d] = R[d, a] = r0_ab
                R[a, b] = R[b, a] = r0_ad

                R[c, b] = R[b, c] = r0_cd
                R[c, d] = R[d, c] = r0_cb

                eff += 1
                break

            att += 1

    return R, eff

def null_model_und_sign(W, bin_swaps=5, wei_freq=.1):
    """
    This function randomizes an undirected network with positive and
    negative weights, while preserving the degree and strength
    distributions. This function calls randmio_und.m
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        undirected weighted connection matrix
    bin_swaps : int
        average number of swaps in each edge binary randomization. Default
        value is 5. 0 swaps implies no binary randomization.
    wei_freq : float
        frequency of weight sorting in weighted randomization. 0<=wei_freq<1.
        wei_freq == 1 implies that weights are sorted at each step.
        wei_freq == 0.1 implies that weights sorted each 10th step (faster,
            default value)
        wei_freq == 0 implies no sorting of weights (not recommended)
    Returns
    -------
    W0 : NxN :obj:`numpy.ndarray`
        randomized weighted connection matrix
    R : 4-tuple of floats
        Correlation coefficients between strength sequences of input and
        output connection matrices, rpos_in, rpos_out, rneg_in, rneg_out
    Notes
    -----
    The value of bin_swaps is ignored when binary topology is fully
        connected (e.g. when the network has no negative weights).
    Randomization may be better (and execution time will be slower) for
        higher values of bin_swaps and wei_freq. Higher values of bin_swaps
        may enable a more random binary organization, and higher values of
        wei_freq may enable a more accurate conservation of strength
        sequences.
    R are the correlation coefficients between positive and negative
        strength sequences of input and output connection matrices and are
        used to evaluate the accuracy with which strengths were preserved.
        Note that correlation coefficients may be a rough measure of
        strength-sequence accuracy and one could implement more formal tests
        (such as the Kolmogorov-Smirnov test) if desired.
    """
    if not np.all(W == W.T):
        raise TypeError("Input must be undirected")
    W = W.copy()
    n = len(W)
    np.fill_diagonal(W, 0)  # clear diagonal
    Ap = (W > 0)  # positive adjmat
    An = (W < 0)  # negative adjmat

    if np.size(np.where(Ap.flat)) < (n * (n - 1)):
        W_r, eff = randmio_und_signed(W, bin_swaps)
        Ap_r = W_r > 0
        An_r = W_r < 0
    else:
        Ap_r = Ap
        An_r = An

    W0 = np.zeros((n, n))
    for s in (1, -1):
        if s == 1:
            Acur = Ap
            A_rcur = Ap_r
        else:
            Acur = An
            A_rcur = An_r

        S = np.sum(W * Acur, axis=0)  # strengths
        Wv = np.sort(W[np.where(np.triu(Acur))])  # sorted weights vector
        i, j = np.where(np.triu(A_rcur))
        Lij, = np.where(np.triu(A_rcur).flat)  # weights indices

        P = np.outer(S, S)

        if wei_freq == 0:  # get indices of Lij that sort P
            Oind = np.argsort(P.flat[Lij])  # assign corresponding sorted
            W0.flat[Lij[Oind]] = s * Wv  # weight at this index
        else:
            wsize = np.size(Wv)
            wei_period = np.round(1 / wei_freq)  # convert frequency to period
            lq = np.arange(wsize, 0, -wei_period, dtype=int)
            for m in lq:  # iteratively explore at this period
                # get indices of Lij that sort P
                Oind = np.argsort(P.flat[Lij])
                R = np.random.permutation(m)[:np.min((m, int(wei_period)))]
                for q, r in enumerate(R):
                    # choose random index of sorted expected weight
                    o = Oind[r]
                    W0.flat[Lij[o]] = s * Wv[r]  # assign corresponding weight

                    # readjust expected weighted probability for i[o],j[o]
                    f = 1 - Wv[r] / S[i[o]]
                    P[i[o], :] *= f
                    P[:, i[o]] *= f
                    f = 1 - Wv[r] / S[j[o]]
                    P[j[o], :] *= f
                    P[:, j[o]] *= f

                    # readjust strength of i[o]
                    S[i[o]] -= Wv[r]
                    # readjust strength of j[o]
                    S[j[o]] -= Wv[r]

                O = Oind[R]
                # remove current indices from further consideration
                Lij = np.delete(Lij, O)
                i = np.delete(i, O)
                j = np.delete(j, O)
                Wv = np.delete(Wv, R)

    W0 = W0 + W0.T

    rpos_in = np.corrcoef(np.sum(W * (W > 0), axis=0),
                          np.sum(W0 * (W0 > 0), axis=0))
    rpos_ou = np.corrcoef(np.sum(W * (W > 0), axis=1),
                          np.sum(W0 * (W0 > 0), axis=1))
    rneg_in = np.corrcoef(np.sum(-W * (W < 0), axis=0),
                          np.sum(-W0 * (W0 < 0), axis=0))
    rneg_ou = np.corrcoef(np.sum(-W * (W < 0), axis=1),
                          np.sum(-W0 * (W0 < 0), axis=1))
    return W0, (rpos_in[0, 1], rpos_ou[0, 1], rneg_in[0, 1], rneg_ou[0, 1])


def participation_coef_norm(W, ci, degree = 'undirected', BA_iters = 100):
    """
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.
    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        binary/weighted directed/undirected connection matrix
    ci : Nx1 :obj:`numpy.ndarray`
        community affiliation vector
    degree : {'undirected', 'in', 'out'}, optional
        Flag to describe nature of graph. 'undirected': For undirected graphs,
        'in': Uses the in-degree, 'out': Uses the out-degree
    Returns
    -------
    P : Nx1 :obj:`numpy.ndarray`
        participation coefficient
    """
    
    if degree == 'in':
        W = W.T
    
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out) degree
    Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation
    
    P_mat = np.zeros((n,BA_iters))
    
    for rand in range(0,BA_iters):
        Kc2 = np.zeros((n,))  # community-specific neighbors
        np.random.seed(rand)
    
        W_rand = null_model_und_sign(W)[0] #Null model 
        Gc_rand = np.dot((W_rand != 0), np.diag(ci))  # neighbor community affiliation
        
        for i in range(1, int(np.max(ci)) + 1):
            
            Kc2 += np.square((np.sum(W * (Gc == i), axis=1) - np.sum(W_rand * (Gc_rand == i), axis=1)) / Ko)
    
        P = np.ones((n,)) - np.sqrt(0.5 * Kc2)
        # P=0 if for nodes with no (out) neighbors
        P[np.where(np.logical_not(Ko))] = 0
        P[P < 0] = 0
        P_mat[:,rand] = P
        print(rand)
        
    P_mean = np.mean(P_mat, axis = 1)
    P_std = np.std(P_mat, axis = 1)

    return(P_mean, P_std)  
    

def rich_club_bd(CIJ, klevel=None):
    """
    The rich club coefficient, R, at level k is the fraction of edges that
    connect nodes of degree k or higher out of the maximum number of edges
    that such nodes might share.
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        binary directed connection matrix
    klevel : int | None
        sets the maximum level at which the rich club coefficient will be
        calculated. If None (default), the maximum level is set to the
        maximum degree of the adjacency matrix
    Returns
    -------
    R : Kx1 :obj:`numpy.ndarray`
        vector of rich-club coefficients for levels 1 to klevel
    Nk : int
        number of nodes with degree > k
    Ek : int
        number of edges remaining in subgraph with degree > k
    """
    # definition of degree as used for RC coefficients
    # degree is taken to be the sum of incoming and outgoing connections
    id, od, deg = degrees_dir(CIJ)

    if klevel is None:
        klevel = int(np.max(deg))

    R = np.zeros((klevel,))
    Nk = np.zeros((klevel,))
    Ek = np.zeros((klevel,))
    for k in range(klevel):
        SmallNodes, = np.where(deg <= k + 1)  # get small nodes with degree <=k
        subCIJ = np.delete(CIJ, SmallNodes, axis=0)
        subCIJ = np.delete(subCIJ, SmallNodes, axis=1)
        Nk[k] = np.size(subCIJ, axis=1)  # number of nodes with degree >k
        Ek[k] = np.sum(subCIJ)  # number of connections in subgraph
        # unweighted rich club coefficient
        R[k] = Ek[k] / (Nk[k] * (Nk[k] - 1))

    return R, Nk, Ek


def rich_club_bu(CIJ, klevel=None):
    """
    The rich club coefficient, R, at level k is the fraction of edges that
    connect nodes of degree k or higher out of the maximum number of edges
    that such nodes might share.
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        binary undirected connection matrix
    klevel : int | None
        sets the maximum level at which the rich club coefficient will be
        calculated. If None (default), the maximum level is set to the
        maximum degree of the adjacency matrix
    Returns
    -------
    R : Kx1 :obj:`numpy.ndarray`
        vector of rich-club coefficients for levels 1 to klevel
    Nk : int
        number of nodes with degree > k
    Ek : int
        number of edges remaining in subgraph with degree > k
    """
    deg = degrees_und(CIJ)  # compute degree of each node

    if klevel is None:
        klevel = int(np.max(deg))

    R = np.zeros((klevel,))
    Nk = np.zeros((klevel,))
    Ek = np.zeros((klevel,))
    for k in range(klevel):
        SmallNodes, = np.where(deg <= k + 1)  # get small nodes with degree <=k
        subCIJ = np.delete(CIJ, SmallNodes, axis=0)
        subCIJ = np.delete(subCIJ, SmallNodes, axis=1)
        Nk[k] = np.size(subCIJ, axis=1)  # number of nodes with degree >k
        Ek[k] = np.sum(subCIJ)  # number of connections in subgraph
        # unweighted rich club coefficient
        R[k] = Ek[k] / (Nk[k] * (Nk[k] - 1))

    return R, Nk, Ek


def rich_club_wd(CIJ, klevel=None):
    """
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        weighted directed connection matrix
    klevel : int | None
        sets the maximum level at which the rich club coefficient will be
        calculated. If None (default), the maximum level is set to the
        maximum degree of the adjacency matrix
    Returns
    -------
    Rw : Kx1 :obj:`numpy.ndarray`
        vector of rich-club coefficients for levels 1 to klevel
    """
    # nr_nodes = len(CIJ)
    # degree of each node is defined here as in+out
    deg = np.sum((CIJ != 0), axis=0) + np.sum((CIJ.T != 0), axis=0)

    if klevel is None:
        klevel = np.max(deg)
    Rw = np.zeros((klevel,))

    # sort the weights of the network, with the strongest connection first
    wrank = np.sort(CIJ.flat)[::-1]

    for k in range(klevel):
        SmallNodes, = np.where(deg < k + 1)
        if np.size(SmallNodes) == 0:
            Rw[k] = np.nan
            continue

        # remove small nodes with node degree < k
        cutCIJ = np.delete(
            np.delete(CIJ, SmallNodes, axis=0), SmallNodes, axis=1)
        # total weight of connections in subset E>r
        Wr = np.sum(cutCIJ)
        # total number of connections in subset E>r
        Er = np.size(np.where(cutCIJ.flat != 0), axis=1)
        # E>r number of connections with max weight in network
        wrank_r = wrank[:Er]
        # weighted rich-club coefficient
        Rw[k] = Wr / np.sum(wrank_r)
    return Rw


def rich_club_wu(CIJ, klevel=None):
    """
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        weighted undirected connection matrix
    klevel : int | None
        sets the maximum level at which the rich club coefficient will be
        calculated. If None (default), the maximum level is set to the
        maximum degree of the adjacency matrix
    Returns
    -------
    Rw : Kx1 :obj:`numpy.ndarray`
        vector of rich-club coefficients for levels 1 to klevel
    """
    # nr_nodes = len(CIJ)
    deg = np.sum((CIJ != 0), axis=0)

    if klevel is None:
        klevel = np.max(deg)
    Rw = np.zeros((klevel,))

    # sort the weights of the network, with the strongest connection first
    wrank = np.sort(CIJ.flat)[::-1]

    for k in range(klevel):
        SmallNodes, = np.where(deg < k + 1)
        if np.size(SmallNodes) == 0:
            Rw[k] = np.nan
            continue

        # remove small nodes with node degree < k
        cutCIJ = np.delete(
            np.delete(CIJ, SmallNodes, axis=0), SmallNodes, axis=1)
        # total weight of connections in subset E>r
        Wr = np.sum(cutCIJ)
        # total number of connections in subset E>r
        Er = np.size(np.where(cutCIJ.flat != 0), axis=1)
        # E>r number of connections with max weight in network
        wrank_r = wrank[:Er]
        # weighted rich-club coefficient
        Rw[k] = Wr / np.sum(wrank_r)
    return Rw


def score_wu(CIJ, s):
    """
    The s-core is the largest subnetwork comprising nodes of strength at
    least s. This function computes the s-core for a given weighted
    undirected connection matrix. Computation is analogous to the more
    widely used k-core, but is based on node strengths instead of node
    degrees.
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        weighted undirected connection matrix
    s : float
        level of s-core. Note that can take on any fractional value.
    Returns
    -------
    CIJscore : NxN :obj:`numpy.ndarray`
        connection matrix of the s-core. This matrix contains only nodes with
        a strength of at least s.
    sn : int
        size of s-core
    """
    CIJscore = CIJ.copy()
    while True:
        str = strengths_und(CIJscore)  # get strengths of matrix

        # find nodes with strength <s
        ff, = np.where(np.logical_and(str < s, str > 0))

        if ff.size == 0:
            break  # if none found -> stop

        # else peel away found nodes
        CIJscore[ff, :] = 0
        CIJscore[:, ff] = 0

    sn = np.sum(str > 0)
    return CIJscore, sn
   
    
    
    


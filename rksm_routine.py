import numpy as np
import scipy.sparse as sp
import math
import scipy.sparse.linalg as sla
from numpy.random import randn as nprand
import scipy.linalg as la
from scipy.io import loadmat, savemat
from time import time
import pickle
import pyamg
from scipy.spatial import ConvexHull


def ratfun(x, eH, s):
    r = np.zeros(x.shape)
    ones_s = np.ones(s.shape, dtype=complex)
    ones_eH = np.ones(eH.shape, dtype=complex)
    
    for j in range(1, len(x)):
        r[j] = la.norm(np.prod(ones_s * x[j] - s)/np.prod(ones_eH * x[j]-eH))
    return r


def newpolei(eHpoints,eH,s):
    if len(s) == 1:
        return eHpoints[0]
    cnt = 200
    rmax, smax = 0., 0.
    for j in range(1, len(eHpoints)-1):
        sval = np.linspace(eHpoints[j], eHpoints[j+1], cnt)
        
        rj = ratfun(sval, eH, s)
        j1 = np.argmax(rj)
        if rj[j1] > rmax:
            rmax = rj[j1]
            smax = eHpoints[j] + j1*(eHpoints[j+1] - eHpoints[j])/(cnt-1)
    if smax.real < 0: smax = -smax.real + 1j * smax.imag
    return smax


def get_eH(a, u):
    return la.eig(u.T.dot(a.dot(u)))[0]


def extend_nparray(array, values, array_flag = False):
    cnt = 1
    if array_flag:
        cnt = values.shape[0]
        
    new_array = np.ndarray(array.shape[0]+cnt, dtype=complex) #array.dtype
    new_array[:-cnt] = array
    new_array[-cnt:] = np.array(values, dtype=complex) #array.dtype
    return new_array


def get_convex_points(array):
    mypoints = np.hstack((array.real.reshape(-1, 1), array.imag.reshape(-1, 1)))
    hull = ConvexHull(mypoints)
    return array[hull.vertices]


def get_eHpoints(eHorig, shifts, s0, complex_flag = False):
    i = len(eHorig)
    eH = eHorig.copy()
    
    if complex_flag: # Complex poles.
        if  np.any(np.fabs(eH.imag) > 1e-8 * np.fabs(eH.real)) and (len(eH)>2):
            eH = extend_nparray(eH, s0[1])
            eH = get_convex_points(eH)  
            
            # include enough points from the border
            while i - len(eH) > 0: eH = extend_nparray(eH, 0.5 * (eH[:len(eH) - 1] + eH[1:len(eH)]), array_flag=True)
            #eH=eH[:i]
            eHpoints = -eH
            eH = eHorig
        else:
            eHpoints = np.sort(extend_nparray(-eH.real, s0, array_flag=True))
            
    else: # Real poles s from real set.
    
        if  np.any(np.fabs(eH.imag) > 1e-8 * np.fabs(eH.real)) and (len(eH)>2): 
            # Roots lambdas come from convex hull too
            eH = extend_nparray(eH, s0, array_flag=True)
            eH = get_convex_points(eH)            

            # include enough points from the border
            while i - len(eH) > 0: eH = extend_nparray(eH, 0.5 * (eH[:len(eH) - 1] + eH[1:len(eH)]), array_flag=True)
            eH=eH[:i]

        eHpoints = np.sort(extend_nparray(-eH.real, s0, array_flag=True))
        eH = eHorig
    return eHpoints


def update_resnrm(a, u, y0, resnrm):    
    au = a.dot(u)
    b = u.T.dot(au)
    yr = u.T.dot(y0)
    z = la.solve_lyapunov(b, yr.dot(yr.T))
    resnrm.append(la.norm((au - u.dot(b)).dot(z)))
    return


def get_new_shift(a, u, shifts, s0, complex_flag, krylov_flag):
    eH = -get_eH(a, u)
    eHpoints = get_eHpoints(eH.copy(), shifts, s0, complex_flag)

    if len(shifts) > 0:
        s = newpolei(eHpoints, eH, np.array(shifts))
        shifts = extend_nparray(shifts, s)
        if s.imag > s.real * 1e-8:
            shifts = extend_nparray(shifts, s.conjugate())
        k = u.shape[1]-1
        if krylov_flag:
            k = k/2
        return shifts[k], shifts
    # initial shift choice
    s = s0[1]
    shifts = np.array([s], dtype=complex)
    return s, shifts
    
    
def get_shift_estimation(a):
    s2 = sla.eigs(-a, k=1, which='LR', return_eigenvectors=False, tol=1e-2)[0].real
    s1 = sla.eigs(-a, k=1, which='SR', return_eigenvectors=False, tol=5e-1)[0].real
    return np.array([s1, s2], dtype=complex)
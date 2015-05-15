import sys
sys.path.append('../')

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

#usefull functions
from rational_krylov_routine import *
from rksm_routine import *

#pyRKSM    
def rksm(a, y0, solver_tol = 1e-14, acc = 1e-8, krylov_flag=False, complex_flag = False, r = 100, mode='direct'):
    
    s0 = get_shift_estimation(a)
    if not complex_flag:
        s0[0], s0[1] = s0[0].real, s0[1].real
    
    solver_data, time_elapse, init_time, resnrm_rksm, shift_list = {}, [], [], [], []
    
    #factorisation/pyamg initialisation
    time_count_start(init_time)
    mode_solver_initialisation(mode, solver_data, a, solver_tol)
    time_count_finish(init_time)
    
    time_count_start(time_elapse)
    
    n = a.shape[0]
    b = np.zeros((2*r+3, 2*r+3), dtype=complex)
    au = np.zeros((n, 2*r+3), dtype=complex)
    res_au = np.zeros((n, 2*r+3), dtype=complex)
    
    # First step
    y0 = y0.reshape(-1, 1) / la.norm(y0)
    u = y0
    au[:, :1] = a.dot(y0)
    b[:1, :1] = u.T.dot(au[:, :1])
    #print 'B\n', b[:2, :2]
    res_au[:, :1] = au[:, :1]
    res_au[:, :1] = res_au[:, :1] - u.dot(u.T.dot(res_au[:, :1]))
    v = np.zeros(y0.shape, dtype=complex)
    
    resnorm = la.norm(res_au[:, 0] / (2*b[0, 0]))
    resnrm_rksm.append(resnorm)
    shift = s0[1]
    shift_list = np.array([s0[1], s0[0]])
    toler = resnorm * acc
    
    time_count_finish(time_elapse)
    
    for i in xrange(r):
        time_count_start(time_elapse)
        # Solving linear system:
        w = u[:, -1:]
        
        #print_cnt = 20 #debugging print
        #if i == print_cnt * (i//print_cnt) and i > 0: print i, 'shift ', shift
        #print shift, la.norm(w.imag)
        
        v[:, 0] = mode_solver(mode, solver_data, -shift, w)[:, 0] # RKSM 'uses' positive shifts. 
        
        # Enlarge U matrix using gram-shmidt
        u = np.hstack((u, gram_schmidt_orth(u, v)))
        
        if krylov_flag:
            u = np.hstack((u, gram_schmidt_orth(u, res_au[:, 2*i])))   
            # Do 2 matvecs:
            au[:, 2*i+1:2*i+3] = a.dot(u[:, -2:])
        else:
            # Do 1 matvec:
            au[:, i+1:i+2] = a.dot(u[:, -1:])
        #print 'dAU', la.norm(au[:, :u.shape[1]] - a.dot(u))

        # Completing B matrix using known A*U (without matvecs).
        b_i = update_b_full_way(au, u, b, double_flag=krylov_flag) 
        # Update (A * U - U * B) residual: 
        res_au = update_res_au(res_au, u, au, b, double_flag=krylov_flag, alr_way=krylov_flag)
        
        # if code breaks below then your subspace matrix U contains nonfinite values:
        z_i = compute_lyapunov_solution(b_i)
        shift, shift_list = get_new_shift(a, u, shift_list, s0, complex_flag, krylov_flag)
        
        # compute residual norm:
        k = i + 2
        if krylov_flag:
            k = 2 * i + 3
        resnorm = la.norm(res_au[:, :k].dot(z_i))
        resnrm_rksm.append(resnorm)
        time_count_finish(time_elapse)

        if resnorm < toler:
            break

    return {'resnorm': resnrm_rksm, 'u': u, 'time' : time_elapse, 'mode': mode, 
            'init_time': init_time, 'solver_timing': solver_data['solver_timing'], 
            'is_solver_direct': solver_data['is_solver_direct'], 'shifts': shift_list}


#pyKPIK
def kpik(a, y0, acc=1e-8, solver_tol=1e-14, mode='direct', r = 100):
    
    solver_data, time_elapse, init_time, resnrm_list = {}, [], [], []
       
    #factorisation/pyamg initialisation
    time_count_start(init_time)
    mode_solver_initialisation(mode, solver_data, a, solver_tol)
    time_count_finish(init_time)
    
    time_count_start(time_elapse)
    n = a.shape[0]
    b = np.zeros((2*r+3, 2*r+3))
    au = np.zeros((n, 2*r+3))
    res_au = np.zeros((n, 2*r+3))
    
    # First step
    y0 = y0.reshape(-1, 1) / la.norm(y0)
    u = y0
    au[:, :1] = a.dot(y0)
    b[:1, :1] = u.T.dot(au[:, :1])
    res_au[:, :1] = au[:, :1]
    res_au[:, :1] = res_au[:, :1] - u.dot(u.T.dot(res_au[:, :1]))
    
    resnorm = la.norm(res_au[:, 0] / (2*b[0, 0]))
    resnrm_list.append(resnorm)
    shift = b[0, 0]
    toler = resnorm * acc
    
    time_count_finish(time_elapse)

    # initial w choice
    w = y0
    for i in xrange(r):
        time_count_start(time_elapse)
        # Solving linear system:
        #print i, u.shape[1]
        if i > 0:
            #w = res_a
            w = u[:, 2*i-1:2*i]
            w /= la.norm(w)
        v = mode_solver(mode, solver_data, 0, w)
        #if i == 0:
        #    w = a.dot(y0) - b[0, 0] * y0
        
        # Enlarge U matrix using gram-shmidt
        u = np.hstack((u, gram_schmidt_orth(u, v)))
        u = np.hstack((u, gram_schmidt_orth(u, a.dot(u[:, 2*i:2*i+1])))) 
        
        # Do 2 matvecs:
        au[:, 2*i+1:2*i+3] = a.dot(u[:, -2:])
        # Completing B matrix using known A*U (without matvecs). 
        b_i = update_b_full_way(au, u, b)
        # Update (A * U - U * B) residual: 
        res_au = update_res_au(res_au, u, au, b, alr_way=True)#, alr_way=True
        
        # if code breaks below then your subspace matrix U contains nonfinite values:
        z_i = compute_lyapunov_solution(b_i)
        
        # compute residual norm:
        resnorm = la.norm(res_au[:, :2*i+3].dot(z_i))
        resnrm_list.append(resnorm)
        time_count_finish(time_elapse)

        if resnorm < toler:
            break
    return {'resnorm': resnrm_list, 'u': u, 'b': b, 'time' : time_elapse, 'mode': mode, 
            'init_time': init_time, 'solver_timing': solver_data['solver_timing'],
            'is_solver_direct': solver_data['is_solver_direct']}



#ALR method
def compute_alr_shift(b, z):
    if b.shape == (1, 1):
        return b[0, 0]
        
    qn = z[-2, :].reshape(-1, 1)   
    qn /= la.norm(qn)    
    return qn.T.dot(b).dot(qn)[0, 0]
    

def alr(a, y0, acc=1e-12, solver_tol=1e-13, mode='direct', r = 50): 

    solver_data, time_elapse, init_time, resnrm_alr = {}, [], [], []
       
    #factorisation/pyamg initialisation
    time_count_start(init_time)
    mode_solver_initialisation(mode, solver_data, a, solver_tol)
    time_count_finish(init_time)
    
    time_count_start(time_elapse)
    n = a.shape[0]
    b = np.zeros((2*r+3, 2*r+3))
    au = np.zeros((n, 2*r+3))
    res_au = np.zeros((n, 2*r+3))
    
    # First step
    y0 = y0.reshape(-1, 1) / la.norm(y0)
    u = y0
    au[:, :1] = a.dot(y0)
    b[:1, :1] = u.T.dot(au[:, :1])
    res_au[:, :1] = au[:, :1]
    res_au[:, :1] = res_au[:, :1] - u.dot(u.T.dot(res_au[:, :1]))
    
    resnorm = la.norm(res_au[:, 0] / (2*b[0, 0]))
    resnrm_alr.append(resnorm)
    shift = b[0, 0]
    toler = resnorm * acc
    
    time_count_finish(time_elapse)

    #print resnorm
    for i in xrange(r):            
        time_count_start(time_elapse)
        # Solving linear system:
        w = res_au[:, 2*i:2*i+1]
        w /= la.norm(w)
        v = mode_solver(mode, solver_data, shift, w)
        
        # Enlarge U matrix using gram-shmidt
        u = np.hstack((u, gram_schmidt_orth(u, v)))
        u = np.hstack((u, gram_schmidt_orth(u, w))) 
        
        # Do 2 matvecs:
        au[:, 2*i+1:2*i+3] = a.dot(u[:, -2:])
        # Completing B matrix using known A*U (without matvecs). 
        b_i = update_b_full_way(au, u, b) #b_i = update_b_alr_way(au, u, b)
        # Update (A * U - U * B) residual: 
        res_au = update_res_au(res_au, u, au, b, alr_way=True)
        
        # if code breaks below then your subspace matrix U contains nonfinite values:
        z_i = compute_lyapunov_solution(b_i)
        shift = compute_alr_shift(b_i, z_i)
        
        # compute residual norm:
        resnorm = la.norm(res_au[:, :2*i+3].dot(z_i))
        resnrm_alr.append(resnorm)
        time_count_finish(time_elapse)

        if resnorm < toler:
            break
    return {'resnorm': resnrm_alr, 'u': u, 'b': b, 'time' : time_elapse, 'mode': mode, 
            'init_time': init_time, 'solver_timing': solver_data['solver_timing'],
            'is_solver_direct': solver_data['is_solver_direct'],}
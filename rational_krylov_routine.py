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


def gram_schmidt_orth(u, v):
    v = v.reshape(-1, 1) 
    v_norm = la.norm(v)
    v -= u.dot(u.T.conjugate().dot(v))
    if la.norm(v) < 0.3 * v_norm:
        v /= la.norm(v)
        v -= u.dot(u.T.conjugate().dot(v))
        v /= la.norm(v)
        v -= u.dot(u.T.conjugate().dot(v))
    v /= la.norm(v)
    return v


def eliminate_nonfinite(v):
    nonfinite_flag = ~np.isfinite(v)
    cnt = np.sum(nonfinite_flag)
    if cnt > 0:
        print "Nonfinite components eliminated."
        v[nonfinite_flag] = np.zeros(cnt)
    return


def get_rand_pole(shifts):
    s = shifts[0] + (shifts[1] - shifts[0]) * rand(1)[0]
    return s


def pyamg_solver_shifting(solver, shift, matrix_hierarchy):
    for i in xrange(len(solver.levels)):
        solver.levels[i].A = matrix_hierarchy['a'][i] + shift * matrix_hierarchy['1'][i]
        
        
def copy_hierarchy(solver):
    matrix_hierarchy = {'a': [], '1': []}
    #matrix_hierarchy = {'a': [a], '1': [np.eye(a.shape[0])]}
    for i in xrange(len(solver.levels)):
        matrix_hierarchy['a'].append(solver.levels[i].A.copy()) 
        matrix_hierarchy['1'].append(sp.eye(solver.levels[i].A.shape[0], format='csr'))
    return matrix_hierarchy


def time_count_start(time_list):
    time_list.append(time())
    return


def time_count_finish(time_list):
    time_list[-1] = time() - time_list[-1]
    return


def direct_solver(a, id_n, s, w, solver_timing, solver_tol=1e-14):
    time_count_start(solver_timing)
    v = sla.spsolve(a - s * id_n, w, tol = solver_tol).reshape(-1, 1)
    time_count_finish(solver_timing)
    return v


def pyamg_solver(amg_hierarchy, orig_hierarchy, s, w, solver_timing, solver_tol=1e-14):
    solver_timing.append(time())
    pyamg_solver_shifting(amg_hierarchy, -s, orig_hierarchy)
    v = amg_hierarchy.solve(w, cycle = 'W', accel = 'gmres', tol = solver_tol, maxiter = 300).reshape(-1, 1)
    #v = amg_hierarchy.solve(w, tol = solver_tol).reshape(-1, 1)
    solver_timing[-1] = time() - solver_timing[-1]
    return v


def compute_lyapunov_solution(b):
    rhs = np.zeros(b.shape)
    rhs[0, 0] = 1.
    z_i = la.solve_lyapunov(b, rhs)
    return z_i


def mode_solver(mode, solver_data, shift, w):
    #safeguard stratagy with positive shifts:
    if shift.real > 0:
        shift = -shift.real + 1j * shift.imag
        
    #safeguard strategy with complex shifts:
    if math.fabs(shift.imag) < 1e-10 * math.fabs(shift.real) and la.norm(w.imag) < 1e-10 * la.norm(w):
        shift, w = shift.real, w.real
    
    time_count_start(solver_data['solver_timing'])
    if (mode == 'pyamg' or mode == 'pyamgE'): #and not np.any(solver_data['is_solver_direct']):
        #print 'Try pyamg'
        amg_hierarchy, orig_hierarchy = solver_data['amg_hierarchy'], solver_data['orig_hierarchy']
        pyamg_solver_shifting(amg_hierarchy, shift, orig_hierarchy)
        v = amg_hierarchy.solve(w, cycle = 'W', accel = 'gmres', tol = solver_data['solver_tol'], maxiter = 300).reshape(-1, 1)
        a, id_n = solver_data['a'], solver_data['id_n']
        if la.norm(a.dot(v) + shift * v - w) > solver_data['solver_tol'] * la.norm(w) * 1e5:
            print 'Fail, direct solver is used.', shift, la.norm(a.dot(v) + shift * v - w)
            v = sla.spsolve(a + shift * id_n, w).reshape(-1, 1)
            solver_data['is_solver_direct'].append(True)
        else:
            solver_data['is_solver_direct'].append(False)
    elif mode == 'splu':
        v = solver_data['lu'].solve(w).reshape(-1, 1)
        solver_data['is_solver_direct'].append(False)
    else:
        a, id_n = solver_data['a'], solver_data['id_n']
        v = sla.spsolve(a + shift * id_n, w).reshape(-1, 1)
        solver_data['is_solver_direct'].append(True)
    time_count_finish(solver_data['solver_timing'])
    return v


def mode_solver_initialisation(mode, solver_data, a, solver_tol):
    solver_data['solver_timing'], solver_data['solver_tol'] = [], solver_tol
    solver_data['is_solver_direct'] = []
    
    id_n = sp.eye(a.shape[0], format='csr')
    solver_data['a'] = a
    solver_data['id_n'] = id_n

    if mode == 'pyamg':
        amg_hierarchy = pyamg.ruge_stuben_solver(a)
        orig_hierarchy = copy_hierarchy(amg_hierarchy)
        solver_data['amg_hierarchy'], solver_data['orig_hierarchy'] = amg_hierarchy, orig_hierarchy
    elif mode == 'pyamgE':
        B = np.ones((a.shape[0],1), dtype=a.dtype);
        amg_hierarchy = pyamg.rootnode_solver(a, max_levels = 15, max_coarse = 300, coarse_solver = 'pinv', presmoother = ('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}), postsmoother = ('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}), BH = B.copy(), 
                                              strength = ('evolution', {'epsilon': 4.0, 'k': 2, 'proj_type': 'l2'}), aggregate ='standard', smooth = ('energy', {'weighting': 'local', 'krylov': 'gmres', 'degree': 1, 'maxiter': 2}), improve_candidates = [('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None, None, None, None, None, None, None, None, None, None, None, None, None, None]) 
        orig_hierarchy = copy_hierarchy(amg_hierarchy)
        solver_data['amg_hierarchy'], solver_data['orig_hierarchy'] = amg_hierarchy, orig_hierarchy
    elif mode == 'splu':
        solver_data['lu'] = sla.splu(a.tocsc())
    return


def update_b_alr_way(au, u, b):
    k = u.shape[1] - 2
    b[k, k-1] = u[:, -2].dot(au[:, k-1])
    b[k+1, k-1] = u[:, -1].dot(au[:, k-1])

    b[:k+2, k] = u.T.dot(au[:, k:k+1]).flatten()
    b[:k+2, k+1] = u.T.dot(au[:, k+1:k+2]).flatten()
    return b[:k+2, :k+2]


def update_b_full_way(au, u, b, double_flag=True, k=0):
    if k==0: k = u.shape[1]
    if double_flag:
        update_angle_in_b(u, au, b, k-2)
        update_angle_in_b(u, au, b, k-1)
    else:
        update_angle_in_b(u, au, b, k-1)
        
    #if k % 5 == 0:
    #    b_true = u.T.conjugate().dot(au[:, :u.shape[1]])
    #    resnorm = la.norm(b[:k, :k] - b_true)
    #    if resnorm > 1e-8 * la.norm(b_true):
    #        print 'Bad B', resnorm
    #        b[:k, :k] = b_true        
    return b[:k, :k]    


def update_angle_in_b(u, au, b, start_k):
    if la.norm(b[start_k, :start_k+1]) == 0:
        b[start_k, :start_k+1] = u[:, start_k].T.conjugate().dot(au[:, :start_k+1])
    if la.norm(b[:start_k, start_k]) == 0:
        b[:start_k, start_k] = u[:, :start_k].T.conjugate().dot(au[:, start_k])
    return


def update_res_au(res_au, u, au, b, double_flag=True, alr_way=False):
    k = u.shape[1]
    res_au[:, :k] = au[:, :k] - u.dot(b[:k, :k])
    if alr_way and k > 0 and la.norm(res_au[:, :k-1]) > 1e-10: 
        res_au[:, :k] -= u.dot(u.T.dot(res_au[:, :k]))
        if la.norm(res_au[:, :k-1]) > 1e-6:
            #print 'Bad res_AU', la.norm(res_au[:, :k-1])
            pass
    return res_au
    
    if double_flag:
        res_au[:, k-2:k] = au[:, k-2:k] - u[:, :k].dot(b[:k, k-2:k])
        res_au[:, :k-2] -= u[:, -2:].dot(b[k-2:k, :k-2])
        #res_au[:, k-2:k] = au[:, k-2:k] - u[:, :k-2].dot(b[:k-2, k-2:k])
        #res_au[:, :k] -= u[:, -2:].dot(b[k-2:k, :k])
    else:
        res_au[:, k-1:k] = au[:, k-1:k] - u.dot(b[:k, k-1:k])
        res_au[:, :k-1] -= u[:, -1:].dot(b[k-1:k, :k-1])

    #res_au[:, k-2:k] -= u.dot(u.T.dot(res_au[:, k-2:k]))
    #res_au[:, :k] -= u[:, -2:-1].dot(u[:, -2:-1].T.dot(res_au[:, :k]))
    #res_au[:, :k] -= u[:, -1:].dot(u[:, -1:].T.dot(res_au[:, :k]))
    return res_au
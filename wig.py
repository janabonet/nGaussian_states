# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:33:55 2024

@author: bencheikh-adm
"""

import numpy as np
from numpy.polynomial.hermite import hermval
from scipy import integrate
import sympy as sp
from qutip import *
from tqdm import tqdm
import math
import os
import csv
import pandas as pd
from joblib import Parallel, delayed

def save_to_csv(file_path, headers, data):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

def x_n(x, n):
    coefs = [0] * n
    coefs.append(1)
    H_pol = hermval(x, coefs)
    inner_prod = 1 / (math.sqrt(math.factorial(n) * 2**n)) * math.pi**(-1/4) * np.exp(-x**2 / 2) * H_pol
    return inner_prod

def compute_x_n_cache(x_values, n_values, x_n_cache):
    for x in x_values:
        for n in n_values:
            if (x, n) not in x_n_cache:
                x_n_cache[(x, n)] = x_n(x, n)

def precompute_values(x1vec, nu1vec, nu2vec, nvec):
    x_values_to_cache = set()

    for x1 in x1vec:
        for nu1 in nu1vec:
            x_values_to_cache.add(x1 + nu1 / 2)
            x_values_to_cache.add(x1 - nu1 / 2)

    for nu2 in nu2vec:
        x_values_to_cache.add(-nu2 / 2)
        x_values_to_cache.add(nu2 / 2)

    x_n_cache = {}
    compute_x_n_cache(x_values_to_cache, nvec, x_n_cache)
    return x_n_cache

def compute_sumtot(nvec, uvec, mvec, wvec, rho_out, x1, nu1, nu2, x_n_cache):
    sumtot = 0
    for n in nvec:
        psin = x_n_cache[(x1 - nu1 / 2, n)]
        for u in uvec:
            psiu = x_n_cache[(-nu2 / 2, u)]
            for m in mvec:
                psim = x_n_cache[(x1 + nu1 / 2, m)]
                for w in wvec:
                    psiw = x_n_cache[(nu2 / 2, w)]
                    sumtot = sumtot + psin * psim * psiu * psiw * rho_out[n*Np+u, m*Np+w]
    return sumtot

def wigner_x1(x1):
    W_x1p1_x1 = np.zeros(len(p1vec))
    pp1 = 0
    for p1 in p1vec:
        W_x1p1x2p2_nu1 = np.zeros(len(nu1vec))
        jj = 0
        for nu1 in nu1vec:
            W_x1p1x2p2_nu1nu2 = Parallel(n_jobs=-1)(delayed(compute_sumtot)(
                nvec, uvec, mvec, wvec, rho_out, x1, nu1, nu2, x_n_cache) for nu2 in nu2vec)
            
            W_x1p1x2p2_nu1[jj] = integrate.simpson(W_x1p1x2p2_nu1nu2, x=nu2vec) * np.exp(1j*nu1*p1)
            jj += 1
        W_x1p1_x1[pp1] = 1 / (2 * math.pi)**2 * integrate.simpson(W_x1p1x2p2_nu1, x=nu1vec) 
        # print("x = " + str(x1) + ", p = " + str(p1) + ", W_x1p1 = " + str(W_x1p1_x1[pp1]))
        pp1 += 1
    return W_x1p1_x1

'''STATE'''
Np = 10
phi1 = 0
phi2 = np.pi
psi_in = tensor(basis(Np,0),basis(Np,0)) #input for structure 1 and 2
psi_in1 = (tensor(basis(Np,2),basis(Np,0))).unit() #input for structure 3
psi_in2 = (tensor(basis(Np,0),basis(Np,2))).unit() #input for structure 3

a1_op = tensor(destroy(Np),qeye(Np))
a2_op = tensor(qeye(Np),destroy(Np))

rr = 0.5

s1 = tensor(squeeze(Np,-rr*np.exp(1j*phi1)),qeye(Np))
s2 = tensor(qeye(Np),squeeze(Np,-rr*np.exp(1j*phi2)))

psi_out_struct1 = (a1_op*s1*s2*psi_in + s1*a2_op*s2*psi_in).unit() #structure 1
psi_out_struct2 = (a1_op*a1_op*s1*a2_op*s2*psi_in + a1_op*s1*a2_op*a2_op*s2*psi_in).unit() #structure 2
psi_out_struct3 = (0.5*(np.sinh(rr)**2)*(squeezing(a1_op,a2_op,-2*rr)*psi_in1 - squeezing(a1_op,a2_op,-2*rr)*psi_in2)).unit() #structure 3

psi_out = psi_out_struct3
# psi_out = tensor(basis(Np,0),basis(Np,0))
rho_out = ket2dm(psi_out)

vecs = np.arange(-3,3,0.2)

x1vec = vecs
p1vec = vecs
nu1vec = vecs
nu2vec = vecs

W_x1p1 = np.zeros((len(x1vec),len(p1vec)))

x2 = 0
p2 = 0

fock_vec = list(range(0,Np)) #list(range(1,Np+1))
nvec = fock_vec
mvec = fock_vec
uvec = fock_vec
wvec = fock_vec

# Precompute x_n values for all necessary combinations
x_values_to_cache = set()
x_n_cache = precompute_values(x1vec, nu1vec, nu2vec, nvec)

results = []
for x1 in tqdm(x1vec):
    W_x1p1_x1 = wigner_x1(x1)
    results.append(W_x1p1_x1)

# Convert results to numpy array
results = np.array(results)


save_path = './'
if not os.path.exists(save_path):
    os.makedirs(save_path)

wig_path = os.path.join(save_path, "wigner_s3.csv")

df = pd.DataFrame(results, index=x1vec, columns=p1vec)

df.to_csv(wig_path)

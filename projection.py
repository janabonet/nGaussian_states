# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:12:23 2024

@author: janal
"""

import numpy as np
from numpy.polynomial.hermite import hermval
from qutip import *
from math import factorial
from tqdm import tqdm
import matplotlib.pyplot as plt

def save_to_csv(file_path, headers, data):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

def x_sq(r,x):
    sigma0 = 1
    Np = 20
    phi = 0
    
    summ = 0 
    for n in range(Np+1):
        
        beta = (np.exp(1j*n*phi)*np.tanh(r)**n)/(factorial(n)*2**(2*n))
        
        coefs = np.zeros(2*n+1)
        coefs[-1] = 1
        herm = hermval(x/(sigma0*np.sqrt(2)),coefs)
        
        summ = summ + beta*herm

    alpha = np.exp(-x**2/(4*sigma0**2))/(np.sqrt(np.sqrt(2*np.pi)*sigma0*np.cosh(r)))
    
    amplitud = alpha * summ
    return amplitud

def x_sq_min(r,x):
    sigma0 = 1
    Np = 10
    phi = 0
    
    summ = 0 
    for n in range(1,Np+1):
        
        beta = (np.exp(1j*n*phi)*np.tanh(r)**n)/(factorial(n-1)*2**(n-1)*np.sqrt(2**(2*n-1)))
        
        coefs = np.zeros(2*n)
        coefs[-1] = 1
        herm = hermval(x/(sigma0*np.sqrt(2)),coefs)
        
        summ = summ + beta*herm

    alpha = np.exp(-x**2/(4*sigma0**2))/(np.sqrt(np.sqrt(2*np.pi)*sigma0*np.cosh(r)))
    
    amplitud = alpha * summ # should be 0
    return amplitud
    
def x_sq_min2(r,x):
    sigma0 = 1
    Np = 9
    phi = 0
    
    summ = 0 
    for n in range(2,Np+1):
        
        # print(n)
        beta = (np.exp(1j*n*phi)*np.tanh(r)**n*np.sqrt(factorial(2*n-1))*np.sqrt(2*n-1))/(factorial(n-1)*2**(n-1)*np.sqrt(factorial(2*n-2)*2**(2*n-1)))

        coefs = np.zeros(2*n-2+1)
        coefs[-1] = 1
        herm = hermval(x/(sigma0*np.sqrt(2)),coefs)
        
        summ = summ + beta*herm

    alpha = np.exp(-x**2/(4*sigma0**2))/(np.sqrt(np.sqrt(2*np.pi)*sigma0*np.cosh(r)))
    
    amplitud = alpha * summ
    return amplitud

def singleph_sq(r):
    return 0
    
def singleph_sq_min(r):
    phi = 0
    return (np.exp(1j*phi)*np.tanh(r))/(np.sqrt(np.cosh(r)))
    
def singleph_sq_min2(r):
    return 0


sqvec = np.arange(0.1,0.5,0.05)
proj_sq = np.zeros(len(sqvec))
proj_sq_min = np.zeros(len(sqvec))
proj_sq_min2 = np.zeros(len(sqvec))
spd_sq_min = np.zeros(len(sqvec))

ii = 0
x = 0
for rr in tqdm(sqvec): 
    proj_sq[ii] = x_sq(rr, x)
    proj_sq_min[ii] = x_sq_min(rr, x)
    proj_sq_min2[ii] = x_sq_min2(rr, x)
    
    spd_sq_min[ii] = singleph_sq_min(rr)
    ii += 1


plt.figure(1)
plt.plot(sqvec,proj_sq, label = 'sq')
plt.plot(sqvec,proj_sq_min, label = 'sq_min')
plt.plot(sqvec,proj_sq_min2, label = 'sq_min2')
plt.plot(sqvec,spd_sq_min,label = 'spd_sq')
plt.legend()
plt.tight_layout()


# Np = 50
# rr = 0.5
# s = squeeze(Np,-rr*np.exp(1j*0))
# psi_in = basis(Np,0)
# psi_in = destroy(Np)*destroy(Np)*s*psi_in

# plt.figure(2)
# plot_fock_distribution(psi_in);



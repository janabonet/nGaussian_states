# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:20:59 2024

@author: janal
"""

import numpy as np
from qutip import *
from scipy import *
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import os
import csv
from tqdm import tqdm
import time

start_time = time.time()

plt.rc('text', usetex=False)

def save_to_csv(file_path, headers, data):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)
        
class numin:
    def __init__(self, psi_out, rr):
        self.psi_out = psi_out
        self.rr = rr
        # self.k = k
        # self.l = l
        # self.n = n
        
    def fm_av(self, x, m):
        # x (subsystem): 0 --> NA, 1 --> NB
        # m (subsystem order, k/l)
        
        NA = a.dag()*a
        NB = b.dag ()*b
    
        '''OMEGA_AVERAGE'''
        NA_av = expect(NA,self.psi_out)
        NA2_av = expect(NA*NA,self.psi_out)
        NA3_av = expect(NA*NA*NA,self.psi_out)
        NA4_av = expect(NA*NA*NA*NA,self.psi_out)
        NA5_av = expect(NA*NA*NA*NA*NA,self.psi_out)
        
        NB_av = expect(NB,self.psi_out)
        NB2_av = expect(NB*NB,self.psi_out)
        NB3_av = expect(NB*NB*NB,self.psi_out)
        NB4_av = expect(NB*NB*NB*NB,self.psi_out)
        NB5_av = expect(NB*NB*NB*NB*NB,self.psi_out)
        
        if x == 0: 
            Nx = NA_av
            Nx2 = NA2_av
            Nx3 = NA3_av
            Nx4 = NA4_av
            Nx5 = NA5_av
        else:
            Nx = NB_av
            Nx2 = NB3_av
            Nx3 = NB3_av
            Nx4 = NB4_av
            Nx5 = NB5_av
        
        if m == 1:
            fm = 1/2
        elif m == 2:
            fm = 2*Nx +1
        elif m == 3:
            fm = 0.5*(9*Nx2 + 9*Nx + 6)
        elif m == 4:
            fm =  8*Nx3 + 12*Nx2 + 28*Nx + 12
        elif m == 5:
            fm = 25/2*Nx4 + 25*Nx3 + 275/2*Nx2 + 125*Nx + 60
        elif m == 6:
            fm = 18*Nx5 + 45*Nx4 + 480*Nx3 + 675*Nx2 + 942*Nx + 360
        return fm 

    def adiag(self, values):
        size = len(values)
        adiag_mat = np.zeros((size,size),dtype = float)
        for i, value in enumerate(values):
            adiag_mat[i,-1-i] = value
        return adiag_mat

    def find_eigmin(self, k, l, n):
        QA_nk = 0.5*(a**(k*n) + a.dag()**(k*n))
        QB_nl = 0.5*(b**(l*n) + b.dag()**(l*n))
        PA_nk = 1j * 0.5 * (a.dag()**(k*n) - a**(k*n))
        PB_nl = 1j * 0.5 * (b.dag()**(l*n) - b**(l*n))  
        
        print('QA = ' + str(expect(QA_nk,self.psi_out)))
        print('QB = ' + str(expect(QB_nl,self.psi_out)))
        print('PA = ' + str(expect(PA_nk,self.psi_out)))
        print('PB = ' + str(expect(PB_nl,self.psi_out)))
        
        means = [expect(QA_nk,self.psi_out),expect(QB_nl,self.psi_out), expect(PA_nk,self.psi_out), expect(PB_nl,self.psi_out) ]
        
        return np.array(means)

'''STATE'''
Np = 50 
phi1 = 0
phi2 = np.pi
psi_in = tensor(basis(Np,0),basis(Np,0)) #input for structure 1 and 2
psi_in1 = (tensor(basis(Np,2),basis(Np,0))).unit() #input for structure 3
psi_in2 = (tensor(basis(Np,0),basis(Np,2))).unit() #input for structure 3

a1 = tensor(destroy(Np),qeye(Np))
a2 = tensor(qeye(Np),destroy(Np))

'''PARAMETERS'''
k = 1 # a_dag operator order
l = 1 # b_dag operator order
n = 2 # hierarchy index

num_states = Np

a = tensor(destroy(num_states),qeye(num_states))
b = tensor(qeye(num_states),destroy(num_states))

'''NU_ref_min vs rr'''

sqvec = np.arange(0.1,1.2,0.1)
quadmean_struct1 = np.zeros((len(sqvec),4))
quadmean_struct2 = np.zeros((len(sqvec),4))
quadmean_struct3 = np.zeros((len(sqvec),4))

ii = 0

for rr in tqdm(sqvec): 
    s1 = tensor(squeeze(Np,-rr*np.exp(1j*phi1)),qeye(Np))
    s2 = tensor(qeye(Np),squeeze(Np,-rr*np.exp(1j*phi2)))
    
    psi_out_struct1 = (a1*s1*s2*psi_in + s1*a2*s2*psi_in).unit() #structure 1
    psi_out_struct2 = (a1*a1*s1*a2*s2*psi_in + a1*s1*a2*a2*s2*psi_in).unit() #structure 2
    psi_out_struct3 = (0.5*(np.sinh(rr)**2)*(squeezing(a1,a2,-2*rr)*psi_in1 - squeezing(a1,a2,-2*rr)*psi_in2)).unit() #structure 3
    
    numin1 = numin(psi_out_struct1, rr)
    quadmean_struct1[ii] = numin1.find_eigmin(k, l, n)
    
    numin2 = numin(psi_out_struct2, rr)
    quadmean_struct2[ii] = numin2.find_eigmin(k, l, n)
    
    numin3 = numin(psi_out_struct3, rr)
    quadmean_struct3[ii] = numin3.find_eigmin(k, l, n)
    
    ii += 1

end_time = time.time()
running_time = end_time - start_time
print("Script running time: {:.2f} seconds".format(running_time))

'''CSV FILES '''    
save_path = './quadratures_mean'
if not os.path.exists(save_path):
    os.makedirs(save_path)

quadmean_rr_path_struct1 = os.path.join(save_path, "quadmean_rr_struct1_" + str(k) + str(l) + str(n) + ".csv")
save_to_csv(quadmean_rr_path_struct1, ['rr', 'qa_mean','qb_mean','pa_mean','pb_mean'], list(zip(sqvec,quadmean_struct1[:,0],quadmean_struct1[:,1],quadmean_struct1[:,2],quadmean_struct1[:,3])))

quadmean_rr_path_struct2 = os.path.join(save_path, "quadmean_rr_struct2_" + str(k) + str(l) + str(n) + ".csv")
save_to_csv(quadmean_rr_path_struct2, ['rr', 'qa_mean','qb_mean','pa_mean','pb_mean'], list(zip(sqvec,quadmean_struct2[:,0],quadmean_struct2[:,1],quadmean_struct2[:,2],quadmean_struct2[:,3])))

quadmean_rr_path_struct3 = os.path.join(save_path, "quadmean_rr_struct3_" + str(k) + str(l) + str(n) + ".csv")
save_to_csv(quadmean_rr_path_struct3, ['rr', 'qa_mean','qb_mean','pa_mean','pb_mean'], list(zip(sqvec,quadmean_struct3[:,0],quadmean_struct3[:,1],quadmean_struct3[:,2],quadmean_struct3[:,3])))

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:15:31 2024

@author: valere.sautel
"""

import numpy as np
from qutip import *
from scipy import *
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import csv
from tqdm import tqdm

plt.rc('text', usetex=False)

def commut(x, m):
    # x (subsystem): 0 --> NA, 1 --> NB
    # m (subsystem order, k/l)
    b = tensor(qeye(num_states),destroy(num_states))
    if x == 0: 
        Nx = NA_av
        Nx2 = NA2_av
        Nx3 = NA3_av
        Nx4 = NA4_av
        Nx5 = NA5_av
    else:
        Nx = NB_av
        Nx2 = NB2_av
        Nx3 = NB3_av
        Nx4 = NB4_av
        Nx5 = NB5_av
    
    if m == 1:
        fm = 1/2
    elif m == 2:
        fm = 2*Nx +1
    elif m == 3:
        # fm = 0.5*(9*Nx2 + 9*Nx + 6)
        # fm = 2*b.dag()*Nx*b + 8*Nx2 + 10*Nx+4
        fm = 2*NB3_comm_av + 8*Nx2 + 10*Nx+4
    elif m == 4:
        # fm =  8*Nx3 + 12*Nx2 + 28*Nx + 12
        fm = 1
    elif m == 5:
        fm = 25/2*Nx4 + 25*Nx3 + 275/2*Nx2 + 125*Nx + 60
    elif m == 6:
        fm = 18*Nx5 + 45*Nx4 + 480*Nx3 + 675*Nx2 + 942*Nx + 360
    return fm 

def save_to_csv(file_path, headers, data):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

def adiag(values):
    size = len(values)
    adiag_mat = np.zeros((size,size),dtype = complex)
    for i, value in enumerate(values):
        adiag_mat[i,-1-i] = value
    return adiag_mat

# '''STATE'''
# Np = 120     # Np
# Mean_Photon_Number = 25
# alpha_p = np.sqrt(Mean_Photon_Number)

# times = np.linspace(0.0,1,80)
    
# ket_p = coherent(Np,alpha_p) # pump
# ket_a = coherent(Np,0) # mode a
# ket_b = coherent(Np,0) # mode b

# psi_in = tensor(ket_p,ket_a,ket_b)


'''STATE'''
Np = 80
phi1 = 0
phi2 = np.pi
psi_in = tensor(basis(Np,0),basis(Np,0)) #input for structure 1 and 2
psi_in1 = (tensor(basis(Np,2),basis(Np,0))).unit() #input for structure 3
psi_in2 = (tensor(basis(Np,0),basis(Np,2))).unit() #input for structure 3

a1 = tensor(destroy(Np),qeye(Np))
a2 = tensor(qeye(Np),destroy(Np))



'''PARAMETERS AND OPERATORS'''
k = 1 # a_dag operator order
l = 1 # b_dag operator order
n = 1   # hierarchy index

# p = tensor(destroy(Np),qeye(Np),qeye(Np))
# a = tensor(qeye(Np),destroy(Np),qeye(Np))
# b = tensor(qeye(Np),qeye(Np),destroy(Np))

num_states = Np
a = tensor(destroy(num_states),qeye(num_states))
b = tensor(qeye(num_states),destroy(num_states))

QA_nk = 0.5*((a**k)**n + (a.dag()**k)**n)
QB_nl = 0.5*((b**l)**n + (b.dag()**l)**n)

PA_nk = 1j * 0.5 * (a.dag()**(k*n) - a**(k*n))
PB_nl = 1j * 0.5 * (b.dag()**(l*n) - b**(l*n))  

NA = a.dag()*a
NB = b.dag()*b
            
xivec = np.arange(0.01,1.2,0.02)
scm = np.zeros(len(xivec))
ii = 0
for xii in tqdm(xivec):
    
    # time = 1
    # kappa = xii/(alpha_p*time)
    # H = 1j*kappa*((a.dag()**k)*(b.dag()**l)*p - (a**k)*(b**l)*p.dag())
    # # mc_result = mcsolve(H_fun,psi_in,times,args = xii, options=options)
    # mc_result = mcsolve(H,psi_in,times)
    # # mc_result = sesolve(H,psi_in,times)
    # psi_out = mc_result.states[-1]
    
    rr = xii
    s1 = tensor(squeeze(Np,-rr*np.exp(1j*phi1)),qeye(Np))
    s2 = tensor(qeye(Np),squeeze(Np,-rr*np.exp(1j*phi2)))
    
    psi_out_struct1 = (a1*s1*s2*psi_in + s1*a2*s2*psi_in).unit() #structure 1
    psi_out_struct2 = (a1*a1*s1*a2*s2*psi_in + a1*s1*a2*a2*s2*psi_in).unit() #structure 2
    # psi_out_struct3 = (0.5*(np.sinh(rr)**2)*(squeezing(a1,a2,-2*rr)*psi_in1 - squeezing(a1,a2,-2*rr)*psi_in2)).unit() #structure 3
    
    psi_out = psi_out_struct1

    '''OMEGA_AVERAGE'''
    
    NA_av = expect(NA,psi_out)
    NA2_av = expect(NA*NA,psi_out)
    NA3_av = expect(NA*NA*NA,psi_out)
    NA4_av = expect(NA*NA*NA*NA,psi_out)
    NA5_av = expect(NA*NA*NA*NA*NA,psi_out)
    
    NB_av = expect(NB,psi_out)
    NB2_av = expect(NB*NB,psi_out)
    NB3_av = expect(NB*NB*NB,psi_out)
    NB4_av = expect(NB*NB*NB*NB,psi_out)
    NB5_av = expect(NB*NB*NB*NB*NB,psi_out)
    
    NB3_comm_av = expect(b.dag()*NB*b,psi_out)
    
    # QA = commut(0,k*n) # QA = 0.5
    QB = commut(1,l*n) # QB = (1+2*NB2_av)
    QB = expect(QB,psi_out)/2
    

    JA = adiag([0,0])
    JB = adiag([QB,-QB]) 

    # JA = adiag([QA,-QA])
    # JB = adiag([0,0])
    
    ab = 1
                
    Omega_kl_av = block_diag(JA, JB)
    
    '''H.O. COVARIANCE MATRIX'''
    # R_vec = [QA_nk, PA_nk, QB_nl, PB_nl]
    
    V_kl = np.zeros((len(Omega_kl_av),len(Omega_kl_av)),dtype=complex)

    V_kl[0,0] = 0.5*expect(2*QA_nk**2,psi_out) - expect(QA_nk,psi_out)**2
    V_kl[1,1] = 0.5*expect(2*PA_nk**2,psi_out) - expect(PA_nk,psi_out)**2
    V_kl[2,2] = 0.5*expect(2*QB_nl**2,psi_out) - expect(QB_nl,psi_out)**2
    V_kl[3,3] = 0.5*expect(2*PB_nl**2,psi_out) - expect(PB_nl,psi_out)**2
    
    V_kl[0,1] = 0.5*expect(QA_nk*PA_nk + PA_nk*QA_nk,psi_out) - expect(QA_nk,psi_out)*expect(PA_nk,psi_out)
    V_kl[1,0] = V_kl[0,1]
    V_kl[2,0] = 0.5*expect(QA_nk*QB_nl + QB_nl*QA_nk,psi_out) - expect(QB_nl,psi_out)*expect(QA_nk,psi_out)  
    V_kl[0,2] = V_kl[2,0]
    V_kl[3,0] = 0.5*expect(QA_nk*PB_nl + PB_nl*QA_nk,psi_out) - expect(PB_nl,psi_out)*expect(QA_nk,psi_out)
    V_kl[0,3] = V_kl[3,0]
    V_kl[2,1] = 0.5*expect(QB_nl*PA_nk + PA_nk*QB_nl,psi_out) - expect(QB_nl,psi_out)*expect(PA_nk,psi_out)
    V_kl[1,2] = V_kl[2,1]
    V_kl[3,1] = 0.5*expect(PB_nl*PA_nk + PA_nk*PB_nl,psi_out) - expect(PB_nl,psi_out)*expect(PA_nk,psi_out)
    V_kl[1,3] = V_kl[3,1]
    V_kl[3,2] = 0.5*expect(PB_nl*QB_nl + QB_nl*PB_nl,psi_out) - expect(QB_nl,psi_out)*expect(QB_nl,psi_out)
    V_kl[2,3] = V_kl[3,2]   

        
    '''CONDICIÓ FINAL'''
    S_kl_ref = V_kl + 1j*0.5* Omega_kl_av
    nu_kl_ref = np.linalg.eigvals(S_kl_ref) # symplectic eigenvalues of S_kl_ref
    scm[ii] = np.real(np.min(nu_kl_ref))
    ii += 1


'''CSV FILES '''    
save_path = './steering_cm_results'
if not os.path.exists(save_path):
    os.makedirs(save_path)

scm_path_struct1 = os.path.join(save_path, "scm_conv_s1_" + str(k) + str(l) + str(n) + '_' + str(ab) + ".csv")
save_to_csv(scm_path_struct1, ['xi', 'scm'], list(zip(xivec, scm)))

# scm_path_struct2 = os.path.join(save_path, "scm_s2_" + str(k) + str(l) + str(n) + '_' + str(ab) + ".csv")
# save_to_csv(scm_path_struct2, ['xi', 'scm'], list(zip(xivec, scm)))

# scm_path_struct3 = os.path.join(save_path,"scm_s3_" + str(k) + str(l) + str(n) + '_' + str(ab) + ".csv")
# save_to_csv(scm_path_struct3, ['xi', 'scm'], list(zip(xivec, scm)))

plt.figure(1)
plt.plot(xivec,scm)
plt.axhline(0, color='red', linestyle='--')
plt.title("Structure 1, k = " + str(k) + ', l = ' + str(l) + ', n = ' + str(n))
plt.ylabel(r"$S_{CM}$")
plt.xlabel(r'$\xi$')
plt.tight_layout()
plt.savefig(os.path.join(save_path, "scm_conv_s1_" + str(k) + str(l) + str(n) + '_' + str(ab) + ".png")) 
plt.show()

# plt.figure(2)
# plt.plot(xivec,scm)
# plt.title("Structure 2, k = " + str(k) + ', l = ' + str(l) + ', n = ' + str(n))
# plt.ylabel(r"$S_{cm}$")
# plt.xlabel(r'$\xi$')
# plt.tight_layout()
# plt.axhline(0, color='red', linestyle='--')
# plt.savefig(os.path.join(save_path, "scm_s2_" + str(k) + str(l) + str(n) + '_' + str(ab) + ".png")) 
# plt.show()

# plt.figure(3)
# plt.plot(xivec,scm)
# plt.title("Structure 3, k = " + str(k) + ', l = ' + str(l) + ', n = ' + str(n))
# plt.ylabel(r"$S_{cm}$")
# plt.xlabel(r'$\xi$')
# plt.tight_layout()
# plt.axhline(0, color='red', linestyle='--')
# plt.savefig(os.path.join(save_path, "scm_s3_" + str(k) + str(l) + str(n) + '_' + str(ab) + ".png")) 
# plt.show()
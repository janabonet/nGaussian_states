# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:34:42 2024

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
    def __init__(self, psi_out, rr, Np):
        self.psi_out = psi_out
        self.rr = rr
        self.Np = Np
        
    def fm_av(self, x, m):
        # x (subsystem): 0 --> NA, 1 --> NB
        # m (subsystem order, k/l)
        
        num_states = self.Np
        
        a = tensor(destroy(num_states),qeye(num_states))
        b = tensor(qeye(num_states),destroy(num_states))
        
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
            Nx2 = NB2_av
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
        num_states = self.Np
        
        a = tensor(destroy(num_states),qeye(num_states))
        b = tensor(qeye(num_states),destroy(num_states))
        
        QA_nk = 0.5*(a**(k*n) + a.dag()**(k*n))
        QB_nl = 0.5*(b**(l*n) + b.dag()**(l*n))
        PA_nk = 1j * 0.5 * (a.dag()**(k*n) - a**(k*n))
        PB_nl = 1j * 0.5 * (b.dag()**(l*n) - b**(l*n))  
        
        JA = self.adiag([self.fm_av(0,k*n),-self.fm_av(0,k*n)])
        JB = self.adiag([self.fm_av(1,l*n),-self.fm_av(1,l*n)])
                    
        Omega_kl_av = block_diag(JA, JB)
        
        '''H.O. COVARIANCE MATRIX'''
        R_vec = [QA_nk, PA_nk, QB_nl, PB_nl]
        
        V_kl = np.zeros((len(Omega_kl_av),len(Omega_kl_av)),dtype=complex)
        for i in range(4):
            for j in range(4):
                if i==j:
                    V_kl[i,j] = variance(R_vec[i], self.psi_out)
                elif ((i==0 and j==1) or (i==1 and j==0)):
                    V_kl[i,j] = 0.5 *(expect(QA_nk*PA_nk,self.psi_out) + expect(PA_nk*QA_nk,self.psi_out))
                elif ((i==2 and j==3) or (i==3 and j==2)):
                    V_kl[i,j] = 0.5 *(expect(QB_nl*PB_nl,self.psi_out) + expect(PB_nl*QB_nl,self.psi_out))
                    # elif ((i==0 and j==1) or (i==1 and j==0) or (i==2 and j==3) or (i==3 and j==2)):
                    #     V_kl[i,j] =0.5 * (expect(R_vec[i]*R_vec[j], psi_out) + expect(R_vec[j]*R_vec[i], psi_out))
                    # else:
                    #     V_kl[i,j] = expect(R_vec[i]*R_vec[j],psi_out)
                    
        # V_kl[0,2] = expect(QB_nl*QA_nk,self.psi_out)  
        V_kl[0,2] = expect(QB_nl*QA_nk,self.psi_out) - expect(QB_nl,self.psi_out)*expect(QA_nk,self.psi_out)
        V_kl[0,3] = expect(QA_nk*PB_nl,self.psi_out) - expect(PB_nl,self.psi_out)*expect(QA_nk,self.psi_out)
        V_kl[1,2] = expect(PA_nk*QB_nl,self.psi_out) - expect(QB_nl,self.psi_out)*expect(PA_nk,self.psi_out)
        V_kl[1,3] = expect(PA_nk*PB_nl,self.psi_out) - expect(PB_nl,self.psi_out)*expect(PA_nk,self.psi_out)
        
        V_kl[2,0] = V_kl[0,2]
        V_kl[2,1] = V_kl[1,2]
        V_kl[3,0] = V_kl[0,3]
        V_kl[3,1] = V_kl[1,3]  
                
        LambdaB = np.diag([1,1,1,-1])
        V_kl_ref = np.matmul( np.matmul(LambdaB,V_kl), LambdaB)
        
        '''CONDICIÓ FINAL'''
        S_kl_ref = V_kl_ref + 1j*0.5* Omega_kl_av
        nu_kl_ref = np.linalg.eigvals(S_kl_ref) # symplectic eigenvalues of S_kl_ref
        eigmin = np.real(np.min(nu_kl_ref))
        return eigmin 

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
n = 4 # hierarchy index

# num_states = Np

# a = tensor(destroy(num_states),qeye(num_states))
# b = tensor(qeye(num_states),destroy(num_states))

'''NU_ref_min vs rr'''

sqvec = np.arange(0.01,1.2,0.05)
numin_struct1 = np.zeros(len(sqvec))
numin_struct2 = np.zeros(len(sqvec))
numin_struct3 = np.zeros(len(sqvec))
# numin_struct4 = np.zeros(len(sqvec))

ii = 0

for rr in tqdm(sqvec): 
    s1 = tensor(squeeze(Np,-rr*np.exp(1j*phi1)),qeye(Np))
    s2 = tensor(qeye(Np),squeeze(Np,-rr*np.exp(1j*phi2)))
    
    psi_out_struct1 = (a1*s1*s2*psi_in + s1*a2*s2*psi_in).unit() #structure 1
    psi_out_struct2 = (a1*a1*s1*a2*s2*psi_in + a1*s1*a2*a2*s2*psi_in).unit() #structure 2
    psi_out_struct3 = (0.5*(np.sinh(rr)**2)*(squeezing(a1,a2,-2*rr)*psi_in1 - squeezing(a1,a2,-2*rr)*psi_in2)).unit() #structure 3
    
    numin1 = numin(psi_out_struct1, rr, Np)
    numin_struct1[ii] = numin1.find_eigmin(k, l, n)
    
    numin2 = numin(psi_out_struct2, rr, Np)
    numin_struct2[ii] = numin2.find_eigmin(k, l, n)
    
    numin3 = numin(psi_out_struct3, rr, Np)
    numin_struct3[ii] = numin3.find_eigmin(k, l, n)
    

    
    # s1 = tensor(squeeze(Np,-rr*np.exp(1j*0)),qeye(Np),qeye(Np),qeye(Np))
    # s2 = tensor(qeye(Np),squeeze(Np,-rr*np.exp(1j*np.pi)),qeye(Np),qeye(Np))
    # s3 = tensor(qeye(Np),qeye(Np),squeeze(Np,-rr*np.exp(1j*0)),qeye(Np))
    # s4 = tensor(qeye(Np),qeye(Np),qeye(Np),squeeze(Np,-rr*np.exp(1j*np.pi)))
    # psi_in = s1*s2*s3*s4*psi_in
    # a_sqmin = (a1*a2*a3*psi_in + a1*a3*a3*psi_in + a1*a2*a4*psi_in + a1*a3*a4*psi_in).unit()
    # a_sq = (a2*a2*a3*psi_in + a2*a3*a3*psi_in + a2*a2*a4*psi_in + a2*a3*a4*psi_in).unit()
    # psi_out_struct4 = (a_sqmin + a_sq).unit()
    # numin4 = numin(psi_out_struct3, rr)
    # numin_struct4[ii] = numin4.find_eigmin(k, l, n)
    
    ii += 1

end_time = time.time()
running_time = end_time - start_time
print("Script running time: {:.2f} seconds".format(running_time))

'''CSV FILES '''    
save_path = './numin_results'
if not os.path.exists(save_path):
    os.makedirs(save_path)

numin_rr_path_struct1 = os.path.join(save_path, "numin_rr_struct1_" + str(k) + str(l) + str(n) + ".csv")
save_to_csv(numin_rr_path_struct1, ['rr', 'nu_min'], list(zip(sqvec, numin_struct1)))

numin_rr_path_struct2 = os.path.join(save_path, "numin_rr_struct2_" + str(k) + str(l) + str(n) + ".csv")
save_to_csv(numin_rr_path_struct2, ['rr', 'nu_min'], list(zip(sqvec, numin_struct2)))

numin_rr_path_struct3 = os.path.join(save_path, "numin_rr_struct3_" + str(k) + str(l) + str(n) + ".csv")
save_to_csv(numin_rr_path_struct3, ['rr', 'nu_min'], list(zip(sqvec, numin_struct3)))

'''FIGURES'''

plt.figure(1)
plt.plot(sqvec,numin_struct1)
plt.title("Structure 1, k = " + str(k) + ', l = ' + str(l) + ', n = ' + str(n))
plt.ylabel(r"$\nu_{-}$")
plt.xlabel(r'$\xi$')
plt.tight_layout()
plt.savefig(os.path.join(save_path, "numin_rr_struct1_" + str(k) + str(l) + str(n) + ".png")) 
plt.show()

plt.figure(2)
plt.plot(sqvec,numin_struct2)
plt.title("Structure 2, k = " + str(k) + ', l = ' + str(l) + ', n = ' + str(n))
plt.ylabel(r"$\nu_{-}$")
plt.xlabel(r'$\xi$')
plt.tight_layout()
plt.savefig(os.path.join(save_path, "numin_rr_struct2_" + str(k) + str(l) + str(n) + ".png")) 
plt.show()

plt.figure(3)
plt.plot(sqvec,numin_struct3)
plt.title("Structure 3, k = " + str(k) + ', l = ' + str(l) + ', n = ' + str(n))
plt.ylabel(r"$\nu_{-}$")
plt.xlabel(r'$\xi$')
plt.tight_layout()
plt.savefig(os.path.join(save_path, "numin_rr_struct3_" + str(k) + str(l) + str(n) + ".png")) 
plt.show()
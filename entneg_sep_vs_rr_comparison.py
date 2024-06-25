# -*- coding: utf-8 -*-

"""
Created on Tue Mar  5 09:51:45 2024

@author: janal
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from mpl_toolkits.mplot3d import Axes3D
from scipy import *
from qutip import *
import os
from tqdm import tqdm
# import sys
import csv

plt.rc('text', usetex=False)

def save_to_csv(file_path, headers, data):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)


def entanglement_negativity(rho):
    mask = [0]*len(rho.dims[0])
    mask[0] = 1
    rho_par = partial_transpose(rho,mask)
    x = rho_par.dag()*rho_par
    v = x.sqrtm()
    m = np.real(v.tr())
    # negativity_t = 0.5*(m-1)
    en = np.log2(m)
    return en

Np = 40
phi1 = 0
phi2 = np.pi

psi_in = tensor(basis(Np,0),basis(Np,0)) #input for structure 1 and 2
psi_in1 = (tensor(basis(Np,2),basis(Np,0))).unit() #input for structure 3
psi_in2 = (tensor(basis(Np,0),basis(Np,2))).unit() #input for structure 3
a1 = tensor(destroy(Np),qeye(Np))
a2 = tensor(qeye(Np),destroy(Np))

# psi_in = tensor(basis(Np,0),basis(Np,0),basis(Np,0),basis(Np,0))
# a1 = tensor(destroy(Np),qeye(Np),qeye(Np),qeye(Np))
# a2 = tensor(qeye(Np),destroy(Np),qeye(Np),qeye(Np))
# a3 = tensor(qeye(Np),qeye(Np),destroy(Np),qeye(Np))
# a4 = tensor(qeye(Np),qeye(Np),qeye(Np),destroy(Np))

'''GAUSSIAN ENTANGLEMENT (BIPARTITE CASE)'''
q1 = 0.5*(a1 + a1.dag())
q2 = 0.5*(a2 + a2.dag())
p1 = 1j * 0.5 * (a1.dag() - a1)
p2 = 1j * 0.5 * (a2.dag() - a2)
alpha = q1 + q2
beta = p1 - p2
# alpha = q1 - p2
# beta = p1 + q2

ii = 0
sqvec = np.arange(0.001,1,0.02)
entneg_struct1 = np.zeros(len(sqvec)) #array to store entanglement negativity values for structure 1
entneg_struct2 = np.zeros(len(sqvec))
entneg_struct3 = np.zeros(len(sqvec))
# entneg_struct4 = np.zeros(len(sqvec))

separability_struct1 = np.zeros(len(sqvec))
separability_struct2 = np.zeros(len(sqvec))
separability_struct3 = np.zeros(len(sqvec))
# separability_struct4 = np.zeros(len(sqvec))
for rr in tqdm(sqvec):
    # print(rr)    
    s1 = tensor(squeeze(Np,-rr*np.exp(1j*phi1)),qeye(Np))
    s2 = tensor(qeye(Np),squeeze(Np,-rr*np.exp(1j*phi2)))
    psi_out_struct1 = (a1*s1*s2*psi_in + s1*a2*s2*psi_in).unit() #structure 1
    psi_out_struct2 = (a1*a1*s1*a2*s2*psi_in + a1*s1*a2*a2*s2*psi_in).unit() #structure 2
    psi_out_struct3 = (0.5*(np.sinh(rr)**2)*(squeezing(a1,a2,-2*rr)*psi_in1 - squeezing(a1,a2,-2*rr)*psi_in2)).unit() #structure 3
    
    # s1 = tensor(squeeze(Np,-rr*np.exp(1j*0)),qeye(Np),qeye(Np),qeye(Np))
    # s2 = tensor(qeye(Np),squeeze(Np,-rr*np.exp(1j*np.pi)),qeye(Np),qeye(Np))
    # s3 = tensor(qeye(Np),qeye(Np),squeeze(Np,-rr*np.exp(1j*0)),qeye(Np))
    # s4 = tensor(qeye(Np),qeye(Np),qeye(Np),squeeze(Np,-rr*np.exp(1j*np.pi)))
    # psi_in = s1*s2*s3*s4*psi_in
    # a_sqmin = (a1*a2*a3*psi_in + a1*a3*a3*psi_in + a1*a2*a4*psi_in + a1*a3*a4*psi_in).unit()
    # a_sq = (a2*a2*a3*psi_in + a2*a3*a3*psi_in + a2*a2*a4*psi_in + a2*a3*a4*psi_in).unit()
    # psi_out_struct4 = (a_sqmin + a_sq).unit()

    # entneg_struct1[ii] = entanglement_negativity(ket2dm(psi_out_struct1))
    # entneg_struct2[ii] = entanglement_negativity(ket2dm(psi_out_struct2))
    # entneg_struct3[ii] = entanglement_negativity(ket2dm(psi_out_struct3))
    # entneg_struct4[ii] = entanglement_negativity(ket2dm(psi_out_struct4))
    
    separability_struct1[ii] = variance(alpha,psi_out_struct1) + variance(beta,psi_out_struct1)
    separability_struct2[ii] = variance(alpha,psi_out_struct2) + variance(beta,psi_out_struct2)
    separability_struct3[ii] = variance(alpha,psi_out_struct3) + variance(beta,psi_out_struct3)
    # separability_struct4[ii] = variance(alpha,psi_out_struct4) + variance(beta,psi_out_struct4)
    
    ii += 1

'''CSV FILES '''    
save_path = './separability_results'
if not os.path.exists(save_path):
    os.makedirs(save_path)
# save_dir = os.path.dirname(save_path)
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# path_struct1 = os.path.join(save_path, "ent_rr_struct1.csv")
# save_to_csv(path_struct1, ['rr', 'entanglement_negativity','separability'], list(zip(sqvec, entneg_struct1, separability_struct1)))

# path_struct2 = os.path.join(save_path, "ent_rr_struct2.csv")
# save_to_csv(path_struct2, ['rr', 'entanglement_negativity','separability'], list(zip(sqvec, entneg_struct2, separability_struct2)))

# path_struct3 = os.path.join(save_path, "ent_rr_struct3.csv")
# save_to_csv(path_struct3, ['rr', 'entanglement_negativity','separability'], list(zip(sqvec, entneg_struct3, separability_struct3)))

# path_struct1 = os.path.join(save_path, "sep_rr_struct1.csv")
# save_to_csv(path_struct1, ['rr', 'separability'], list(zip(sqvec, separability_struct1)))

# path_struct2 = os.path.join(save_path, "sep_rr_struct2.csv")
# save_to_csv(path_struct2, ['rr', 'separability'], list(zip(sqvec, separability_struct2)))

# path_struct3 = os.path.join(save_path, "sep_rr_struct3.csv")
# save_to_csv(path_struct3, ['rr', 'separability'], list(zip(sqvec, separability_struct3)))

# path_struct4 = os.path.join(save_path, "ent_rr_struct4.csv")
# save_to_csv(path_struct4, ['rr', 'entanglement_negativity'], list(zip(sqvec, entneg_struct3)))

# path_struct4 = os.path.join(save_path, "sep_rr_struct4.csv")
# save_to_csv(path_struct4, ['rr', 'separability'], list(zip(sqvec, separability_struct4)))


'''GENERACIÓ DE FIGURES'''    
plt.figure(1)
# plt.plot(sqvec,entneg_struct1, label = 'Entanglement negativity')
plt.plot(sqvec,separability_struct1, label = 'Separability (G)')
# plt.title('Structure 1, $\phi_1 = $' + str(phi1) + ', $\phi_2 = $' + str(phi2))
plt.title('Structure 1, $\phi_1 = $' + str(phi1) + ', $\phi_2 = \pi$')
# plt.ylabel("Entanglement negativity")
plt.xlabel(r'Squeezing factor (r)')
plt.legend()
plt.tight_layout()
# plt.savefig(os.path.join(save_path, "sep_struc1.png")) 
plt.show()

plt.figure(2)
# plt.plot(sqvec,entneg_struct2, label = 'Entanglement negativity')
plt.plot(sqvec,separability_struct2, label = 'Separability (G)')
plt.title('Structure 2, $\phi_1 = $' + str(phi1) + ', $\phi_2 = \pi$')
# plt.ylabel("Entanglement negativity")
plt.xlabel(r'Squeezing factor (r)')
plt.legend()
plt.tight_layout()
# plt.savefig(os.path.join(save_path, "sep_struc2.png")) 
plt.show()

plt.figure(3)
# plt.plot(sqvec,entneg_struct3, label = 'Entanglement negativity')
plt.plot(sqvec,separability_struct3, label = 'Separability (G)')
plt.title('Structure 3, $\phi_1 = $' + str(phi1) + ', $\phi_2 = \pi$')
# plt.ylabel("Entanglement negativity")s
plt.xlabel(r'Squeezing factor (r)')
plt.legend()
plt.tight_layout()
# plt.savefig(os.path.join(save_path, "sep_struc3.png")) 
plt.show()

# plt.figure(4)
# plt.plot(sqvec,entneg_struct4, label = 'Entanglement negativity')
# plt.plot(sqvec,separability_struct4, label = 'Separability (G)')
# plt.title('Structure 3, $\phi_1 = $' + str(phi1) + ', $\phi_2 = \pi$')
# # plt.ylabel("Entanglement negativity")s
# plt.xlabel(r'Squeezing factor (r)')
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(save_path, "sep_struc4.png")) 
# plt.show()







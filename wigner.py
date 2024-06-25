import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from mpl_toolkits.mplot3d import Axes3D
from scipy import *
from qutip import *

plt.rc('text', usetex=False)

def plot_wigner_2d_3d(psi):
    fig = plt.figure(figsize=(17, 8))

    ax = fig.add_subplot(1, 2, 1)
    plot_wigner(psi, fig=fig, ax=ax, alpha_max=6)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    plot_wigner(psi, fig=fig, ax=ax, projection="3d", alpha_max=6)

    plt.close(fig)
    return fig

Np =50
rr = 0.5

psi_in = tensor(basis(Np,0),basis(Np,0))
s1 = tensor(squeeze(Np,rr*np.exp(1j*0*np.pi/2)),qeye(Np))
s2 = tensor(qeye(Np),squeeze(Np,rr*np.exp(1j*np.pi)))
a1 = tensor(destroy(Np),qeye(Np))
a2 = tensor(qeye(Np),destroy(Np))

psi_out1 = (a1*s1*s2*psi_in + s1*a2*s2*psi_in).unit() #structure 1
psi_out2 = (a1*a1*s1*a2*s2*psi_in + a1*s1*a2*a2*s2*psi_in).unit() #structure 2

# theta = np.pi/4
# bs = theta * (a1.dag()*a2 - a1*a2.dag())
# bs_op = bs.expm()
psi_in1 = tensor(basis(Np,2),basis(Np,0))
psi_in2 = tensor(basis(Np,0),basis(Np,2))
psi_out3 = (0.5*(np.sinh(rr)**2)*(squeezing(a1,a2,-2*rr)*psi_in1 - squeezing(a1,a2,-2*rr)*psi_in2)).unit() 
# psi_out = bs_op*s1*s2

psi_out = psi_out3
structure = 3
# '''STRUCTURE 4'''
    
# psi_in = tensor(basis(Np,0),basis(Np,0),basis(Np,0),basis(Np,0))
# s1 = tensor(squeeze(Np,-rr*np.exp(1j*0)),qeye(Np),qeye(Np),qeye(Np))
# s2 = tensor(qeye(Np),squeeze(Np,-rr*np.exp(1j*np.pi)),qeye(Np),qeye(Np))
# s3 = tensor(qeye(Np),qeye(Np),squeeze(Np,-rr*np.exp(1j*0)),qeye(Np))
# s4 = tensor(qeye(Np),qeye(Np),qeye(Np),squeeze(Np,-rr*np.exp(1j*np.pi)))
# a1 = tensor(destroy(Np),qeye(Np),qeye(Np),qeye(Np))
# a2 = tensor(qeye(Np),destroy(Np),qeye(Np),qeye(Np))
# a3 = tensor(qeye(Np),qeye(Np),destroy(Np),qeye(Np))
# a4 = tensor(qeye(Np),qeye(Np),qeye(Np),destroy(Np))
# psi_in = s1*s2*s3*s4*psi_in

# a_sqmin = (a1*a2*a3*psi_in + a1*a3*a3*psi_in + a1*a2*a4*psi_in + a1*a3*a4*psi_in).unit()
# a_sq = (a2*a2*a3*psi_in + a2*a3*a3*psi_in + a2*a2*a4*psi_in + a2*a3*a4*psi_in).unit()
# psi_out = (a_sqmin + a_sq).unit()


# alpha = 10.0
# coherent_state_alpha = coherent(N=50, alpha=alpha)
# coherent_state_minus_alpha = coherent(N=50, alpha=-alpha)
# cat_state = (coherent_state_alpha + coherent_state_minus_alpha).unit()
# psi_out = cat_state

x = np.linspace(-5, 5, 100)
p = np.linspace(-5, 5, 100)
X, P = np.meshgrid(x, p)


'''FIGURES'''
'''Total state 1'''
# ket_p = (tensor(sq_state_menys,sq_state)).unit()
W = wigner(ket2dm(psi_out).ptrace(0), x, p, g=2)
plt.figure(1)
plt.contourf(X, P, W, levels=100, cmap='RdYlBu_r')
# plt.title(r'$\Psi_1$, Structure ' + str(structure) + ', r = ' + str(rr))
# plt.title(r'|$\xi$->(|$\xi$>)')
plt.xlabel('X')
plt.ylabel('P')
plt.colorbar()
plt.tight_layout()
plt.savefig("wigner_psi1_struct" + str(structure) + ".png") 
plt.show()


'''Total state 2'''
# ket_p = (tensor(sq_state_menys,sq_state)).unit()
W = wigner(ket2dm(psi_out).ptrace(1), x, p, g=2)
plt.figure(2)
plt.contourf(X, P, W, levels=100, cmap='RdYlBu_r')
# plt.contourf(X, P, W, levels=100, cmap='viridis')
# plt.title(r'$\Psi_2$, Structure ' + str(structure) + ', r = ' + str(rr))
# plt.title(r'(|$\xi$->)|$\xi$>')
plt.xlabel('X')
plt.ylabel('P')
plt.colorbar()
plt.tight_layout()
plt.savefig("wigner_psi2_struct" + str(structure) + ".png") 
plt.show()

# plt.figure(3)
# plot_wigner_2d_3d(psi_out)
# plt.show()

# '''Total state 3'''
# # ket_p = (tensor(sq_state_menys,sq_state)).unit()
# W = wigner(ket2dm(psi_out).ptrace(2), x, p, g=2)
# plt.figure(3)
# plt.contourf(X, P, W, levels=100, cmap='RdYlBu')
# plt.title(r'$\Psi_3$, Structure ' + str(structure) + ', r = ' + str(rr))
# # plt.title(r'(|$\xi$->)|$\xi$>')
# plt.xlabel('X')
# plt.ylabel('P')
# plt.colorbar()
# plt.savefig("wigner_psi3_struct" + str(structure) + ".png") 
# plt.show()

# '''Total state 4'''
# # ket_p = (tensor(sq_state_menys,sq_state)).unit()
# W = wigner(ket2dm(psi_out).ptrace(3), x, p, g=2)
# plt.figure(4)
# plt.contourf(X, P, W, levels=100, cmap='RdYlBu')
# plt.title(r'$\Psi_4$, Structure ' + str(structure) + ', r = ' + str(rr))
# # plt.title(r'(|$\xi$->)|$\xi$>')
# plt.xlabel('X')
# plt.ylabel('P')
# plt.colorbar()
# plt.savefig("wigner_psi4_struct" + str(structure) + ".png") 
# plt.show()

# # sq_state = squeeze(Np,-rr*np.exp(1j*0*np.pi/2))*basis(Np,0)
# # sq_state_menys = destroy(Np) * sq_state
# # # struct1_state = (tensor(sq_state_menys,sq_state) + tensor(sq_state,sq_state_menys)).unit()
# # ket_p = (tensor(sq_state_menys,sq_state)).unit()
# # ''' Squeezed state -1 photon'''
# # W =wigner(ket2dm(ket_p).ptrace(0), x, p, g=2)
# # plt.figure(3) 
# # plt.contourf(X, P, W, levels=100, cmap='RdYlBu')
# # plt.title(r'|$\xi$->, Structure ' + str(structure))
# # plt.xlabel('X')
# # plt.ylabel('P')
# # plt.colorbar()
# # plt.show()

# # ''' Squeezed state'''
# # W =wigner(ket2dm(ket_p).ptrace(1), x, p, g=2)
# # plt.figure(4) 
# # plt.contourf(X, P, W, levels=100, cmap='RdYlBu')
# # plt.title(r'|$\xi$>, Structure ' + str(structure))
# # plt.xlabel('X')
# # plt.ylabel('P')
# # plt.colorbar()
# # plt.show()



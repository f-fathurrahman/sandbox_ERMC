
import sys
# from time import time

import numpy as np
import copy
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from matplotlib import pyplot as plt


class monte_carlo():
  
  def __init__(self, h=0.001, inseed=None):
    self.h = h
    self.inseed = inseed

  def rho(self, psi, rs, psi_args):
    rhof=(np.conj(psi(rs, **psi_args)))*psi(rs, **psi_args)
    return rhof

  def ERmom(self, psi, rs, psi_args, s):
    npart, ndim = rs.shape
    rhoold2 = self.rho(psi, rs, psi_args)
    
    p1 = complex(0.,0.)
    p2 = 0.
    p_arr = np.zeros((npart,ndim),dtype=complex)
    hbar = 1.

    for ipart in range(npart):
        dif1 = 0.
        dif_sqrho1 = 0.
        psi0t = 0.
        sqrho0t = 0.

        dif = 0.
        dif_conj = complex(0.,0.)
        dif_rho = 0.

        for idim in range(ndim):
            r = rs[ipart,idim]
            rs[ipart,idim] = r + self.h
            psip = psi(rs, **psi_args)                  #p2
            psip_conj = np.conj(psi(rs, **psi_args))     #p2
            rhop = self.rho(psi, rs, psi_args)       #p1
            
            sqrhop = np.sqrt(rhop)
            rs[ipart,idim] = r
            psi0 = psi(rs, **psi_args)                 #p2
            psi0_conj = np.conj(psi(rs, **psi_args))     #p2
            rho0 = self.rho(psi, rs, psi_args)       #p1

            dif = (psip-psi0)/self.h                   #p2
            dif_conj = (psip_conj-psi0_conj)/self.h    #p2
            dif_rho = (rhop-rho0)/self.h   #p1
            
            p1 = (0.5*hbar/complex(0,1))*((dif/psi0)-(dif_conj/psi0_conj))
            p2 = s*0.5*(dif_rho/rhoold2) #eq. 12, Budiyono 2020
            p_arr[ipart,idim] = (p1+p2)

    return p_arr                 #momentum (added for calculating Heisenberg UR)

  def sampling(self, psi, psi_args, Ncal, npart, ndim):
    nm, th = 100, 0.8
    
    if self.inseed == None:
      pass
    else:
      np.random.seed(self.inseed)
    
    rolds = np.random.uniform(-1, 1, (npart, ndim))
    psiold = psi(rolds, **psi_args)
    iacc, Nsample = 0, 0
    ERp = np.zeros((Ncal, npart, ndim))
    ERq = np.zeros((Ncal, npart, ndim))
    
    for itot in range (nm*Ncal):
        rnews = rolds+th*np.random.uniform(-1,1,(npart, ndim))
        psinew = psi(rnews, **psi_args)
        psiratio = (psinew/psiold)**2
           
        if psiratio > np.random.uniform(0,1):
            rolds = np.copy(rnews)
            psiold = psinew
            iacc +=1
        if (itot%nm)==0:
            s = np.random.binomial(1, 0.5, 1)
            s = np.where(s==0, -1, s)
            ERq[Nsample, :, :] = rolds
            ERp[Nsample, :, :] = self.ERmom(psi, rolds, psi_args, s)
            Nsample += 1
            
    return rolds, Nsample, ERp, ERq


def psi(rs, oms, deltaq, aa):
    val = np.exp(-aa*np.sum(oms**2*(rs-deltaq)**2))
    return val

psi_args = dict(
    oms=np.array([1., 1., 1.]),
    deltaq=0.5,
    aa=0.9
)

sampler = monte_carlo(inseed=8735)

N = [10000, 100000]
npart = 1
ndim = 3

dir = " "

for i in range(len(N)):
  _, _, p, q = sampler.sampling(psi, psi_args, N[i], npart, ndim)
  p = p.reshape(N[i], npart*ndim)
  q = q.reshape(N[i], npart*ndim)

  q_name = dir + "q_" + str(N[i]) + "_" + str(npart) + "_" + str(ndim) + ".txt"
  p_name = dir + "p_" + str(N[i]) + "_" + str(npart) + "_" + str(ndim) + ".txt"

  np.savetxt(q_name, q)
  np.savetxt(p_name, p)
  print("Save successful for N =", N[i])

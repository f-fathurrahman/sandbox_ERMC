import numpy as np
from my_monte_carlo import MonteCarlo


# Ψ(q) = exp(−α/2 * ω * (q−Δq)^2)
# α = 0.9 
# ωx,ωy,ωz = 1.0
# Δq (gaussian center)=0.5

# Nama file = q_N_npart_ndim.txt
# Cont. = q_10000_1_3.txt
# Di dalam folder "sample_set"


def psi(rs, oms, al, deltaq):
    rexp = np.sum(oms*(rs - deltaq)**2)
    return np.exp(-0.5*al*rexp)

psi_args = dict(
    oms=np.array([1., 1., 1.]),
    al=0.9,
    deltaq=0.5
)

sampler = MonteCarlo(inseed=8735)

N = [100]
npart = 1 
ndim = 3

dirsave = "DATA_sample_set"

for i in range(len(N)):
    _, _, p, q = sampler.sampling(psi, psi_args, N[i], npart, ndim)
    p = p.reshape(N[i], npart*ndim)
    q = q.reshape(N[i], npart*ndim)

    # XXX: do not include manually inside the path
    # Probably need something like joinpath function in Julia
    q_name = dirsave + "/q_" + str(N[i]) + "_" + str(npart) + "_" + str(ndim) + ".txt"
    p_name = dirsave + "/p_" + str(N[i]) + "_" + str(npart) + "_" + str(ndim) + ".txt"

    # XXX use binary format? Like pickle
    np.savetxt(q_name, q)
    np.savetxt(p_name, p)
    print("Save successful for N =", N[i])


print("All samples saved successfuly.")

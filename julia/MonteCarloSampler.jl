import Random
using Distributions

struct MonteCarloSampler
    iseed::Int64
    h::Float64
end

struct PsiArgs
    ω::Vector{Float64}
    α::Float64
    Δq::Float64
end
# FIXME: Add Ndim?

function PsiArgs()
    return PsiArgs([1.0, 1.0, 1.0], 0.9, 0.5)
end




function calc_psi( rs; ω=[1.0, 1.0, 1.0], α=0.9, Δq=0.5 )
    Ndim = size(rs, 1)
    Nparticles = size(rs, 2)
    rexp = 0.0
    for ipart in 1:Nparticles
        for idim in 1:Ndim
            rexp += ω[idim] * ( rs[idim,ipart] - Δq )^2
        end
    end
    return exp(-0.5 * α * rexp)
end

# psi is always real in this case?
function calc_rho(rs)
    psi = calc_psi(rs)
    return conj(psi) * psi
end

function calc_ERmom(rs, s, h)
    #
    Ndim = size(rs, 1)
    Nparticles = size(rs, 2)
    #
    rhoold2 = calc_rho(rs)
    p1 = 0.0 + im*0.0
    p2 = 0.0
    p_arr = zeros(ComplexF64, Ndim, Nparticles)
    ħ = 1.0
    for ipart in 1:Nparticles
        dif1 = 0.0
        dif_sqrho1 = 0.0
        psi0t = 0.0
        sqrho0t = 0.0
        dif = 0.0
        dif_conj = 0.0 + im*0.0
        dif_rho = 0.0
        for idim in 1:Ndim
            #
            r = rs[idim,ipart]
            #
            rs[idim,ipart] = r + h
            psip = calc_psi(rs) # p2
            psip_conj = conj(psip)     #p2
            rhop = calc_rho(rs)# p1
            sqrhop = sqrt(rhop)
            #
            rs[idim,ipart] = r
            psi0 = calc_psi(rs) # p2
            psi0_conj = conj(psi0)     # p2
            rho0 = calc_rho(rs)       # p1
            #
            dif = (psip-psi0)/h                   # p2
            dif_conj = (psip_conj - psi0_conj)/h    # p2
            dif_rho = (rhop - rho0)/h   # p1
            p1 = (0.5*ħ/im) * ( dif/psi0 - dif_conj/psi0_conj )
            p2 = s*0.5*(dif_rho/rhoold2) # eq. 12, Budiyono 2020
            p_arr[idim,ipart] = (p1 + p2)
        end
    end
    return p_arr # momentum (added for calculating Heisenberg UR)
end

function sample_ER( sampler::MonteCarloSampler, Nparticles, Ndim, Ncal )
    iseed = sampler.iseed
    h = sampler.h

    Random.seed!(iseed)
    rnd = Distributions.Uniform(-1.0, 1.0)
    rnd_binom = Binomial()

    nm = 100
    th = 0.8

    r_old = rand(rnd, Ndim, Nparticles)
    psi_old = calc_psi(r_old)

    iacc = 0
    Nsample = 0 # counter for accepted samples
    ERq = Vector{Matrix{Float64}}(undef, Ncal)
    ERp = Vector{Matrix{Float64}}(undef, Ncal)
    for i in 1:Ncal
        ERq[i] = zeros(Ndim, Nparticles)
        ERp[i] = zeros(Ndim, Nparticles)
    end

    # Preallocate r_new
    r_new = zeros(Ndim, Nparticles)

    #println("nm*Ncal = ", nm*Ncal)
    for itot in 1:(nm*Ncal)
        @views r_new[:,:] .= r_old[:,:] + th * rand(rnd, Ndim, Nparticles)
        psi_new = calc_psi(r_new)
        psiratio = (psi_new/psi_old)^2
        if psiratio > rand()
            @views r_old[:,:] .= r_new[:,:]
            psi_old = psi_new
            iacc += 1
        end
        #println("itot = ", itot, " r_old = ", r_old)
    
        if itot%nm == 0
            s = rand(rnd_binom)
            if s == 0
                s = -1
            end
            Nsample += 1
            @views ERq[Nsample][:,:] = r_old[:,:]
            @views ERp[Nsample][:,:] = calc_ERmom(r_old, s, h)
        end
        #if itot % 10_000 == 0
        #    println("Done itot = ", itot, " from ", nm*Ncal)
        #end
    end
    #println("iacc = ", iacc)
    return ERq, ERp
end


function main()
    sampler = MonteCarloSampler(1234, 0.01)
    Ndim = 3
    Nparticles = 1
    Ncal = 100_000
    #Ncal = 10_000_000
    #Ncal = 100
    ERp, ERq = sample_ER(sampler, Nparticles, Ndim, Ncal)
end

main()
@time main()

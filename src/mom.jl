"""
    mom_obj!(obs, β, τ, Lγ, lγω, lω; eV, needgrad::Bool)  
    mom_obj!(m; needgrad::Bool)

Evaluate the method of moments objective function for the given data and 
parameter estimates and also the gradient. eV can be precalculated to reduce 
computation time in fitting. 
"""
function mom_obj!(
    obs     ::VarLmmObs{T},
    β       ::Vector{T},
    τ       ::Vector{T},
    Lγ      ::Matrix{T}, # must be lower triangular
    lγω     ::Vector{T},
    lω      ::Vector{T},
    mgfγω   ::T,
    needgrad::Bool = true
    ) where T <: BlasReal        
    n, p = size(obs.X)
    q, l = size(obs.Z, 2), size(obs.W, 2)
    if needgrad
        fill!(obs.∇β, 0)
        fill!(obs.∇τ, 0)
        fill!(obs.∇Lγ, 0)
        fill!(obs.∇lγω, 0)
        fill!(obs.∇lω, 0)
    end
    # update residuals ri=yi-Xi*β and the variance residual matrix
    update_res!(obs, β)
    # mul!(obs.R, obs.res, transpose(obs.res))
    # comparing to above line, BLAS utilizes symmetry.
    BLAS.syrk!('U', 'N', T(1), obs.res, T(0), obs.R)
    # mul!(obs.storage_nq, obs.Z, Lγ) # storage_nq = Z * Lγ
    # comparing to above line, BLAS utilizes lower triangular property of Lγ
    copy!(obs.storage_nq, obs.Z)
    BLAS.trmm!('R', 'L', 'N', 'N', T(1), Lγ, obs.storage_nq)
    # mul!(obs.R, obs.storage_nq, transpose(obs.storage_nq), T(-1), T(1))
    # comparing to above line, BLAS utilizes symmetry.
    BLAS.syrk!('U', 'N', T(-1), obs.storage_nq, T(1), obs.R)
    # only upper triangular part of R is updated; now make it symmetric
    LinearAlgebra.copytri!(obs.R, 'U')
    mul!(obs.expwτ, obs.W, τ)
    @inbounds for i in 1:n
        obs.expwτ[i] = exp(obs.expwτ[i])
        obs.R[i, i] -= mgfγω * obs.expwτ[i]
    end
    obj = (1//2) * sum(abs2, obs.R)
    
    # gradient
    if needgrad
        # wrt β
        # BLAS.symv!('U', T(1), obs.R, obs.res, T(0), obs.storage_n1)
        # this line is faster than BLAS symv!
        mul!(obs.storage_n1, obs.R, obs.res)
        mul!(obs.∇β, transpose(obs.X), obs.storage_n1)
        obs.∇β .*= -2
        
        # wrt τ
        @inbounds for i in 1:n
            obs.storage_n1[i] = obs.R[i, i] * obs.expwτ[i]
        end
        mul!(obs.∇τ, transpose(obs.W), obs.storage_n1)
        obs.∇τ .*= -mgfγω
        
        # wrt Lγ
        mgfγωsumriw = - mgfγω * sum(obs.storage_n1)
        # ∇Lγ = Z' * R * Z * Lγ (for now)
        # mul!(obs.storage_nq, obs.R, obs.Z)
        BLAS.symm!('L', 'U', T(1), obs.R, obs.Z, T(0), obs.storage_nq)
        mul!(obs.storage_qq, transpose(obs.Z), obs.storage_nq)
        # mul!(obs.∇Lγ, obs.storage_qq, Lγ)
        # BLAS utilizing triangular property is slower than following line
        copy!(obs.∇Lγ, obs.storage_qq)
        BLAS.trmm!('R', 'L', 'N', 'N', T(1), Lγ, obs.∇Lγ)
        
        # ∇Lγ = ∂f / ∂Lγ
        # storage_q1 = (I + Lγ') * lγω
        copy!(obs.storage_q1, lγω)
        mul!(obs.storage_q1, transpose(Lγ), lγω, T(1), T(1))        
        # mul!(obs.∇Lγ, lγω, transpose(obs.storage_q1), mgfγωsumriw, T(1))
        # above line incurs 3 memory allocations. don't know why.
        # resort to BLAS.ger here
        @inbounds for j in eachindex(obs.∇Lγ)
            obs.∇Lγ[j] *= -2
        end
        BLAS.ger!(mgfγωsumriw, lγω, obs.storage_q1, obs.∇Lγ)
        
        # wrt lγω
        copy!(obs.∇lγω, obs.storage_q1)
        mul!(obs.∇lγω, Lγ, obs.storage_q1, T(1), T(1))
        obs.∇lγω .*= mgfγωsumriw
        
        # wrt lω
        obs.∇lω[1] = mgfγωsumriw * lω[1]
    end
    obj
end

"""
TODO
"""
function mom_obj!(
    m::VarLmmModel{T},
    needgrad::Bool = true
    ) where T <: BlasReal
    if needgrad
        fill!(  m.∇β, 0)
        fill!(  m.∇τ, 0)
        fill!( m.∇Lγ, 0)
        fill!(m.∇lγω, 0)
        fill!( m.∇lω, 0)
    end
    # pre-calculate mgfγω
    mgfγω = mgf_γω(m)
    # accumulate obj and gradient
    obj = zero(T)
    for i in eachindex(m.data)
        obji = mom_obj!(m.data[i], m.β, m.τ, m.Lγ, m.lγω, m.lω, mgfγω, needgrad)
        obj += obji
        if needgrad
            m.∇β   .+= m.data[i].∇β
            m.∇τ   .+= m.data[i].∇τ
            # m.∇Lγ  .+= m.data[i].∇Lγ # There's allocation here. Why?
            @inbounds for j in eachindex(m.∇Lγ)
                m.∇Lγ[j] += m.data[i].∇Lγ[j]
            end
            m.∇lγω .+= m.data[i].∇lγω
            m.∇lω  .+= m.data[i].∇lω
        end
    end
    obj
end

"""
    update_res!(obs, β)

Update the residual vector of `obs::VarLmmObs` according to `β`.
"""
function update_res!(
    obs::VarLmmObs{T}, 
    β::Vector{T}
    ) where T <: BlasReal    
    copyto!(obs.res, obs.y)
    mul!(obs.res, obs.X, β, -one(T), one(T))
end

"""
    update_res!(m::VarLmmModel)

Update residual vector of each observation in `m` according to `m.β`.
"""
function update_res!(m::VarLmmModel{T}) where T <: BlasReal
    for i in eachindex(m.data)
        update_res!(m.data[i], m.β)
    end
    nothing
end

"""
    mgf_γω(m::VarLmmModel)

Moment generating function of random effects `[γ; ω]` evaluated at point
`[ℓγω; 1]`. For now, we assume `[γ; ω]` are jointly normal with mean 0 and 
covariance `Σγω`. Later we will allow user flexibility to specify this mgf 
to relax the normality assumption on random effects.
"""
function mgf_γω(m::VarLmmModel{T}) where T <: BlasReal
    # m.storage_q = (I + Lγ') * lγω
    copy!(m.storage_q, m.lγω)
    mul!(m.storage_q, transpose(m.Lγ), m.lγω, one(T), one(T))
    exp((1//2) * (abs2(m.lω[1]) + sum(abs2, m.storage_q)))
end

"""
    mom_obj!(obs, β, τ, Lγ, lγω, lω; eV, needgrad::Bool)  
    mom_obj!(m; needgrad::Bool)

Evaluate the method of moments objective function for the given data and 
parameter estimates and also the gradient. eV can be precalculated to reduce 
computation time in fitting. 
"""
function mom_obj!(
    obs      ::VarLmmObs{T},
    β        ::Vector{T},
    τ        ::Vector{T},
    Lγ       ::Matrix{T}, # must be lower triangular
    needgrad ::Bool = true
    ) where T <: BlasReal        
    n, p = size(obs.X)
    q, l = size(obs.Z, 2), size(obs.W, 2)
    if needgrad
        fill!(obs.∇β, 0)
        fill!(obs.∇τ, 0)
        fill!(obs.∇Lγ, 0)
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
        obs.R[i, i] -= obs.expwτ[i]
    end
    obj = (1//2) * abs2(norm(obs.R))
    
    # gradient
    if needgrad
        # wrt β
        # BLAS.symv!('U', T(-2), obs.R, obs.res, T(0), obs.storage_n1)
        # this line is faster than BLAS symv!
        mul!(obs.storage_n1, obs.R, obs.res)
        BLAS.gemv!('T', T(-2), obs.X, obs.storage_n1, T(0), obs.∇β)
        
        # wrt τ
        @inbounds for i in 1:n
            obs.storage_n1[i] = obs.R[i, i] * obs.expwτ[i]
        end
        BLAS.gemv!('T', T(-1), obs.W, obs.storage_n1, T(0), obs.∇τ)
        
        # wrt Lγ
        # ∇Lγ = Z' * R * Z * Lγ
        # mul!(obs.storage_nq, obs.R, obs.Z)
        BLAS.symm!('L', 'U', T(1), obs.R, obs.Z, T(0), obs.storage_nq)
        mul!(obs.storage_qq, transpose(obs.Z), obs.storage_nq)
        # mul!(obs.∇Lγ, obs.storage_qq, Lγ)
        # BLAS utilizing triangular property may be slower for small q
        copy!(obs.∇Lγ, obs.storage_qq)
        BLAS.trmm!('R', 'L', 'N', 'N', T(-2), Lγ, obs.∇Lγ)
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
    end
    # accumulate obj and gradient
    obj = zero(T)
    for i in eachindex(m.data)
        obji = mom_obj!(m.data[i], m.β, m.τ, m.Lγ, needgrad)
        obj += obji
        if needgrad
            m.∇β .+= m.data[i].∇β
            m.∇τ .+= m.data[i].∇τ
            @inbounds for j in eachindex(m.∇Lγ)
                m.∇Lγ[j] += m.data[i].∇Lγ[j]
            end
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
    mul!(obs.res, obs.X, β, T(-1), T(1))
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

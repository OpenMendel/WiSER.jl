"""
    mom_obj!(obs, β, τ, Lγ, needgrad::Bool)  
    mom_obj!(m; needgrad::Bool)

Evaluate the method of moments objective function for the given data and 
parameter values. Gradient is also calculated if `needgrad=true`.
"""
function mom_obj!(
    obs       :: VarLmmObs{T},
    β         :: Vector{T},
    τ         :: Vector{T},
    Lγ        :: Matrix{T}, # must be lower triangular
    needgrad  :: Bool = true,
    updateres :: Bool = true
    ) where T <: BlasReal
    n, q = length(obs.y), size(obs.Z, 2)
    if needgrad
        fill!(obs.∇β, 0)
        fill!(obs.∇τ, 0)
        fill!(obs.∇Lγ, 0)
    end
    # update the residual vector ri = y_i - Xi β
    updateres && update_res!(obs, β)
    # obs.storage_qq = (Z' * Z) * L
    # obs.∇Lγ = (Z' * Z) * (L * L') * (Z' * Z)
    # BLAS.symm!('L', 'U', T(1), obs.ztz, Lγ, T(0), obs.storage_qq)
    copy!(obs.storage_qq, obs.ztz)
    BLAS.trmm!('R', 'L', 'N', 'N', T(1), Lγ, obs.storage_qq)
    BLAS.syrk!('U', 'N', T(1), obs.storage_qq, T(0), obs.∇Lγ)
    LinearAlgebra.copytri!(obs.∇Lγ, 'U')
    # mul!(obs.storage_qq, Lγ, transpose(Lγ))
    copy!(obs.storage_qq, Lγ)
    BLAS.trmm!('R', 'L', 'T', 'N', T(1), Lγ, obs.storage_qq)
    # storage_qn = L' * Z'
    mul!(obs.storage_qn, transpose(Lγ), transpose(obs.Z))
    # storage_q1 = L' * Z' * res
    mul!(obs.storage_q1, transpose(Lγ), obs.ztres)
    # update W * τ
    mul!(obs.expwτ, obs.W, τ)
    obj  = (1//2) * (dot(obs.∇Lγ, obs.storage_qq) + abs2(obs.resnrm2[1])) 
    obj -= abs2(norm(obs.storage_q1))
    @inbounds for i in 1:n
        obs.expwτ[i] = exp(obs.expwτ[i])
        obj += (1//2) * abs2(obs.expwτ[i])
        obs.zlltzt_dg[i] = 0
        for j in 1:q
            obs.zlltzt_dg[i] += abs2(obs.storage_qn[j, i])
        end
        obj += obs.expwτ[i] * (obs.zlltzt_dg[i] - obs.res2[i])
    end
    
    # gradient
    if needgrad
        # wrt τ
        @inbounds for i in 1:n
            obs.storage_n1[i] = (obs.res2[i] - obs.expwτ[i] - obs.zlltzt_dg[i]) * obs.expwτ[i]
        end
        BLAS.gemv!('T', T(-1), obs.W, obs.storage_n1, T(0), obs.∇τ)        
        # wrt Lγ
        # ∇Lγ = (Z' * R * Z) * Lγ
        # mul!(obs.storage_qn, transpose(obs.Z), Diagonal(obs.expwτ))
        @inbounds for i in 1:n
            sqrtei = sqrt(obs.expwτ[i])
            for j in 1:q
                obs.storage_qn[j, i] = sqrtei * obs.Z[i, j]
            end
        end
        BLAS.syrk!('U', 'N', T(1), obs.storage_qn, T(1), obs.∇Lγ)
        BLAS.syrk!('U', 'N', T(-1), obs.ztres, T(1), obs.∇Lγ)
        LinearAlgebra.copytri!(obs.∇Lγ, 'U')
        # mul!(obs.∇Lγ, obs.storage_qq, Lγ)
        # BLAS utilizing triangular property may be slower for small q
        BLAS.trmm!('R', 'L', 'N', 'N', T(2), Lγ, obs.∇Lγ)
    end
    obj
end

"""
TODO
"""
function mom_obj!(
    m         :: VarLmmModel{T},
    needgrad  :: Bool = true,
    updateres :: Bool = true
    ) where T <: BlasReal
    if needgrad
        fill!(  m.∇β, 0)
        fill!(  m.∇τ, 0)
        fill!( m.∇Lγ, 0)
    end
    # accumulate obj and gradient
    obj = zero(T)
    for i in eachindex(m.data)
        obj += mom_obj!(m.data[i], m.β, m.τ, m.Lγ, needgrad, updateres)
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
    obs.res2      .= abs2.(obs.res)
    obs.resnrm2[1] = sum(obs.res2)
    mul!(obs.ztres, transpose(obs.Z), obs.res)
    obs.res
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

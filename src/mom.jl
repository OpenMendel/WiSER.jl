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
    needhess  :: Bool = true,
    updateres :: Bool = true
    ) where T <: BlasReal
    (q, n) = size(obs.Zt)
    # update the residual vector ri = y_i - Xi β
    updateres && update_res!(obs, β)
    # obs.storage_qq = (Z' * Z) * L
    copy!(obs.storage_qq, obs.ztz)
    BLAS.trmm!('R', 'L', 'N', 'N', T(1), Lγ, obs.storage_qq)
    # ∇Lγ = (Z' * Z) * L * L' * (Z' * Z) for now, needed for gradient later
    needgrad && BLAS.syrk!('U', 'N', T(1), obs.storage_qq, T(0), obs.∇Lγ)
    # HLγLγ = C'(L'Z'Z ⊗ Z'ZL)KC for now, needed for hessian later
    needhess && Ct_At_kron_A_KC!(fill!(obs.HLγLγ, 0), obs.storage_qq)
    # obs.storage_qq = L' * (Z' * Z) * L
    BLAS.trmm!('L', 'L', 'T', 'N', T(1), Lγ, obs.storage_qq)
    # storage_qn = L' * Z'
    copy!(obs.storage_qn, obs.Zt)
    BLAS.trmm!('L', 'L', 'T', 'N', T(1), Lγ, obs.storage_qn)
    # storage_q◺n = Cq' * (L'Z' ⊙ Z'), needed for hessian later
    needhess && Ct_A_kr_B!(fill!(obs.storage_q◺n, 0), obs.storage_qn, obs.Zt)
    # storage_q1 = L' * Z' * res
    copy!(obs.storage_q1, obs.ztres)
    BLAS.trmv!('L', 'T', 'N', Lγ, obs.storage_q1)
    # update W * τ
    mul!(obs.expwτ, transpose(obs.Wt), τ)
    obj  = (1//2) * (abs2(norm(obs.storage_qq)) + abs2(obs.resnrm2[1])) 
    obj -= abs2(norm(obs.storage_q1))
    @inbounds for j in 1:n
        obs.expwτ[j] = exp(obs.expwτ[j])
        obj += (1//2) * abs2(obs.expwτ[j])
        obs.zlltzt_dg[j] = 0
        for i in 1:q
            obs.zlltzt_dg[j] += abs2(obs.storage_qn[i, j])
        end
        obj += obs.expwτ[j] * (obs.zlltzt_dg[j] - obs.res2[j])
    end
    
    # gradient
    if needgrad
        # wrt τ
        @inbounds for j in 1:n
            obs.storage_n1[j] = (obs.res2[j] - obs.expwτ[j] - obs.zlltzt_dg[j]) * obs.expwτ[j]
        end
        BLAS.gemv!('N', T(-1), obs.Wt, obs.storage_n1, T(0), obs.∇τ)
        # wrt Lγ
        # ∇Lγ = (Z' * R * Z) * Lγ
        # obs.storage_qn = obs.Zt * Diagonal(sqrt.(obs.expwτ))
        @inbounds for j in 1:n
            sqrtej = sqrt(obs.expwτ[j])
            for i in 1:q
                obs.storage_qn[i, j] = sqrtej * obs.Zt[i, j]
            end
        end
        # ∇Lγ += storage_qn * storage_qn' - ztres * ztres'
        # ∇Lγ = (Z' * Z) * L * L' * (Z' * Z) was computed earlier
        BLAS.syrk!('U', 'N',  T(1), obs.storage_qn, T(1), obs.∇Lγ)
        BLAS.syrk!('U', 'N', T(-1),      obs.ztres, T(1), obs.∇Lγ)
        copytri!(obs.∇Lγ, 'U')
        # BLAS utilizing triangular property may be slower for small q
        BLAS.trmm!('R', 'L', 'N', 'N', T(2), Lγ, obs.∇Lγ)
    end

    # hessian
    if needhess
        # Hττ = W' * Diagonal(expwτ.^2) * W
        # storage_ln = W' * Diagonal(expwτ)
        @inbounds for j in 1:n
            ej = obs.expwτ[j]
            for i in 1:size(obs.Wt, 1)
                obs.storage_ln[i, j] = ej * obs.Wt[i, j]
            end
        end
        BLAS.syrk!('U', 'N', T(1), obs.storage_ln, T(0), obs.Hττ)
        copytri!(obs.Hττ, 'U')
        # HτLγ = 2 W' * Diagonal(expwτ) * (L'Z' ⊙ Z')' * Cq
        # storage_ln = W' * Diagonal(expwτ) was computed above
        # storage_q◺n = Cq' * (L'Z' ⊙ Z') was computed earlier
        BLAS.gemm!('N', 'T', T(2), obs.storage_ln, obs.storage_q◺n, T(0), obs.HτLγ)
        # HLγLγ = 2 [ C'(L'Z'ZL ⊗ Z'Z)C + C'(L'Z'Z ⊗ Z'ZL)KC ]
        # HLγLγ = C'(L'Z'Z ⊗ Z'ZL)KC was calcualted earlier
        Ct_A_kron_B_C!(obs.HLγLγ, obs.storage_qq, obs.ztz)
        lmul!(2, obs.HLγLγ)
    end
    obj
end

"""
TODO
"""
function mom_obj!(
    m         :: VarLmmModel{T},
    needgrad  :: Bool = true,
    needhess  :: Bool = true,
    updateres :: Bool = true
    ) where T <: BlasReal
    # accumulate obj and gradient
    obj = zero(T)
    if needgrad
        fill!(   m.∇β, 0)
        fill!(   m.∇τ, 0)
        fill!(  m.∇Lγ, 0)    
    end
    if needhess
        fill!(m.Hττ  , 0)
        fill!(m.HτLγ , 0)
        fill!(m.HLγLγ, 0)
    end
    for i in eachindex(m.data)
        obj += mom_obj!(m.data[i], m.β, m.τ, m.Lγ, needgrad, needhess, updateres)
        if needgrad
            BLAS.axpy!(T(1), m.data[i].∇β , m.∇β )
            BLAS.axpy!(T(1), m.data[i].∇τ , m.∇τ )
            BLAS.axpy!(T(1), m.data[i].∇Lγ, m.∇Lγ)
        end
        if needhess
            BLAS.axpy!(T(1), m.data[i].Hττ  , m.Hττ  )
            BLAS.axpy!(T(1), m.data[i].HτLγ , m.HτLγ )
            BLAS.axpy!(T(1), m.data[i].HLγLγ, m.HLγLγ)
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
    BLAS.gemv!('T', T(-1), obs.Xt, β, T(1), obs.res)
    obs.res2      .= abs2.(obs.res)
    obs.resnrm2[1] = sum(obs.res2)
    mul!(obs.ztres, obs.Zt, obs.res)
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


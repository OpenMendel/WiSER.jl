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
              :: Val{false}, # un-weighted fitting    
    needgrad  :: Bool = true,
    needhess  :: Bool = true,
    updateres :: Bool = true
    ) where T <: BlasReal
    (q, n) = size(obs.Zt)
    l = size(obs.Wt, 1)
    ###########
    # objective
    ###########
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
    ###########
    # gradient
    ###########
    if needgrad
        # wrt τ
        @inbounds for j in 1:n
            Rjj = obs.res2[j] - obs.expwτ[j] - obs.zlltzt_dg[j]
            obs.storage_n1[j] = Rjj * obs.expwτ[j]
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
        # ∇Lγ = (Z' * Z) * L * L' * (Z' * Z) was computed earlier        
        # ∇Lγ += storage_qn * storage_qn' - ztres * ztres'
        BLAS.syrk!('U', 'N',  T(1), obs.storage_qn, T(1), obs.∇Lγ)
        BLAS.syrk!('U', 'N', T(-1),      obs.ztres, T(1), obs.∇Lγ)
        copytri!(obs.∇Lγ, 'U')
        # so far ∇Lγ holds ∇Σγ, now ∇Lγ = ∇Σγ * Lγ
        BLAS.trmm!('R', 'L', 'N', 'N', T(2), Lγ, obs.∇Lγ)
    end
    ###########
    # hessian
    ###########
    if needhess
        # Hττ = W' * Diagonal(expwτ.^2) * W
        # storage_ln = W' * Diagonal(expwτ)
        @inbounds for j in 1:n
            ej = obs.expwτ[j]
            for i in 1:l
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

function mom_obj!(
    obs       :: VarLmmObs{T},
    β         :: Vector{T},
    τ         :: Vector{T},
    Lγ        :: Matrix{T}, # must be lower triangular
              :: Val{true}, # weighted fitting
    needgrad  :: Bool = true,
    needhess  :: Bool = true,
    updateres :: Bool = true,
    ) where T <: BlasReal
    # form Ri
    R = transpose(obs.Zt) * Lγ * transpose(Lγ) * obs.Zt
    mul!(obs.expwτ, transpose(obs.Wt), τ)
    for j in 1:size(obs.Xt, 2)
        obs.expwτ[j] = exp(obs.expwτ[j])
        R[j, j] += obs.expwτ[j]
    end
    R .= obs.res * transpose(obs.res) .- R
    VinvRVinv = obs.wtmat * R * obs.wtmat
    obj = (1//2)dot(VinvRVinv, R)
    if needgrad
        obs.∇τ  .= - obs.Wt * (obs.expwτ .* diag(VinvRVinv))
        obs.∇Lγ .= -2obs.Zt * VinvRVinv * transpose(obs.Zt) * Lγ
    end
    if needhess
        obs.Hττ   .= obs.Wt * Diagonal(obs.expwτ) * (obs.wtmat .* obs.wtmat) *
            Diagonal(obs.expwτ) * transpose(obs.Wt)
        obs.HτLγ  .= 2obs.Wt * Diagonal(obs.expwτ) * 
            transpose(Ct_A_kr_B(Lγ * obs.Zt * obs.wtmat, obs.Zt * obs.wtmat))
        obs.HLγLγ .= 
        2Ct_A_kron_B_C(Lγ * obs.Zt * obs.wtmat * transpose(obs.Zt) * Lγ, 
                       obs.Zt * obs.wtmat * transpose(obs.Zt)) + 
        2Ct_At_kron_A_KC(Lγ * obs.Zt * obs.wtmat * transpose(obs.Zt))
    end
    obj
end

"""
    mom_obj!(m::VarLMM, needgrad::Bool, needhess:Bool, updateres::Bool)

Calculate the objective function of a `VarLMM` object and optionally the 
gradient and hessian.
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
        obj += mom_obj!(m.data[i], m.β, m.τ, m.Lγ, Val(m.weighted[1]),
            needgrad, needhess, updateres)
        if needgrad
            BLAS.axpy!(T(1), m.data[i].∇τ   , m.∇τ )
            BLAS.axpy!(T(1), m.data[i].∇Lγ  , m.∇Lγ)
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

"""
    update_wtmat!(m::VarLmmModel)

Update the observation weight matrix according to the current parameter values 
in and `VarLmmModel` object `m`.

"""
function update_wtmat!(m::VarLmmModel{T}) where T <: BlasReal
    # form Vi inverse according to Woodbury Formula
    q = size(m.Lγ, 1)
    for obs in m.data
        fill!(obs.wtmat, T(0))
        mul!(obs.expwτ, transpose(obs.Wt), m.τ)
        # Step 1: assemble Ip + Lt Zt diag(e^{-\eta_j}) Z L
        # storage_qn = L' * Z'
        copy!(obs.storage_qn, obs.Zt)
        BLAS.trmm!('L', 'L', 'T', 'N', T(1), m.Lγ, obs.storage_qn)
        # wtmat = Diagonal(e^{-Wτ})
        @inbounds for j in 1:length(obs.expwτ)
            obs.wtmat[j, j] = exp(-obs.expwτ[j])
        end
        # storage_qn = Lt Zt Diagonal(e^{-Wτ})
        BLAS.trmm!('R', 'L', 'N', 'N', T(1), obs.wtmat, obs.storage_qn) #faster than rmul!
        # storage_qq = Lt Zt diag(e^{-wτ}) Z
        mul!(obs.storage_qq, obs.storage_qn, transpose(obs.Zt))
        #BLAS.gemm!('N', 'T', T(1), obs.storage_qn, obs.Zt, T(0), obs.storage_qq)
        # storage_qq = Lt Zt diag(e^{-wτ}) Z L 
        BLAS.trmm!('R', 'L', 'N', 'N', T(1), m.Lγ, obs.storage_qq)
        # fewer allocs than computing Lt Zt diag(e^{-0.5wτ}) and using syrk
        # storage_qq = Iq + Lt Zt diag(e^{-wτ}) Z L 
        #BLAS.syrk!('U', 'N', T(1), obs.storage_qn, T(0), obs.storage_qq)
        @inbounds for i in 1:q
            obs.storage_qq[i, i] += 1.0
        end
        # Step 2: invert U^{-1} = (Ip + Lt Zt diag(e^{-Wt}) Z L)^{-1}: cholesky
        cholesky!(Symmetric(obs.storage_qq))

        # Step 3: get storange_qn = U'^{-1/2} Lt Zt Diagonal(e^{-Wt})
        #storage_qn = Lt Zt Diagonal(e^{-Wτ})
        BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq, obs.storage_qn)
        
        #now form full matrix 
        BLAS.syrk!('U', 'T', T(-1), obs.storage_qn, T(1), obs.wtmat)
        copytri!(obs.wtmat, 'U')
    end
    nothing
end
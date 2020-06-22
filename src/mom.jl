"""
    mom_obj!(obs::VarLmmObs, β, τ, Lγ, needgrad::Bool)  
    mom_obj!(m::VarLmmModel; needgrad::Bool)

Evaluate the method of moments objective function for the given data and 
parameter values. Gradient is also calculated if `needgrad=true`. hessian
is calculated is `needhess=true`. It updates residuals before evaluating 
if `updateres=true`. If `m.weighted[1]` = true, it will evaluate the weighted
`mom_obj!()` function. `update_wtmat!(m)` should be called prior to using the 
weighted version of `mom_obj!()` to update the weight matrix components. 
"""
function mom_obj!(
    obs       :: VarLmmObs{T},
    β         :: Vector{T},
    τ         :: Vector{T},
    Lγ        :: Matrix{T},  # must be lower triangular
              :: Val{false}, # un-weighted fitting    
    needgrad  :: Bool = true,
    needhess  :: Bool = true,
    updateres :: Bool = false
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
    obs.obj[1]  = (1//2) * (abs2(norm(obs.storage_qq)) + abs2(obs.resnrm2[1])) 
    obs.obj[1] -= abs2(norm(obs.storage_q1))
    map!(exp, obs.expwτ, obs.expwτ)
    @inbounds for j in 1:n
        obs.obj[1] += (1//2) * abs2(obs.expwτ[j])
        obs.zlltzt_dg[j] = 0
        for i in 1:q
            obs.zlltzt_dg[j] += abs2(obs.storage_qn[i, j])
        end
        obs.obj[1] += obs.expwτ[j] * (obs.zlltzt_dg[j] - obs.res2[j])
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
        # obs.∇Lγ = obs.∇Σγ, collect all then multiply by Lγ at model level for ∇Lγ. 
        # BLAS.trmm!('R', 'L', 'N', 'N', T(2), Lγ, obs.∇Lγ)
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
    obs.obj[1]
end

function mom_obj!(
    obs       :: VarLmmObs{T},
    β         :: Vector{T},
    τ         :: Vector{T},
    Lγ        :: Matrix{T}, # must be lower triangular
              :: Val{true}, # weighted fitting
    needgrad  :: Bool = true,
    needhess  :: Bool = true,
    updateres :: Bool = false,
    ) where T <: BlasReal
    (q, n) = size(obs.Zt)
    l = size(obs.Wt, 1)
    
    # update the residual vector ri = y_i - Xi β
    updateres && update_res!(obs, β)

    # Precompute and fill necessary objects
    fill!(obs.Ut_D_U, 0) 
    if needgrad
        fill!(obs.∇τ, 0)
        fill!(obs.diagDVRV, 0)
    end
    if needhess 
        fill!(obs.Hττ, 0)
        fill!(obs.HτLγ, 0)
        fill!(obs.storage_q◺n, 0)
    end
    # obs.expwτ = exp.(Wτ)
    mul!(obs.expwτ, transpose(obs.Wt), τ)
    # for ∇Lγ need storage_n1 = diag(exp.(0.5Wτ))
    needgrad && map!(x -> exp(0.5x), obs.storage_n1, obs.expwτ) 
    map!(exp, obs.expwτ, obs.expwτ)

    #terms to compute and store 
    BLAS.gemm!('T', 'N', one(T), Lγ, obs.Zt_Dinv_r, zero(T), obs.Lt_Zt_Dinv_r)
    BLAS.gemm!('T', 'N', one(T), obs.Zt_Dinv, Lγ, zero(T), obs.Dinv_Z_L)
    BLAS.gemm!('T', 'N', one(T), obs.Zt_UUt, Lγ, zero(T), obs.UUt_Z_L)
    #Lt_Zt_Dinv_Z_L =  Zt_Dinv_Z * L
    BLAS.gemm!('N', 'N', one(T), obs.Zt_Dinv_Z, Lγ, zero(T), obs.Lt_Zt_Dinv_Z_L)
    BLAS.trmm!('L', 'L', 'T', 'N', one(T), Lγ, obs.Lt_Zt_Dinv_Z_L)
    #Lt_Zt_UUt_Z_L =  Zt_UUt_Z_L
    BLAS.gemm!('N', 'N', one(T), obs.Zt_UUt_Z, Lγ, zero(T), obs.Lt_Zt_UUt_Z_L)
    BLAS.trmm!('L', 'L', 'T', 'N', one(T), Lγ, obs.Lt_Zt_UUt_Z_L)

    # Evalute objective function and precompute diag(Vinv*R*Vinv) for ∇τ
    #############
    # objective #
    #############

    #objective function sums
    obs.obj[1] = zero(T)
    obs.obj[1] += abs2(obs.rt_Dinv_r[1]) #1 
    obs.obj[1] -= 2 * obs.rt_UUt_r[1] * obs.rt_Dinv_r[1] #2 
    obs.obj[1] += abs2(obs.rt_UUt_r[1]) #7

    #calculate Ut * D * U
    @inbounds for k in 1:n 
        for j in 1:q 
            for i in j:q
                obs.Ut_D_U[i, j] += obs.Ut[j, k] * obs.Ut[i, k] * obs.expwτ[k]
            end
        end
    end
    copytri!(obs.Ut_D_U, 'L')
 
    if needgrad 
        #storage_qn = Ut_D_U_Ut
        BLAS.gemm!('N', 'N', one(T), obs.Ut_D_U, obs.Ut, zero(T), obs.storage_qn)
        @inbounds for j in 1:n #avx speeds up
            for i in 1:q 
                obs.diagDVRV[j] -= obs.storage_qn[i, j] * obs.Ut[i, j] * obs.expwτ[j] #τ 8
            end
        end
    end

    @inbounds @simd for j in 1:n
        obs.obj[1] += (-2 * abs2(obs.Dinv_r[j]) #3
            + 2 * obs.rt_UUt[j] * obs.Dinv_r[j]  #4
            + 2 * obs.Dinv_r[j] * obs.rt_UUt[j] #8
            - 2 * abs2(obs.rt_UUt[j]) # 9 
            + abs2(obs.Dinv[j]) * obs.expwτ[j] #12 
            - 2 * obs.diagUUt_Dinv[j] * obs.expwτ[j]) * obs.expwτ[j] #13
    end 
    if needgrad 
        @inbounds @simd for j in 1:n
            obs.diagDVRV[j] += (abs2(obs.Dinv_r[j])   #τ 1 
                - 2 * obs.Dinv_r[j] * obs.rt_UUt[j] #τ 2
                - obs.Dinv[j]^2 * obs.expwτ[j]  #τ 3
                + 2 * obs.diagUUt_Dinv[j] * obs.expwτ[j]  #τ 4
                + abs2(obs.rt_UUt[j])) * obs.expwτ[j]  #τ 7
        end
    end

    @inbounds for j in 1:q  #j-i looping for memory access 
        for i in 1:n
            obs.obj[1] += 2 * (abs2(obs.Dinv_Z_L[i, j]) # 14
                - obs.UUt_Z_L[i, j] * obs.Dinv_Z_L[i, j] # 15
                - obs.Dinv_Z_L[i, j] * obs.UUt_Z_L[i, j] # 17
                + abs2(obs.UUt_Z_L[i, j])) * obs.expwτ[i]# 18
        end
    end
    if needgrad 
        @inbounds for j in 1:q
            for i in 1:n
                obs.diagDVRV[i] += (-abs2(obs.Dinv_Z_L[i, j]) #τ 5
                    + 2 * obs.Dinv_Z_L[i, j] * obs.UUt_Z_L[i, j] #τ 6
                    - abs2(obs.UUt_Z_L[i, j])) * obs.expwτ[i] #τ 9
            end
        end
    end

    #obs.storage_qq = L' * Z' * UU' * rr' * Dinv * Z * L
    copyto!(obs.storage_qq, obs.Zt_UUt_rrt_Dinv_Z)
    BLAS.trmm!('L', 'L', 'T', 'N', one(T), Lγ, obs.storage_qq)
    BLAS.trmm!('R', 'L', 'N', 'N', one(T), Lγ, obs.storage_qq) #for #10
    #use ∇Lγ as temporary storage 
    #obs.∇Lγ = L' Zt_UUt_rrt_UUt_Z * L
    copyto!(obs.∇Lγ, obs.Zt_UUt_rrt_UUt_Z)
    BLAS.trmm!('L', 'L', 'T', 'N', one(T), Lγ, obs.∇Lγ)
    BLAS.trmm!('R', 'L', 'N', 'N', one(T), Lγ, obs.∇Lγ) #for #11

    @inbounds for j in 1:q
        obs.obj[1] -= 2 * abs2(obs.Lt_Zt_Dinv_r[j]) #5
        obs.obj[1] += 2 * obs.storage_qq[j, j] #10 
        obs.obj[1] -= 2 * obs.∇Lγ[j, j] # 11
        for i in 1:q
            obs.obj[1] += obs.Ut_D_U[i, j]^2 #16
            obs.obj[1] += 2 * obs.rt_UUt_Z[i] * Lγ[i, j] * obs.Lt_Zt_Dinv_r[j] #6
            obs.obj[1] += abs2(obs.Lt_Zt_Dinv_Z_L[i, j]) #19 
            obs.obj[1] -= 2 * obs.Lt_Zt_Dinv_Z_L[i, j] * obs.Lt_Zt_UUt_Z_L[i,j] #20
            obs.obj[1] += abs2(obs.Lt_Zt_UUt_Z_L[i, j]) #21 
        end
    end

    obs.obj[1] *= (1//2)

    ############
    # Gradient #
    ############
    if needgrad
        #wrt τ
        #∇τ = -W' * diag(D * Vinv * R * Vinv)
        BLAS.gemv!('N', -one(T), obs.Wt, obs.diagDVRV, zero(T), obs.∇τ)

        #wrt Lγ
        # ∇Lγ = -2(Z' * Vinv * R *  Vinv * Z * L
        # obs.storage_qq = (Z' * Vinv * Z) * L
        copy!(obs.storage_qq, obs.Zt_Vinv_Z)
        BLAS.trmm!('R', 'L', 'N', 'N', one(T), Lγ, obs.storage_qq)
        #∇Lγ = Z' * Vinv * Z * L * L' * Z' * Vinv * Z 
        BLAS.syrk!('U', 'N', one(T), obs.storage_qq, zero(T), obs.∇Lγ)
        # obs.storage_qn = Zt_Vinv * sqrt.(D)
        # storage_n1 = diag(exp^{1/2 Wt})
        @inbounds for j in 1:n
            for i in 1:q
                obs.storage_qn[i, j] = obs.storage_n1[j] * obs.Zt_Vinv[i, j]
            end
        end
        # obs.∇Lγ += Zt_Vinv_D_Vinv_Z
        BLAS.syrk!('U', 'N',  one(T), obs.storage_qn, one(T), obs.∇Lγ)
        # obs.∇Lγ += Zt_Vinv_rrt_Vinv_Z
        BLAS.syr!('U', -one(T), obs.Zt_Vinv_r, obs.∇Lγ)
        copytri!(obs.∇Lγ, 'U')
        # obs.∇Lγ = obs.∇Σγ, collect all then multiply by Lγ at model level for ∇Lγ. 
    end

    ###########
    # Hessian #
    ###########
    if needhess
        #wrt τ
        # Hττ = W' * D * Vinv .* Vinv * D  * W

        # storage_ln = Wt * D 
        @inbounds @simd for j in 1:n 
            for i in 1:l
                obs.storage_ln[i, j] = obs.Wt[i, j] * obs.expwτ[j]
            end
        end
        
        @inbounds for j in 1:n 
            for i in 1:l
                obs.Wt_D_Dinv[i, j] = obs.Dinv[j] * obs.storage_ln[i, j]
                obs.Wt_D_sqrtdiagDinv_UUt[i, j] = obs.storage_ln[i, j] * obs.sqrtDinv_UUt[j]
            end
        end 
        BLAS.syrk!('U', 'N', one(T), obs.Wt_D_Dinv, zero(T), obs.Hττ) #first term 
        BLAS.syrk!('U', 'N', T(-2), obs.Wt_D_sqrtdiagDinv_UUt, one(T), obs.Hττ) #second term 
        BLAS.gemm!('N', 'T', one(T), obs.storage_ln, obs.Ut_kr_Ut, zero(T), obs.Wt_D_Ut_kr_Utt) 
        BLAS.syrk!('U', 'N', one(T), obs.Wt_D_Ut_kr_Utt, one(T), obs.Hττ) #third term
        copytri!(obs.Hττ, 'U')


        #wrt HτLγ
        # HτLγ = 2 W' * Diagonal(expwτ) * (L'Z'(V^-1) ⊙ Z'(V^-1))' * Cq
        # storage_ln = W' * Diagonal(expwτ) was computed above
        # storage_q◺n = Cq' * (L'Z'(V^-1) ⊙ Z'(V^-1)) 
        copy!(obs.storage_qn, obs.Zt_Vinv)
        BLAS.trmm!('L', 'L', 'T', 'N', one(T), Lγ, obs.storage_qn)
        Ct_A_kr_B!(obs.storage_q◺n, obs.storage_qn, obs.Zt_Vinv)
        BLAS.gemm!('N', 'T', T(2), obs.storage_ln, obs.storage_q◺n, zero(T), obs.HτLγ)

        #wrt HLγLγ
        # HLγLγ = 2 [ C'(L'Z'(V^-1)ZL ⊗ Z'(V^-1)Z)C + C'(L'Z'(V^-1)Z ⊗ Z'(V^-1)ZL)KC ]
        # obs.storage_qq = (Z' (V^-1) Z) * L
        copy!(obs.storage_qq, obs.Zt_Vinv_Z)
        BLAS.trmm!('R', 'L', 'N', 'N', one(T), Lγ, obs.storage_qq)
        # HLγLγ = C'(L'Z' (V^-1) Z ⊗ Z' (V^-1) ZL)KC first
        Ct_At_kron_A_KC!(fill!(obs.HLγLγ, 0), obs.storage_qq)

        # obs.storage_qq = L' * (Z' (V^-1) Z) * L
        BLAS.trmm!('L', 'L', 'T', 'N', one(T), Lγ, obs.storage_qq)

        Ct_A_kron_B_C!(obs.HLγLγ, obs.storage_qq, obs.Zt_Vinv_Z)
        lmul!(2, obs.HLγLγ)
    end
    obs.obj[1]
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
    updateres :: Bool = false
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
    if m.ismthrd[1]
        Threads.@threads for obs in m.data
            mom_obj!(obs, m.β, m.τ, m.Lγ, Val(m.iswtnls[1]),
                needgrad, needhess, updateres)
        end
        for obs in m.data
            obj += obs.obj[1]
            if needgrad
                BLAS.axpy!(T(1), obs.∇τ   , m.∇τ )
                BLAS.axpy!(T(1), obs.∇Lγ  , m.∇Lγ)
            end
            if needhess
                BLAS.axpy!(T(1), obs.Hττ  , m.Hττ  )
                BLAS.axpy!(T(1), obs.HτLγ , m.HτLγ )
                BLAS.axpy!(T(1), obs.HLγLγ, m.HLγLγ)
            end
        end
    else
        for obs in m.data
            obj += mom_obj!(obs, m.β, m.τ, m.Lγ, Val(m.iswtnls[1]),
                needgrad, needhess, updateres)
            if needgrad
                BLAS.axpy!(T(1), obs.∇τ   , m.∇τ )
                BLAS.axpy!(T(1), obs.∇Lγ  , m.∇Lγ)
            end
            if needhess
                BLAS.axpy!(T(1), obs.Hττ  , m.Hττ  )
                BLAS.axpy!(T(1), obs.HτLγ , m.HτLγ )
                BLAS.axpy!(T(1), obs.HLγLγ, m.HLγLγ)
            end
        end
    end
    # multiply m.∇Lγ by Lγ once instead of at observation level 
    needgrad && BLAS.trmm!('R', 'L', 'N', 'N', T(2), m.Lγ, m.∇Lγ)
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
    BLAS.gemv!('T', T(-1), obs.Xt, β, T(1), copyto!(obs.res, obs.y))
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
    for obs in m.data
        update_res!(obs, m.β)
    end
    nothing
end

"""
    update_wtmat!(m::VarLmmModel)

Update the observation weight matrix according to the parameter values 
`m.τ` and `m.Lγ`. Update `m.β` by WLS and update the residuals accordingly. 
Precompute and store various objects needed to  evaluate the objective function, 
gradient, and Hessian. At return, 
- `m.data[i].∇β  = Xi' inv(Vi) ri`
- `m.data[i].Hββ = Xi' inv(Vi) Xi`
- `m.∇β  = sum_i Xi' inv(Vi) ri`
- `m.Hββ = sum_i Xi' inv(Vi) Xi`
"""
function update_wtmat!(m::VarLmmModel{T}) where T <: BlasReal
    p, q = m.p, m.q
    # update Dinv and U such that Vinv = Dinv - U * U'
    # accumulate quantities for updating β
    fill!(m.Hββ, 0)
    fill!(m.∇β , 0)
    for obs in m.data
        # Form Dinv - UU^T (Woodbury structure of Vi inverse)
        n = length(obs.y)
        # Step 1: assemble Iq + Lt Zt diag(e^{-η}) Z L
        # storage_qn = Lt Zt Diagonal(e^{-0.5η})
        # storage_pn = Xt Diagonal(e^{-0.5η})
        # storage_n1 = Diagonal(e^{-0.5η}) * y 
        copyto!(obs.storage_qn, obs.Zt)
        BLAS.trmm!('L', 'L', 'T', 'N', T(1), m.Lγ, obs.storage_qn)
        mul!(obs.expwτ, transpose(obs.Wt), m.τ)
        @inbounds for j in 1:n
            invsqrtj    = exp(-(1//2)obs.expwτ[j])
            obs.Dinv[j] = abs2(invsqrtj)
            for i in 1:q
                obs.storage_qn[i, j] *= invsqrtj
            end
            for i in 1:p
                obs.storage_pn[i, j] = invsqrtj * obs.Xt[i, j]
            end
            obs.storage_n1[j] = invsqrtj * obs.y[j]
        end
        # storage_qq = Iq + Lt Zt diag(e^{-η}) Z L
        BLAS.syrk!('U', 'N', T(1), obs.storage_qn, T(0), obs.storage_qq)
        @inbounds for i in 1:q
            obs.storage_qq[i, i] += 1
        end
        # Step 2: Cholesky: (Iq + Lt Zt diag(e^{-η}) Z L) = R'R
        LAPACK.potrf!('U', obs.storage_qq)
        # storage_qn = inv(R') Lt Zt diag(exp(-0.5η))
        BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq, obs.storage_qn)
        # storage_qp = inv(R') Lt Zt diag(exp(-η)) X = U' X
        BLAS.gemm!('N', 'T', T(1), obs.storage_qn, obs.storage_pn, T(0), obs.storage_qp)
        # Step 3: accumulate X' Vinv X = X' Dinv X - X' U U' X
        BLAS.syrk!('U', 'N',  T(1), obs.storage_pn, T(0), obs.Hββ)
        BLAS.syrk!('U', 'T', T(-1), obs.storage_qp, T(1), obs.Hββ)
        copytri!(obs.Hββ, 'U')
        BLAS.axpy!(T(1), obs.Hββ, m.Hββ)
        # Step 4: accumulate X' Vinv y = X' Dinv y - X' U U' y
        BLAS.gemv!('N',  T(1), obs.storage_pn, obs.storage_n1, T(0), obs.∇β)
        BLAS.gemv!('N',  T(1), obs.storage_qn, obs.storage_n1, T(0), obs.storage_q1)
        BLAS.gemv!('T', T(-1), obs.storage_qp, obs.storage_q1, T(1), obs.∇β)      
        BLAS.axpy!(T(1), obs.∇β, m.∇β)
    end
    # update β by WLS
    copytri!(m.Hββ, 'U')
    copyto!(m.data[1].storage_pp, m.Hββ) # m.data[1].storage_pp as scratch space
    _, info = LAPACK.potrf!('U', m.data[1].storage_pp)
    info > 0 && throw("sum_i Xi' Vi^{-1} Xi is not positive definite")
    LAPACK.potrs!('U', m.data[1].storage_pp, copyto!(m.β, m.∇β))
    # update residuals according to new β
    update_res!(m)
    # precompute quantities for obj, grad and Hess evaluations
    for obs in m.data
        n = length(obs.y)
        # update obs.∇β = X' Vinv (y - Xβ) = X' Vinv y - X' Vinv X β
        BLAS.gemv!('N', T(-1), obs.Hββ, m.β, T(1), obs.∇β)
        # Update: Dinv_r, rt_Dinv_r, Ut = R'^{-1/2} Lt Zt diag(exp(-η))
        # storange_qn = R'^{-1/2} Lt Zt diag(exp(-0.5η)) from 1st loop        
        copyto!(obs.Ut, obs.storage_qn)
        obs.rt_Dinv_r[1] = 0
        @inbounds for j in 1:n
            obs.Dinv_r[j]     = obs.Dinv[j] * obs.res[j]
            obs.rt_Dinv_r[1] += obs.Dinv[j] * obs.res2[j]
            invsqrtj          = sqrt(obs.Dinv[j])
            for i in 1:q
                obs.Ut[i, j] *= invsqrtj
            end
        end        
        # Zt_Dinv_r
        BLAS.gemv!('N', T(1), obs.Zt, obs.Dinv_r, T(0), obs.Zt_Dinv_r)
        # rt_U and rt_UUt_r
        BLAS.gemm!('T', 'T', T(1), obs.res, obs.Ut, T(0), obs.rt_U)
        obs.rt_UUt_r[1] = abs2(norm(obs.rt_U))
        # rt_UUt
        BLAS.gemm!('N', 'N', T(1), obs.rt_U, obs.Ut, T(0), obs.rt_UUt)
        # rt_UUt_Z
        BLAS.gemm!('N', 'T', T(1), obs.rt_UUt, obs.Zt, T(0), obs.rt_UUt_Z) 
        # Zt_Dinv_r
        BLAS.gemv!('N', T(1), obs.Zt, obs.Dinv_r, T(0), obs.Zt_Dinv_r)
        # storage_qq = Ut * Z
        BLAS.gemm!('N', 'T', T(1), obs.Ut, obs.Zt, T(0), obs.storage_qq)
        BLAS.gemm!('T', 'N', T(1), obs.storage_qq, obs.Ut, T(0), obs.Zt_UUt)
        # storage_qn = Zt * Dinv 
        fill!(obs.diagUUt_Dinv, 0)
        @inbounds for j in 1:n
            invj = obs.Dinv[j]
            for i in 1:q 
                obs.diagUUt_Dinv[j] += abs2(obs.Ut[i, j])
                obs.storage_qn[i, j] = obs.Zt[i, j] * invj
            end
            obs.diagUUt_Dinv[j] *= invj
        end
        copyto!(obs.Zt_Dinv, obs.storage_qn)
        # obs.Zt_Vinv = Z' * Dinv
        copyto!(obs.Zt_Vinv, obs.storage_qn)
        BLAS.gemm!('N', 'T', T(1), obs.storage_qn, obs.Zt, T(0), obs.Zt_Dinv_Z)
        # storage_qq = Zt * U
        BLAS.gemm!('N', 'T', T(1), obs.Zt, obs.Ut, T(0), obs.storage_qq)
        BLAS.syrk!('U', 'N', T(1), obs.storage_qq, T(0), obs.Zt_UUt_Z)
        copytri!(obs.Zt_UUt_Z, 'U')
        # for gradient wrt Lγ
        BLAS.gemm!('T', 'T', T(1), obs.rt_UUt_Z, obs.Zt_Dinv_r, T(0), obs.Zt_UUt_rrt_Dinv_Z)
        BLAS.syrk!('U', 'T', T(1), obs.rt_UUt_Z, T(0), obs.Zt_UUt_rrt_UUt_Z)
        copytri!(obs.Zt_UUt_rrt_UUt_Z, 'U')
        # for Hessian wrt τ
        map!(sqrt, obs.sqrtDinv_UUt, obs.diagUUt_Dinv)
        fill!(obs.Ut_kr_Ut, 0)
        kr_axpy!(obs.Ut, obs.Ut, obs.Ut_kr_Ut)
        # From earlier, obs.Zt_Vinv = Z' * Dinv, continue forming it
        BLAS.axpy!(T(-1), obs.Zt_UUt, obs.Zt_Vinv)
        copyto!(obs.Zt_Vinv_Z, obs.Zt_Dinv_Z)
        BLAS.axpy!(T(-1), obs.Zt_UUt_Z, obs.Zt_Vinv_Z)
        BLAS.gemv!('N', T(1), obs.Zt_Vinv, obs.res, T(0), obs.Zt_Vinv_r)
    end
    nothing
end

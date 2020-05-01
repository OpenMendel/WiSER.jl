"""
    mom_obj!(obs::VarLmmObs, β, τ, Lγ, needgrad::Bool)  
    mom_obj!(m::VarLmmModel; needgrad::Bool)

Evaluate the method of moments objective function for the given data and 
parameter values. Gradient is also calculated if `needgrad=true`. hessian
is calculated is `needhess=true`. It updates residuals before evaluating 
if `updateres=true`. If m.weighted[1] = true, it will evaluate the weighted
mom_obj! function. update_wtmat!(m) should be called prior to using the 
weighted version of mom_obj!() to update the weight matrix components. 
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
    #############
    # objective #
    #############
    # update the residual vector ri = y_i - Xi β
    updateres && update_res!(obs, β)
    if needgrad
        fill!(obs.∇τ, 0)
        fill!(obs.∇Lγ, 0)
        fill!(obs.diagDVRV, 0)
    end
    if needhess 
        fill!(obs.Hττ, 0)
        fill!(obs.HτLγ, 0)
    end
    l, q = size(obs.Wt, 1), size(obs.Zt, 1)
    # obs.expwτ = Wτ
    mul!(obs.expwτ, transpose(obs.Wt), τ)
    fill!(obs.Ut_D_U, 0) 
    fill!(obs.Ut_D_Dinv_Z, 0)
    fill!(obs.Zt_Dinv_D_Dinv_Z, 0)

    #terms to compute and store 
    BLAS.gemm!('T', 'N', T(1), Lγ, obs.Zt_Dinv_r, T(0), obs.Lt_Zt_Dinv_r)
    BLAS.gemm!('N', 'N', T(1), obs.Dinv_Z, Lγ, T(0), obs.Dinv_Z_L)
    BLAS.gemm!('N', 'N', T(1), obs.UUt_Z, Lγ, T(0), obs.UUt_Z_L)
    #Lt_Zt_Dinv_Z_L =  Zt_Dinv_Z * L
    BLAS.gemm!('N', 'N', T(1), obs.Zt_Dinv_Z, Lγ, T(0), obs.Lt_Zt_Dinv_Z_L)
    copyto!(obs.Zt_Dinv_Z_L, obs.Lt_Zt_Dinv_Z_L)
    BLAS.trmm!('L', 'L', 'T', 'N', T(1), Lγ, obs.Lt_Zt_Dinv_Z_L)
    #Lt_Zt_UUt_Z_L =  Zt_UUt_Z_L
    BLAS.gemm!('N', 'N', T(1), obs.Zt_UUt_Z, Lγ, T(0), obs.Lt_Zt_UUt_Z_L)
    copyto!(obs.Zt_UUt_Z_L, obs.Lt_Zt_UUt_Z_L)
    BLAS.trmm!('L', 'L', 'T', 'N', T(1), Lγ, obs.Lt_Zt_UUt_Z_L)

    #objective function sums
    obj = 0.0
    obj += obs.rt_Dinv_r[1]^2 #1 
    obj -= 2 * obs.rt_UUt_r[1] * obs.rt_Dinv_r[1] #2 
    obj += obs.rt_UUt_r[1]^2 #7
    @inbounds for j in 1:length(obs.expwτ)
        obs.expwτ[j] = exp(obs.expwτ[j])
        obj -= 2 * obs.Dinv_r[j]^2 * obs.expwτ[j] #3
        needgrad ? obs.diagDVRV[j] += obs.Dinv_r[j]^2 * obs.expwτ[j] : nothing#τ 1 
        obj += 2 * obs.rt_UUt[j] * obs.expwτ[j] * obs.Dinv_r[j] #4
        obj += 2 * obs.Dinv_r[j] * obs.rt_UUt[j] * obs.expwτ[j] #8
        needgrad ? obs.diagDVRV[j] -= 2 * obs.Dinv_r[j] * obs.rt_UUt[j] * obs.expwτ[j] : nothing  #τ 2
        obj -= 2 * obs.rt_UUt[j]^2 * obs.expwτ[j] # 9 
        obj += obs.Dinv[j]^2 * obs.expwτ[j]^2 #12 
        needgrad ? obs.diagDVRV[j] -= obs.Dinv[j]^2 * obs.expwτ[j]^2 : nothing#τ 3
        obj -= 2 * obs.diagUUt_Dinv[j] * obs.expwτ[j]^2 #13
        needgrad ? obs.diagDVRV[j] += 2 * obs.diagUUt_Dinv[j] * obs.expwτ[j]^2 : nothing #τ 4
        needgrad ? obs.diagDVRV[j] += obs.rt_UUt[j]^2 * obs.expwτ[j] : nothing #τ 7
        for k in 1:q
            for i in 1:q #evalute/store U' * D * U in O(ni q^2)
                obs.Ut_D_U[i, k] += obs.Ut[k, j] * obs.Ut[i, j] * obs.expwτ[j]
                if needgrad #needed in gradient evlalution # Lγ 9 and 3
                    obs.Ut_D_Dinv_Z[i, k] += obs.Ut[i, j] * obs.Dinv_Z[j, k] * obs.expwτ[j]
                    obs.Zt_Dinv_D_Dinv_Z[i, k] += obs.Dinv_Z[j, k] * obs.Dinv_Z[j, i] * obs.expwτ[j]
                end
            end
        end
    end

    for j in 1:q  #j-i looping for memory access 
        for i in 1:length(obs.expwτ)
            obj += 2 * obs.Dinv_Z_L[i, j]^2 * obs.expwτ[i] # 14
            needgrad ? obs.diagDVRV[i] -= obs.Dinv_Z_L[i, j]^2 * obs.expwτ[i] : nothing #τ 5
            obj -= 2 * obs.UUt_Z_L[i, j] * obs.Dinv_Z_L[i, j] * obs.expwτ[i] # 15
            obj -= 2 * obs.Dinv_Z_L[i, j] * obs.UUt_Z_L[i, j] * obs.expwτ[i] # 17
            needgrad ? obs.diagDVRV[i] += 2 * obs.Dinv_Z_L[i, j] * obs.UUt_Z_L[i, j] * obs.expwτ[i] : nothing #τ 6
            obj += 2 * obs.UUt_Z_L[i, j]^2 * obs.expwτ[i] # 18
            needgrad ? obs.diagDVRV[i] -= obs.UUt_Z_L[i, j]^2 * obs.expwτ[i] : nothing #τ 9
        end
    end
    
    #obs.storage_qq = L'Z'UU'rr'DinvZL
    copyto!(obs.storage_qq, obs.Zt_UUt_rrt_Dinv_Z)
    BLAS.trmm!('L', 'L', 'T', 'N', T(1), Lγ, obs.storage_qq)
    BLAS.trmm!('R', 'L', 'N', 'N', T(1), Lγ, obs.storage_qq) #for #10
    #use ∇Lγ as temporary storage 
    #obs.∇Lγ = L' Zt_UUt_rrt_UUt_Z * L
    copyto!(obs.∇Lγ, obs.Zt_UUt_rrt_UUt_Z)
    BLAS.trmm!('L', 'L', 'T', 'N', T(1), Lγ, obs.∇Lγ)
    BLAS.trmm!('R', 'L', 'N', 'N', T(1), Lγ, obs.∇Lγ) #for #11

    #storage_qn = Ut_D_U_Ut
    BLAS.gemm!('N', 'N', T(1), obs.Ut_D_U, obs.Ut, T(0), obs.storage_qn)

    if needgrad 
        for j in 1:length(obs.expwτ)
            for i in 1:q 
                obs.diagDVRV[j] -= obs.storage_qn[i, j] * obs.Ut[i, j] * obs.expwτ[j] #τ 8
            end
        end
    end
    for j in 1:q
        obj -= 2 * obs.Lt_Zt_Dinv_r[j]^2 #5
        obj += 2 * obs.storage_qq[j, j] #10 
        obj -= 2 * obs.∇Lγ[j, j] # 11 ^*
        for i in 1:q
            obj += obs.Ut_D_U[i, j]^2 #16
            obj += 2 * obs.rt_UUt_Z[i] * Lγ[i, j] * obs.Lt_Zt_Dinv_r[j] #6
            obj += obs.Lt_Zt_Dinv_Z_L[i, j]^2 #19 
            obj -= 2 * obs.Lt_Zt_Dinv_Z_L[i, j] * obs.Lt_Zt_UUt_Z_L[i,j] #20
            obj += obs.Lt_Zt_UUt_Z_L[i,j]^2 #21 
        end
    end

    obj *= (1//2)

    ############
    # Gradient #
    ############
    if needgrad
        #wrt τ
        BLAS.gemv!('N', T(-1), obs.Wt, obs.diagDVRV, T(0), obs.∇τ)

        BLAS.axpby!(T(-2), obs.Zt_Dinv_rrt_Dinv_Z, T(0), obs.∇Lγ) #Lγ 1
        BLAS.axpy!(T(2), obs.Zt_Dinv_rrt_UUt_Z, obs.∇Lγ) #Lγ 2
        BLAS.axpy!(T(2), obs.Zt_Dinv_D_Dinv_Z, obs.∇Lγ) #Lγ 3
        BLAS.gemm!('T', 'T', T(-2), obs.Ut_D_Dinv_Z, obs.Zt_U, T(1), obs.∇Lγ) #Lγ 4
        BLAS.axpy!(T(2), obs.Zt_UUt_rrt_Dinv_Z, obs.∇Lγ) #Lγ 7 #**
        BLAS.axpy!(T(-2), obs.Zt_UUt_rrt_UUt_Z, obs.∇Lγ) #Lγ 8
        BLAS.gemm!('N', 'N', T(-2), obs.Zt_U, obs.Ut_D_Dinv_Z, T(1), obs.∇Lγ) #Lγ 9 
        #storage_qq = Zt_UUt_D_U
        BLAS.gemm!('N', 'N', T(1), obs.Zt_U, obs.Ut_D_U, T(0), obs.storage_qq) 
        BLAS.gemm!('N', 'T', T(2), obs.storage_qq, obs.Zt_U, T(1), obs.∇Lγ) #Lγ 10

        BLAS.trmm!('R', 'L', 'N', 'N', T(1), Lγ, obs.∇Lγ) 
        #after L is multiplied on right 
        BLAS.gemm!('N', 'N', T(2), obs.Zt_Dinv_Z_L, obs.Lt_Zt_Dinv_Z_L, T(1), obs.∇Lγ)#Lγ 5
        BLAS.gemm!('N', 'N', T(-2), obs.Zt_Dinv_Z_L, obs.Lt_Zt_UUt_Z_L, T(1), obs.∇Lγ)#Lγ 6
        BLAS.gemm!('N', 'N', T(-2), obs.Zt_UUt_Z_L, obs.Lt_Zt_Dinv_Z_L, T(1), obs.∇Lγ)#Lγ 11
        BLAS.gemm!('N', 'N', T(2), obs.Zt_UUt_Z_L, obs.Lt_Zt_UUt_Z_L, T(1), obs.∇Lγ)#Lγ 12
    end

    ###########
    # Hessian #
    ###########
    if needhess
        #wrt τ
        # Hττ = W' * D * Vinv .* Vinv * D  * W
        fill!(obs.storage_ln, 0)
        fill!(obs.Wt_D_Dinv, 0)
        fill!(obs.Wt_D_sqrtdiagDinv_UUt, 0)
        #storage_ln = Wt * D
        for j in 1:length(obs.expwτ)
            for i in 1:l
                obs.storage_ln[i, j] += obs.Wt[i, j] * obs.expwτ[j]
                obs.Wt_D_Dinv[i, j] += obs.Dinv[j] * obs.expwτ[j] * obs.Wt[i, j]
                obs.Wt_D_sqrtdiagDinv_UUt[i, j] += obs.Wt[i, j] * obs.expwτ[j] * obs.sqrtDinv_UUt[j]
            end
        end 
        BLAS.syrk!('U', 'N', T(1), obs.Wt_D_Dinv, T(0), obs.Hττ) #first term 
        BLAS.syrk!('U', 'N', T(-2), obs.Wt_D_sqrtdiagDinv_UUt, T(1), obs.Hττ) #second term 
        BLAS.gemm!('N', 'T', T(1), obs.storage_ln, obs.Ut_kr_Ut, T(0), obs.Wt_D_Ut_kr_Utt) 
        BLAS.syrk!('U', 'N', T(1), obs.Wt_D_Ut_kr_Utt, T(1), obs.Hττ) #third term
        copytri!(obs.Hττ, 'U')


        #wrt HτLγ
        # HτLγ = 2 W' * Diagonal(expwτ) * (L'Z'(V^-1) ⊙ Z'(V^-1))' * Cq
        # storage_ln = W' * Diagonal(expwτ) was computed above
        # storage_q◺n = Cq' * (L'Z'(V^-1) ⊙ Z'(V^-1)) 
        copy!(obs.storage_qn, obs.Zt_Vinv)
        BLAS.trmm!('L', 'L', 'T', 'N', T(1), Lγ, obs.storage_qn)
        Ct_A_kr_B!(fill!(obs.storage_q◺n, 0), obs.storage_qn, obs.Zt_Vinv)

        BLAS.gemm!('N', 'T', T(2), obs.storage_ln, obs.storage_q◺n, T(0), obs.HτLγ)

        #wrt HLγLγ
        # HLγLγ = 2 [ C'(L'Z'(V^-1)ZL ⊗ Z'(V^-1)Z)C + C'(L'Z'(V^-1)Z ⊗ Z'(V^-1)ZL)KC ]
        # obs.storage_qq = (Z' (V^-1) Z) * L
        copy!(obs.storage_qq, obs.Zt_Vinv_Z)
        BLAS.trmm!('R', 'L', 'N', 'N', T(1), Lγ, obs.storage_qq)
        # HLγLγ = C'(L'Z' (V^-1) Z ⊗ Z' (V^-1) ZL)KC first
        Ct_At_kron_A_KC!(fill!(obs.HLγLγ, 0), obs.storage_qq)

        # obs.storage_qq = L' * (Z' (V^-1) Z) * L
        BLAS.trmm!('L', 'L', 'T', 'N', T(1), Lγ, obs.storage_qq)

        Ct_A_kron_B_C!(obs.HLγLγ, obs.storage_qq, obs.Zt_Vinv_Z)
        lmul!(2, obs.HLγLγ)
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
in and `VarLmmModel` object `m`. It also precomputes and stores various 
parameters needed to evaluate the objective function, gradient, and Hessian in O(n_i) flops.

"""
function update_wtmat!(m::VarLmmModel{T}) where T <: BlasReal
    # form Dinv - UU^T (Woodbury structure of Vi inverse)
    
    update_res!(m)
    q = size(m.Lγ, 1)
    for obs in m.data
        mul!(obs.expwτ, transpose(obs.Wt), m.τ)
        # Step 1: assemble Ip + Lt Zt diag(e^{-\eta_j}) Z L
        # storage_qn = L' * Z'
        copy!(obs.storage_qn, obs.Zt)
        BLAS.trmm!('L', 'L', 'T', 'N', T(1), m.Lγ, obs.storage_qn)
        # Dinv = diag(e^{-Wτ})
        obs.rt_Dinv_r[1] = 0.0
        @inbounds for j in 1:length(obs.expwτ)
            Dinvjj = exp(-obs.expwτ[j])
            obs.Dinv[j] = Dinvjj
            obs.Dinv_r[j] = Dinvjj * obs.res[j]
            obs.rt_Dinv_r[1] += Dinvjj * obs.res[j]^2
            for i in 1:q
                obs.storage_qn[i, j] *= Dinvjj
            end
        end
        # storage_qq = Lt Zt diag(e^{-wτ}) Z
        BLAS.gemm!('N', 'T', T(1), obs.storage_qn, obs.Zt, T(0), obs.storage_qq)
        # storage_qq = Lt Zt diag(e^{-wτ}) Z L
        BLAS.trmm!('R', 'L', 'N', 'N', T(1), m.Lγ, obs.storage_qq)
        # storage_qq = Iq + Lt Zt diag(e^{-wτ}) Z L 
        @inbounds for i in 1:m.q
            obs.storage_qq[i, i] += 1.0
        end
        # Step 2: invert U^{-1} = (Ip + Lt Zt diag(e^{-Wt}) Z L)^{-1}: cholesky
        cholesky!(Symmetric(obs.storage_qq))

        # Step 3: get storange_qn = U^T = R'^{-1/2} Lt Zt Diagonal(e^{-Wt})
        #storage_qn = Lt Zt Diagonal(e^{-Wτ})
        BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq, obs.storage_qn)
        copyto!(obs.Ut, obs.storage_qn)
        
        #now to precompute components needed for objective function, Hessian, and gradient
        BLAS.gemv!('N', T(1), obs.Zt, obs.Dinv_r, T(0), obs.Zt_Dinv_r)
        #BLAS.gemv!('N', T(1), obs.Ut, obs.res, T(0), obs.rt_U) #*
        BLAS.gemm!('T', 'T', 1.0, obs.res, obs.Ut, T(0), obs.rt_U)
        obs.rt_UUt_r[1] = sum(abs2, obs.rt_U)
        BLAS.gemm!('N', 'N', T(1), obs.rt_U, obs.Ut, T(0), obs.rt_UUt) 
        BLAS.gemm!('N', 'T', T(1), obs.rt_UUt, obs.Zt, T(0), obs.rt_UUt_Z) 
        BLAS.gemv!('N', T(1), obs.Zt, obs.Dinv_r, T(0), obs.Zt_Dinv_r)
        #storage_qq = Ut * Z
        BLAS.gemm!('N', 'T', T(1), obs.Ut, obs.Zt, T(0), obs.storage_qq)
        BLAS.gemm!('T', 'N', T(1), obs.Ut, obs.storage_qq, T(0), obs.UUt_Z)

        fill!(obs.diagUUt_Dinv, 0)
        #storage_qn = Zt * Dinv 
        @inbounds for j in 1:length(obs.expwτ)
            for i in 1:m.q 
                obs.diagUUt_Dinv[j] += obs.Ut[i, j]^2 * obs.Dinv[j]
                obs.storage_qn[i, j] = obs.Zt[i, j] * obs.Dinv[j]
            end
        end
        copyto!(obs.Dinv_Z, transpose(obs.storage_qn))
        #obs.Zt_Vinv = Z' * Dinv
        copyto!(obs.Zt_Vinv, obs.storage_qn)

        BLAS.gemm!('N', 'T', T(1), obs.storage_qn, obs.Zt, T(0), obs.Zt_Dinv_Z)
        #storage_qn = Zt * U 
        BLAS.gemm!('N', 'T', T(1), obs.Zt, obs.Ut, T(0), obs.Zt_U)
        BLAS.syrk!('U', 'N', T(1), obs.Zt_U, T(0), obs.Zt_UUt_Z)
        copytri!(obs.Zt_UUt_Z, 'U')

        #for gradient wrt Lγ
        BLAS.syr!('U', T(1), obs.Zt_Dinv_r, fill!(obs.Zt_Dinv_rrt_Dinv_Z, 0))
        copytri!(obs.Zt_Dinv_rrt_Dinv_Z, 'U')
        BLAS.gemm!('N', 'N', T(1), obs.Zt_Dinv_r, obs.rt_UUt_Z, T(0), obs.Zt_Dinv_rrt_UUt_Z) 
        BLAS.gemm!('T', 'T', T(1), obs.rt_UUt_Z, obs.Zt_Dinv_r, T(0), obs.Zt_UUt_rrt_Dinv_Z)
        BLAS.syrk!('U', 'T', T(1), obs.rt_UUt_Z, T(0), obs.Zt_UUt_rrt_UUt_Z)
        copytri!(obs.Zt_UUt_rrt_UUt_Z, 'U')
        copyto!(obs.Zt_UUt, transpose(obs.UUt_Z)) # need? #*

        #for Hessian wrt τ
        for j in 1:length(obs.expwτ)
            obs.sqrtDinv_UUt[j] = sqrt(obs.diagUUt_Dinv[j])
        end
        fill!(obs.Ut_kr_Ut, 0)
        kr_axpy!(obs.Ut, obs.Ut, obs.Ut_kr_Ut)

        #before this step, obs.Zt_Vinv = Z' * Dinv
        BLAS.axpy!(T(-1), obs.Zt_UUt, obs.Zt_Vinv)
        BLAS.axpby!(T(1), obs.Zt_Dinv_Z, T(0), obs.Zt_Vinv_Z)
        BLAS.axpy!(T(-1), obs.Zt_UUt_Z, obs.Zt_Vinv_Z)

    end
    nothing
end
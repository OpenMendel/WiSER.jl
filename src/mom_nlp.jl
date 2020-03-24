"""
    fit!(m::VarLmmModel, solver=Ipopt.IpoptSolver(print_level=5))

Fit a `VarLMMModel` object by method of moment using nonlinear programming 
solver.
"""
function fit!(
    m::VarLmmModel,
    solver=Ipopt.IpoptSolver(print_level=5)
    #solver=NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=10000)
    #solver=NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=10000)
    )
    q, l = m.q, m.l
    npar = l + ◺(q)
    optm = MathProgBase.NonlinearModel(solver)
    lb = fill(-Inf, npar)
    ub = fill( Inf, npar)
    # offset = l
    # for j in 1:q, i in j:q
    #     i == j && (lb[offset] = 0)
    #     offset += 1
    # end
    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Min, m)
    # starting point
    par0 = zeros(npar)
    modelpar_to_optimpar!(par0, m)
    MathProgBase.setwarmstart!(optm, par0)
    # optimize
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    # update parameters and refresh gradient
    optimpar_to_modelpar!(m, MathProgBase.getsolution(optm))
    # diagonal entries of cholesky factor should be >= 0
    if m.Lγ[1, 1] < 0
        lmul!(-1, m.Lγ)
    end
    mom_obj!(m, true, true, true)
    m
end

"""
    modelpar_to_optimpar!(par, m)

Translate model parameters in `m` to optimization variables in `par`.
"""
function modelpar_to_optimpar!(
    par::Vector,
    m::VarLmmModel{T}
    ) where T <: BlasReal
    q, l = m.q, m.l
    # τ
    copyto!(par, m.τ)
    # Lγ
    offset = l + 1
    @inbounds for j in 1:q, i in j:q
        par[offset] = m.Lγ[i, j]
        offset += 1
    end
    par
end

"""
    optimpar_to_modelpar!(m, par)

Translate optimization variables in `par` to the model parameters in `m`.
"""
function optimpar_to_modelpar!(
    m::VarLmmModel, 
    par::Vector
    )
    q, l = m.q, m.l
    # τ
    copyto!(m.τ, 1, par, 1, l)
    # Lγ
    fill!(m.Lγ, 0)
    offset = l + 1
    @inbounds for j in 1:q, i in j:q
        m.Lγ[i, j] = par[offset]
        offset += 1
    end
    m
end

function MathProgBase.initialize(
    m::VarLmmModel, 
    requested_features::Vector{Symbol}
    )
    for feat in requested_features
        if !(feat in [:Grad, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(m::VarLmmModel) = [:Grad, :Hess]

function MathProgBase.eval_f(
    m::VarLmmModel, 
    par::Vector
    )
    optimpar_to_modelpar!(m, par)
    mom_obj!(m, false, false, false)
end

function MathProgBase.eval_grad_f(
    m::VarLmmModel, 
    grad::Vector, 
    par::Vector
    )
    q, l = m.q, m.l
    optimpar_to_modelpar!(m, par) 
    obj = mom_obj!(m, true, false, false)
    # gradient wrt τ
    copyto!(grad, m.∇τ)
    # gradient wrt Lγ
    offset = l + 1
    @inbounds for j in 1:q, i in j:q
        grad[offset] = m.∇Lγ[i, j]
        offset += 1
    end
    obj
end

MathProgBase.eval_g(m::VarLmmModel, g, par) = nothing
MathProgBase.jac_structure(m::VarLmmModel) = Int[], Int[]
MathProgBase.eval_jac_g(m::VarLmmModel, J, par) = nothing

function MathProgBase.hesslag_structure(m::VarLmmModel)
    # our Hessian is a dense matrix, work on the upper triangular part
    npar = m.l + ◺(m.q)
    arr1 = Vector{Int}(undef, ◺(npar))
    arr2 = Vector{Int}(undef, ◺(npar))
    idx = 1
    for j in 1:npar
        for i in 1:j
            arr1[idx] = i
            arr2[idx] = j
            idx += 1
        end
    end
    return (arr1, arr2)
end

function MathProgBase.eval_hesslag(m::VarLmmModel, H::Vector{T},
    par::Vector{T}, σ::T, μ::Vector{T}) where {T}    
    l, q◺ = m.l, ◺(m.q)
    optimpar_to_modelpar!(m, par)
    mom_obj!(m, true, true, false)
    idx = 1
    @inbounds for j in 1:l, i in 1:j
        H[idx] = m.Hττ[i, j]
        idx += 1
    end
    @inbounds for j in 1:q◺
        for i in 1:l
            H[idx] = m.HτLγ[i, j]
            idx += 1
        end
        for i in 1:j
            H[idx] = m.HLγLγ[i, j]
            idx += 1
        end
    end
    lmul!(σ, H)
end

"""
    init_ls!(m::VarLMMModel)

Initialize parameters of a `VarLMMModel` object from least squares estimate.
"""
function init_ls!(m::VarLmmModel{T}) where T <: BlasReal
    p, q, l = m.p, m.q, m.l
    # LS estimate for β
    xtx = zeros(T, p, p)
    xty = zeros(T, p)
    for i in eachindex(m.data)
        # Xi'Xi
        BLAS.syrk!('U', 'N', T(1), m.data[i].Xt, T(1), xtx)
        # Xi'yi
        BLAS.gemv!('N', T(1), m.data[i].Xt, m.data[i].y, T(1), xty)
    end
    ldiv!(m.β, cholesky!(Symmetric(xtx)), xty)
    update_res!(m)
    # LS etimate for σ2ω
    n, σ2ω = 0, T(0)
    ztz2 = zeros(T, q * q, q * q)
    ztr2 = zeros(T, q * q)
    @inbounds for i in eachindex(m.data)
        σ2ω   += m.data[i].resnrm2[1]
        n     += length(m.data[i].y)
        # Zi'Zi ⊗ Zi'Zi
        ztz    = m.data[i].ztz 
        kron_axpy!(ztz, ztz, ztz2)
        # Zi'res ⊗ Zi'res
        ztres  = m.data[i].ztres
        kron_axpy!(ztres, ztres, ztr2) 
    end
    # WS intercept only model
    fill!(m.τ, 0)
    σ2ω /= n
    m.τ[1] = log(σ2ω)
    # LS estimate for Σγ
    Σγ = reshape(cholesky!(Symmetric(ztz2)) \ ztr2, (q, q))
    copy!(m.Lγ, cholesky!(Symmetric(Σγ), check = false).L)
    m
end

"""
    init_wls!(m::VarLMMModel)

Initialize parameter `β` of a `VarLMMModel` object from weighted least squares 
estimate and update residuals `resi = yi - Xi * β`. 
"""
function init_wls!(m::VarLmmModel{T}) where T <: BlasReal
    p, q, l = m.p, m.q, m.l
    # LS estimate for β
    xtx = zeros(T, p, p)
    xty = zeros(T, p)
    for obs in m.data
        # Step 1: assemble Ip + Lt Zt diag(e^{-\eta_j}) Z L
        # storage_qn = L' * Z'
        copy!(obs.storage_qn, obs.Zt)
        BLAS.trmm!('L', 'L', 'T', 'N', T(1), m.Lγ, obs.storage_qn)
        # storage_qn = Lt Zt Diagonal(e^{-1/2 \eta_j})
        # storage_pn = Xt Diagonal(e^{-1/2 \eta_j})
        # storage_n1 = Diagonal(e^{-1/2 \eta_j}) * y
        mul!(obs.expwτ, transpose(obs.Wt), m.τ)
        @inbounds for j in 1:length(obs.y)
            invsqrtj = exp(- (1//2)obs.expwτ[j])
            for i in 1:q
                obs.storage_qn[i, j] *= invsqrtj
            end
            for i in 1:p
                obs.storage_pn[i, j] = invsqrtj * obs.Xt[i, j]
            end
            obs.storage_n1[j] = invsqrtj * obs.y[j]
        end
        # storage_qq = Ip + Lt Zt diag(e^{-\eta_j}) Z L
        BLAS.syrk!('U', 'N', T(1), obs.storage_qn, T(0), obs.storage_qq)
        @inbounds for i in 1:q
            obs.storage_qq[i, i] += 1
        end
        # Step 2: invert (Ip + Lt Zt diag(e^{-\eta_j}) Z L) by cholesky
        C = cholesky!(Symmetric(obs.storage_qq))
        # Step 3: assemble X' V^{-1} X
        # storage_qn = U'^{-1} Lt Zt Diagonal(e^{-1/2 \eta_j})
        BLAS.trsm!('L', 'U', 'T', 'N', T(1), C.U.data, obs.storage_qn)
        # storage_qp = U'^{-1} Lt Zt Diagonal(e^{-\eta_j}) X
        BLAS.gemm!('N', 'T', T(1), obs.storage_qn, obs.storage_pn, T(0), obs.storage_qp)
        # assemble storage_pp = X' V^{-1} X
        BLAS.syrk!('U', 'N',  T(1), obs.storage_pn, T(0), obs.storage_pp)
        BLAS.syrk!('U', 'T', T(-1), obs.storage_qp, T(1), obs.storage_pp)
        # Step 4: assemble storage_p1 = X' V^{-1} y
        BLAS.gemv!('N',  T(1), obs.storage_pn, obs.storage_n1, T(0), obs.storage_p1)
        BLAS.gemv!('N',  T(1), obs.storage_qn, obs.storage_n1, T(0), obs.storage_q1)
        BLAS.gemv!('T',  T(1), obs.storage_qn, obs.storage_q1, T(0), obs.storage_n1)
        BLAS.gemv!('N', T(-1), obs.storage_pn, obs.storage_n1, T(1), obs.storage_p1)
        # accumulate
        BLAS.axpy!(T(1), obs.storage_pp, xtx)
        BLAS.axpy!(T(1), obs.storage_p1, xty)
    end
    ldiv!(m.β, cholesky!(Symmetric(xtx)), xty)
    update_res!(m)
    m
end

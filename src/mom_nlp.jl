"""
    fit!(m::VarLmmModel, solver=Ipopt.IpoptSolver(print_level=5);
    fittype=:Hybrid, weightedruns=1)

Fit a `VarLMMModel` object by method of moment using nonlinear programming 
solver.

The `fit!()` function takes the following arguments:
* `m::VarLmmModel` the model to fit.
* `solver` by default this is Ipopt.IpoptSolver(print_level=5, watchdog_shortened_iter_trigger=3)
* `fittype` by default this is :Hybrid. Performing the Hybrid fit described below. The other options are :Weighted and :Unweighted.
* `weightedruns` number of weighted runs, by default this is 1.


"""
function fit!(
    m::VarLmmModel,
    solver = Ipopt.IpoptSolver(print_level=5, watchdog_shortened_iter_trigger=3);
    fittype::Symbol = :Hybrid,
    weightedruns::Int = 1)

    # Ensure fittype is correctly specified
    fittype in [:Hybrid, :Weighted, :Unweighted] || 
        throw("fittype $fittype is not valid. Please use one of [:Hybrid, :Weighted, :Unweighted]")

    npar = m.l + ◺(m.q)
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
    init_ls!(m)
    par0 = zeros(npar)
    modelpar_to_optimpar!(par0, m)
    MathProgBase.setwarmstart!(optm, par0)
    # optimize unweighted obj function
    if fittype in [:Hybrid, :Unweighted]
        m.weighted[1] = false
        MathProgBase.optimize!(optm)
        optimpar_to_modelpar!(m, MathProgBase.getsolution(optm))
    end
    init_wls!(m)
    if fittype in [:Hybrid, :Weighted]
        m.weighted[1] = true
        for run in 1:weightedruns 
            update_wtmat!(m)
            modelpar_to_optimpar!(par0, m)
            MathProgBase.setwarmstart!(optm, par0)
            MathProgBase.optimize!(optm)
            optimpar_to_modelpar!(m, MathProgBase.getsolution(optm))
        end
    end
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    # update parameters and refresh gradient
    optimpar_to_modelpar!(m, MathProgBase.getsolution(optm))
    # diagonal entries of cholesky factor should be >= 0
    if m.Lγ[1, 1] < 0
        lmul!(-1, m.Lγ)
    end
    mom_obj!(m, true, true, false)
    get_inference(m)
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
estimate, updates residuals `resi = yi - Xi * β`, and updates model/observation fields 
`m.XtVinvX` and `m.data[i].XtVinvres` for inference. 
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
            obs.storage_qq[i, i] += 1.0
        end
        # Step 2: invert (Ip + Lt Zt diag(e^{-\eta_j}) Z L) by cholesky
        LAPACK.potrf!('U', obs.storage_qq)
        # Step 3: assemble X' V^{-1} X
        # storage_qn = U'^{-1} Lt Zt Diagonal(e^{-1/2 \eta_j})
        BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq, obs.storage_qn)
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
        copytri!(obs.storage_pp, 'U')
        BLAS.axpy!(T(1), obs.storage_pp, xtx)
        BLAS.axpy!(T(1), obs.storage_p1, xty)
    end
    copyto!(m.XtVinvX, xtx)
    ldiv!(m.β, cholesky!(Symmetric(xtx)), xty)
    #for inference 
    for obs in m.data #X' V^{-1} (y - Xβ) = X' V^{-1} y - X' V^{-1} X β
        BLAS.axpby!(T(1), obs.storage_p1, T(0), obs.XtVinvres)
        BLAS.gemv!('N', T(-1), obs.storage_pp, m.β, T(1), obs.XtVinvres)
    end
    update_res!(m)
    m
end


"""
    update_var!(m::VarLMMModel)

Updates model field `m.V` with asymptotic variance of the parameter estimates.
"""
function update_var!(m::VarLmmModel{T}) where T <: BlasReal
    p, q, l = m.p, m.q, m.l
    q◺ = ◺(q)
    mtotal = length(m.data)
    divm = 1 / mtotal
    fill!(m.Ainv, 0)
    fill!(m.B, 0)
    gradvec = Vector{T}(undef, p + q◺ + l)

    #form A matrix
    copyto!(m.Ainv, CartesianIndices((1:p, 1:p)), 
            m.XtVinvX, CartesianIndices((1:p, 1:p)))
    copyto!(m.Ainv, CartesianIndices(((p + 1):(p + l), 
        (p + 1):(p + l))), m.Hττ, CartesianIndices((1:l, 1:l)))
    copyto!(m.Ainv, CartesianIndices(((p + 1):(p + l), 
        (p + l + 1):(p + l + q◺))), m.HτLγ, CartesianIndices((1:l, 1:q◺)))
    copyto!(m.Ainv, CartesianIndices(((p + l + 1):(p + l + q◺), 
        (p + 1):(p + l))), transpose(m.HτLγ), CartesianIndices((1:q◺, 1:l)))
    copyto!(m.Ainv, CartesianIndices(((p + l + 1):(p + l + q◺), 
    (p + l + 1):(p + l + q◺))), m.HLγLγ, CartesianIndices((1:q◺, 1:q◺)))

    #form B matrix 
    for ob in m.data
        copyto!(gradvec, 1, ob.XtVinvres)
        copyto!(gradvec, 1 + p, ob.∇τ)
        offset = p + l + 1
        @inbounds for j in 1:q, i in j:q
            gradvec[offset] = ob.∇Lγ[i, j]
            offset += 1
        end

        BLAS.syr!('U', T(1), gradvec, m.B)
    end
    copytri!(m.B, 'U')

    lmul!(divm, m.Ainv)
    lmul!(divm, m.B)

    #Calculuate A inverse 
    LAPACK.potrf!('U', m.Ainv)
    # put here 
    LAPACK.potri!('U', m.Ainv)

    #Calculate V 
    copytri!(m.Ainv, 'U')
    mul!(m.AinvB, m.Ainv, m.B)
    BLAS.symm!('R', 'U', T(1), m.Ainv, m.AinvB, zero(T), m.V)
    copytri!(m.V, 'U')
    
end

"""
    get_inference!(m::VarLMMModel)

Returns inference of parameter estimates based on asymptotic normality of M-estimator framework. 
"""
function get_inference(m::VarLmmModel{T}) where T <: BlasReal
    update_var!(m)
    mtotal = length(m.data)
    pars = [m.β; m.τ]
    npars = length([m.β; m.τ])
    diagV = diag(m.V)[1:npars]
    names = [m.meannames; m.wsvarnames]
    if any(diagV .< 0)
        @warn "Asymptotic Variance is negative, cannot give valid inference"
        return nothing
    end
    stder = sqrt.(diagV ./ mtotal)
    wald = mtotal .* ([m.β; m.τ].^2 ./ diagV)
    pvals = Distributions.ccdf.(Chisq(1), wald)
    
    StatsModels.CoefTable(hcat(pars, stder, wald, pvals),
        ["Estimate", "Std. Error", "Wald Statistic", "Pr(>|Wald|)"],
        names, 4, 3)
end

# function Base.show(io::IO, m::VarLmmModel)
#     p, q, l = m.p, m.q, m.l
#     println(io, "Variance linear mixed model fit by method of moments")
#     #println(io, " ", m.formula)
    
#     #include fit info here?

#     show(io, VarCorr(m))
#     println("Σγ : Random Effects Covariance Matrix")
#     show(io, m.Lγ * transpose(m.Lγ))

#     print(io, " Number of individuals/clusters: $(m.m); total observations: $(m.obs)")
#     println(io, "\n  Fixed-effects parameters:")
#     # show(io, coeftable(m)) #write a coeftable function to replace below
#     get_inference(m)
# end



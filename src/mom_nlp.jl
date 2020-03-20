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
    npar = l + (q * (q + 1)) >> 1
    optm = MathProgBase.NonlinearModel(solver)
    lb = fill(-Inf, npar)
    ub = fill( Inf, npar)
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
        BLAS.syrk!('U', 'N', T(1), m.data[i].Xt, T(1), xtx)
        BLAS.gemv!('N', T(1), m.data[i].Xt, m.data[i].y, T(1), xty)
    end
    ldiv!(m.β, cholesky!(Symmetric(xtx)), xty)
    update_res!(m)
    # LS etimate for σ2ω
    n, σ2ω = 0, T(0)
    @inbounds for i in eachindex(m.data)
        σ2ω += m.data[i].resnrm2[1]
        n   += length(m.data[i].y)
    end
    σ2ω /= n
    # WS intercept only model
    fill!(m.τ, 0)
    m.τ[1] = log(σ2ω)
    # LS estimate for Σγ
    ztz2 = zeros(T, q * q, q * q)
    ztr2 = zeros(T, q * q)
    @inbounds for i in eachindex(m.data)
        ztz    = m.data[i].ztz 
        kron_axpy!(ztz, ztz, ztz2)
        ztres  = m.data[i].ztres
        kron_axpy!(ztres, ztres, ztr2)
    end
    Σγ = reshape(cholesky!(Symmetric(ztz2)) \ ztr2, (q, q))
    copy!(m.Lγ, cholesky!(Symmetric(Σγ), check = false).L)
    m
end

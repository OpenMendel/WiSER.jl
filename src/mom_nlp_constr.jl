"""
TODO
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
        m.Lγ .*= -1
    end
    mom_obj!(m, true, true)
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
        if !(feat in [:Grad])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(m::VarLmmModel) = [:Grad]

function MathProgBase.eval_f(
    m::VarLmmModel, 
    par::Vector
    )
    optimpar_to_modelpar!(m, par)
    mom_obj!(m, false, false)
end

function MathProgBase.eval_grad_f(
    m::VarLmmModel, 
    grad::Vector, 
    par::Vector
    )
    q, l = m.q, m.l
    optimpar_to_modelpar!(m, par) 
    obj = mom_obj!(m, true, false)
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

MathProgBase.eval_g(m::VarLmmModel, g, par) = fill!(g, 0)

"""
    init_ls!(m::VarLMM)

Initialize parameter from least squares estimate.
TODO: need to optimize the code
"""
function init_ls!(m::VarLmmModel{T}) where T <: BlasReal
    p, q, l = m.p, m.q, m.l
    # LS estimate for β
    xtx = zeros(T, p, p)
    xty = zeros(T, p)
    for i in eachindex(m.data)
        xtx .+= m.data[i].X'm.data[i].X
        xty .+= m.data[i].X'm.data[i].y
    end
    copy!(m.β, cholesky(Symmetric(xtx)) \ xty)
    update_res!(m)
    # LS etimate for σ2ω
    σ2ω = T(0)
    n = 0
    for i in eachindex(m.data)
        σ2ω += abs2(norm(m.data[i].res))
        n   += length(m.data[i].y)
    end
    σ2ω /= n
    # WS intercept only model
    fill!(m.τ, 0)
    m.τ[1] = log(σ2ω)
    # LS estimate for Σγ
    ztz2 = zeros(T, q * q, q * q)
    ztr2 = zeros(T, q * q)
    for i in eachindex(m.data)
        ztz    = m.data[i].Z'm.data[i].Z 
        ztz2 .+= kron(ztz, ztz)
        ztr    = m.data[i].Z'm.data[i].res
        ztr2 .+= kron(ztr, ztr)
    end
    Σγ = reshape(cholesky(Symmetric(ztz2)) \ ztr2, (q, q))
    copy!(m.Lγ, cholesky(Symmetric(Σγ), check = false).L)
    m
end

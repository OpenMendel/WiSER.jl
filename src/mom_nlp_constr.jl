"""
TODO
"""
function fit!(
    m::VarLmmModel,
    solver=Ipopt.IpoptSolver(print_level=5)
    #solver=NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=10000)
    #solver=NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=10000)
    )
    p, q, l, npar = m.p, m.q, m.l, m.npar
    optm = MathProgBase.NonlinearModel(solver)
    # diagonal entries of Cholesky factor is lower bounded by  0
    lb = fill(-Inf, npar)
    ub = fill( Inf, npar)
    offset = p + l + 1
    for j in 1:q, i in j:q
        (i == j) && (lb[offset] = 0)
        offset += 1
    end
    # lb[1:p] .= m.β
    # ub[1:p] .= m.β
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
    mom_obj!(m, true)
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
    p, q, l = m.p, m.q, m.l
    # β, τ
    copyto!(par, m.β)
    copyto!(par, p + 1, m.τ) 
    # Lγ
    offset = p + l + 1
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
    p, q, l = m.p, m.q, m.l
    # β, τ
    copyto!(m.β, 1, par,     1, p)
    copyto!(m.τ, 1, par, p + 1, l)
    # Lγ
    fill!(m.Lγ, 0)
    offset = p + l + 1
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
    mom_obj!(m, false)
end

function MathProgBase.eval_grad_f(
    m::VarLmmModel, 
    grad::Vector, 
    par::Vector
    )
    p, q, l = m.p, m.q, m.l
    optimpar_to_modelpar!(m, par) 
    obj = mom_obj!(m, true)
    # gradient wrt β
    copyto!(grad, m.∇β)
    # gradient wrt τ
    copyto!(grad, p + 1, m.∇τ)
    # gradient wrt Lγ
    offset = p + l + 1
    @inbounds for j in 1:q, i in j:q
        grad[offset] = m.∇Lγ[i, j]
        offset += 1
    end
    obj
end

MathProgBase.eval_g(m::VarLmmModel, g, par) = fill!(g, 0)

"""
    fit!(m::VarLmmModel, solver=Ipopt.IpoptSolver(print_level=5);
    fittype=:Hybrid, weightedruns=1)

Fit a `VarLMMModel` object by method of moment using nonlinear programming 
solver.

The `fit!()` function takes the following arguments:
* `m::VarLmmModel` the model to fit.
* `solver` by default this is Ipopt.IpoptSolver(print_level=5, watchdog_shortened_iter_trigger=3)
* `fittype` by default this is :Hybrid. The other options are :Weighted and :Unweighted.
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
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat. It may be worth trying to change `fittype=:Weighted`.")
    # update parameters and refresh gradient
    optimpar_to_modelpar!(m, MathProgBase.getsolution(optm))
    # diagonal entries of cholesky factor should be >= 0
    if m.Lγ[1, 1] < 0
        lmul!(-1, m.Lγ)
    end
    mom_obj!(m, true, true, false)
    update_var!(m)
    mul!(m.Σγ, m.Lγ, transpose(m.Lγ))
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


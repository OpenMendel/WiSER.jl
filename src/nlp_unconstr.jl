"""
    fit!(m::VarLmmModel, solver=Ipopt.IpoptSolver(print_level=5);
    init = init_ls!(m), runs = 1)

Fit a `VarLMMModel` object using a weighted NLS method.

# Positional arguments
- `m::VarLmmModel`: Model to fit.
- `solver`: Default is `Ipopt.IpoptSolver(print_level=5, 
    watchdog_shortened_iter_trigger=3)`.

# Keyword arguments
- `init`: Initialization strategy. `fit!` will use `m.τ` and `m.Lγ` to set the 
    weight matrices `Vi` and solve the weighted NLS to obtain an
    estimate for `m.β`, `m.τ`, and `m.Lγ`.  Choices for `init` include  
    - `init_ls!(m)` (default): initialize by the least squares analytical solution.  
    - `init_mom!(m)`: initialize by the unweighted NLS (MoM).  
    - `m`: initilize from user supplied values in `m.τ` and `m.Lγ`.
- `runs`: Number of weighted NLS runs; default is 2. Each run will use the 
    newest `m.τ` and `m.Lγ` to update the weight matrices `Vi` and solve the 
    new weighted NLS.
"""
function fit!(
    m::VarLmmModel,
    solver = Ipopt.IpoptSolver(print_level=5, 
            watchdog_shortened_iter_trigger=3);
    init = init_ls!(m),
    runs :: Integer = 2)
    # set up NLP optimization problem
    npar = m.l + ◺(m.q)
    optm = MathProgBase.NonlinearModel(solver)
    lb   = fill(-Inf, npar)
    ub   = fill( Inf, npar)
    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Min, m)
    par0 = Vector{Float64}(undef, npar)
    # optimize weighted obj function
    m.iswtnls[1] = true
    for run in 1:runs
        update_wtmat!(m)
        modelpar_to_optimpar!(par0, m)
        MathProgBase.setwarmstart!(optm, par0)
        MathProgBase.optimize!(optm)
        optimpar_to_modelpar!(m, MathProgBase.getsolution(optm))
        optstat = MathProgBase.status(optm)
        optstat == :Optimal || 
        @warn("Optimization unsuccesful; got $optstat; run = $run")
    end
    # refresh objective, gradient, and Hessian
    mul!(m.Σγ, m.Lγ, transpose(m.Lγ))
    mom_obj!(m, true, true, false)
    # sandwich estimator
    sandwich!(m)
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
        par[offset] = i == j ? m.Lγ[i, j] == 0 ? -1e19 : log(m.Lγ[i, j]) : m.Lγ[i, j]
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
        m.Lγ[i, j] = i == j ? exp(par[offset]) : par[offset]
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
    @inbounds for j in 1:q, i in j:q # traverse lower triangular of Lγ[i, j]
        grad[offset] = m.∇Lγ[i, j]
        i == j && (grad[offset] *= m.Lγ[i, j])
        offset += 1
    end
end

MathProgBase.eval_g(m::VarLmmModel, g, par) = nothing
MathProgBase.jac_structure(m::VarLmmModel) = Int[], Int[]
MathProgBase.eval_jac_g(m::VarLmmModel, J, par) = nothing

function MathProgBase.hesslag_structure(m::VarLmmModel)
    # our Hessian is a dense matrix, work on the upper triangular part
    npar = m.l + ◺(m.q)
    arr1 = Vector{Int}(undef, ◺(npar))
    arr2 = Vector{Int}(undef, ◺(npar))
    idx  = 1
    @inbounds for j in 1:npar, i in 1:j
        arr1[idx] = i
        arr2[idx] = j
        idx      += 1
    end
    return (arr1, arr2)
end

function MathProgBase.eval_hesslag(m::VarLmmModel, H::Vector{T},
    par::Vector{T}, σ::T, μ::Vector{T}) where {T}
    q, l = m.q, m.l
    # refresh obj, gradient, and hessian
    optimpar_to_modelpar!(m, par)
    mom_obj!(m, true, true, false)
    # Hττ
    idx = 1
    @inbounds for j in 1:l, i in 1:j
        H[idx] = m.Hττ[i, j]
        idx   += 1
    end
    j = 1 # index columns of HτLγ and HLγLγ
    @inbounds for j2 in 1:q, j1 in j2:q # traverse lower triangular of Lγ[j1, j2]
        # HτLγ
        for i in 1:l # i index rows of HτLγ
            H[idx] = m.HτLγ[i, j]
            j1 == j2 && (H[idx] *= m.Lγ[j1, j2])
            idx += 1
        end
        # HLγLγ
        i = 1 # index rows of HLγLγ
        for i2 in 1:q, i1 in i2:q # traverse lower triangular of Lγ[i1, i2]
            i > j && break # skip lower triangular of HLγLγ
            H[idx] = m.HLγLγ[i, j]
            # different diagonal entries of Lγ
            i1 == i2 && (H[idx] *= m.Lγ[i1, i2])
            j1 == j2 && (H[idx] *= m.Lγ[j1, j2])
            # same diagonal entry of Lγ
            i1 == i2 == j1 == j2 && (H[idx] += m.∇Lγ[j1, j2] * m.Lγ[j1, j2])
            idx += 1
            i   += 1
        end
        j += 1
    end
    lmul!(σ, H)
end

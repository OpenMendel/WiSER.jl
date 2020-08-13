"""
    fit!(m::WSVarLmmModel, 
    solver=IpoptSolver(print_level=0, mehrotra_algorithm="yes", max_iter=100);
    init=init_ls!(m), runs = 2)

Fit a `WSVarLMMModel` object using a weighted NLS method.

# Positional arguments
- `m::WSVarLmmModel`: Model to fit.
- `solver`: Nonlinear programming solver to use. Common choices include:  
    - `Ipopt.IpoptSolver(print_level=0, mehrotra_algorithm="yes", max_iter=100)`.
    - `Ipopt.IpoptSolver(print_level=0, mehrotra_algorithm="yes", warm_start_init_point="yes", max_iter=100)`.
    - `Ipopt.IpoptSolver(print_level=0, watchdog_shortened_iter_trigger=3, max_iter=100)`.
    - `KNITRO.KnitroSolver(outlev=3)`. (Knitro is commercial software)
    - `NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000)`.  
    - `NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxeval=4000)`.

# Keyword arguments
- `init`: Initialization strategy. `fit!` will use `m.τ` and `m.Lγ` to set the 
    weight matrices `Vi` and solve the weighted NLS to obtain an
    estimate for `m.β`, `m.τ`, and `m.Lγ`.  Choices for `init` include  
    - `init_ls!(m)` (default): initialize by the least squares analytical solution.  
    - `init_mom!(m)`: initialize by the unweighted NLS (MoM).  
    - `m`: initilize from user supplied values in `m.τ` and `m.Lγ`.
- `runs::Integer`: Number of weighted NLS runs; default is 2. Each run will use the 
    newest `m.τ` and `m.Lγ` to update the weight matrices `Vi` and solve the 
    new weighted NLS.
- `parallel::Bool`: Multi-threading or not. Default is `false`. 
- `verbose::Bool`: Verbose display or not, Default is `true`.
"""
function fit!(
    m        :: WSVarLmmModel,
    solver = Ipopt.IpoptSolver(print_level=0, mehrotra_algorithm = "yes",
    warm_start_init_point="yes", max_iter=100);
    init     :: WSVarLmmModel = init_ls!(m),
    runs     :: Integer = 2,
    parallel :: Bool = false,
    verbose  :: Bool = true)
    # set up NLP optimization problem
    npar = m.l + ◺(m.q)
    optm = MathProgBase.NonlinearModel(solver)
    lb   = fill(-Inf, npar)
    ub   = fill( Inf, npar)
    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Min, m)
    par0 = Vector{Float64}(undef, npar)
    # optimize weighted NLS
    m.iswtnls[1] = true
    m.ismthrd[1] = parallel
    βprev, τprev, Lγprev = similar(m.β), similar(m.τ), similar(m.Lγ)
    for run in 1:runs
        copyto!(βprev, m.β); copyto!(τprev, m.τ), copyto!(Lγprev, m.Lγ)
        tic = time() # start timing
        # update Vi, then β and residuals with WLS
        update_wtmat!(m)
        # update τ and Lγ by WNLS        
        modelpar_to_optimpar!(par0, m)
        MathProgBase.setwarmstart!(optm, par0)
        MathProgBase.optimize!(optm)
        optimpar_to_modelpar!(m, MathProgBase.getsolution(optm))
        toc = time()
        optstat = MathProgBase.status(optm)
        optstat == :Optimal || 
        @warn("Optimization unsuccesful; got $optstat; run = $run")
        verbose && @printf(
            "run = %d, ‖Δβ‖ = %f, ‖Δτ‖ = %f, ‖ΔL‖ = %f, status = %s, time(s) = %f\n", 
            run, 
            norm(m.β  - βprev ), 
            norm(m.τ  - τprev ),
            norm(m.Lγ - Lγprev),
            optstat,
            toc - tic)
    end
    m.isfitted[1] = true
    # refresh objective, gradient, and Hessian
    mul!(m.Σγ, m.Lγ, transpose(m.Lγ))
    nlsv_obj!(m, true, true, false)
    # sandwich estimator
    sandwich!(m)
    m
end

"""
    modelpar_to_optimpar!(par, m)

Translate model parameters in `m` to optimization variables in `par`.
"""
function modelpar_to_optimpar!(
    par :: Vector,
    m   :: WSVarLmmModel{T}
    ) where T <: BlasReal
    q, l = m.q, m.l
    # τ
    copyto!(par, m.τ)
    # Lγ
    offset = l + 1
    @inbounds for j in 1:q, i in j:q
        par[offset] = i == j ? log(max(m.Lγ[i, j], floatmin(T))) : m.Lγ[i, j]
        offset += 1
    end
    par
end

"""
    optimpar_to_modelpar!(m, par)

Translate optimization variables in `par` to the model parameters in `m`.
"""
function optimpar_to_modelpar!(
    m   :: WSVarLmmModel, 
    par :: Vector
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
    m                  :: WSVarLmmModel, 
    requested_features :: Vector{Symbol}
    )
    for feat in requested_features
        if !(feat in [:Grad, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(m::WSVarLmmModel) = [:Grad, :Hess]

function MathProgBase.eval_f(
    m   :: WSVarLmmModel, 
    par :: Vector
    )
    optimpar_to_modelpar!(m, par)
    nlsv_obj!(m, false, false, false)
end

function MathProgBase.eval_grad_f(
    m    :: WSVarLmmModel, 
    grad :: Vector, 
    par  :: Vector
    )
    q, l = m.q, m.l
    optimpar_to_modelpar!(m, par) 
    obj = nlsv_obj!(m, true, false, false)
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

MathProgBase.eval_g(m::WSVarLmmModel, g, par) = nothing
MathProgBase.jac_structure(m::WSVarLmmModel) = Int[], Int[]
MathProgBase.eval_jac_g(m::WSVarLmmModel, J, par) = nothing

function MathProgBase.hesslag_structure(m::WSVarLmmModel)
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

function MathProgBase.eval_hesslag(m::WSVarLmmModel, H::Vector{T},
    par::Vector{T}, σ::T, μ::Vector{T}) where {T}
    q, l = m.q, m.l
    # refresh obj, gradient, and hessian
    optimpar_to_modelpar!(m, par)
    nlsv_obj!(m, true, true, false)
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

"""
    init_ls!(m::WSVarLmmModel; gniters::Integer = 5)

Initialize parameters of a `WSVarLmmModel` object from least squares estimate. 
`m.β`  is initialized to be `inv(sum(xi'xi)) * sum(xi'yi)`. 
`m.Σγ` is initialized to be `inv(sum(zi'zi⊗zi'zi)) * sum(zi'ri⊗zi'ri)`.  
`m.τ`  is initialized to be `inv(sum(wi'wi)) * sum(wi'log(abs2(ri)))`.  
If `gniters > 0`, run `gniters` Gauss-Newton iterations to improve τ.
"""
function init_ls!(
    m       :: WSVarLmmModel{T};
    gniters :: Integer = 5
    ) where T <: BlasReal
    # dimensions
    q, l = m.q, m.l
    # LS estimate for β
    _, info = LAPACK.potrf!('U', copyto!(m.Hββ, m.xtx))
    info > 0 && throw("design matrix X is rank deficient")
    LAPACK.potrs!('U', m.Hββ, copyto!(m.β, m.xty))
    # refresh residuals
    update_res!(m)
    # accumulate quantities for initilizing Σγ and τ
    fill!(m.∇τ , 0) # to accumulate Wi' * log(ri.^2)
    fill!(m.∇Σγ, 0) # to accumulate Zi'ri ⊗ Zi'ri
    fill!(m.Lγ , 0)  # scratch space to accumulate Zi'diag(r) diag(r)Zi
    for obs in m.data
        n = length(obs.y)
        # storage_n1 = log(diag(rr'))
        map!(r2 -> log(max(r2, floatmin(T))), obs.storage_n1, obs.res2)
        # accumulate Wi' * log(ri.^2)
        BLAS.gemv!('N', T(1), obs.Wt, obs.storage_n1, T(1), m.∇τ)
        # accumulate Zi'ri ⊗ Zi'ri
        kron_axpy!(obs.ztres, obs.ztres, m.∇Σγ)
        # storage_qn = Zi'diag(r)
        copyto!(obs.storage_qn, obs.Zt)
        @inbounds for j in 1:n, i in 1:q
            obs.storage_qn[i, j] *= obs.res[j]
        end
        # accmulate vec(Zi'diag(r) diag(r)Zi)
        BLAS.syrk!('U', 'N', T(1), obs.storage_qn, T(1), m.Lγ)
    end
    copytri!(m.Lγ, 'U')
    # LS estimate for Σγ
    _, info = LAPACK.potrf!('U', copyto!(m.HΣγΣγ, m.ztz2od))
    info > 0 && throw("design matrix Z is rank defficient")
    # sum_i (Zi'ri ⊗ Zi'ri - vec(Zi'diag(r) diag(r)Zi))
    @inbounds for i in eachindex(m.∇Σγ) 
        m.∇Σγ[i] -= m.Lγ[i]
    end
    LAPACK.potrs!('U', m.HΣγΣγ, m.∇Σγ)
    _, info = LAPACK.potrf!('L', copyto!(m.Lγ, m.∇Σγ))
    # make upper triangular of Lγ zero
    @inbounds for j in 2:q, i in 1:j-1 
        m.Lγ[i, j] = 0
    end
    # Σγ is singular; set columns L[:, info:end] = 0
    if info > 0
        @warn("starting Lγ is rank deficient")
        @inbounds for j in info:q, i in j:q
            m.Lγ[i, j] = 0
        end
    end
    # regress log(ri .* ri) on Wi to initialize τ
    _, info = LAPACK.potrf!('U', copyto!(m.Hττ, m.wtw))
    info > 0 && throw("design matrix W is singular")
    LAPACK.potrs!('U', m.Hττ, copyto!(m.τ, m.∇τ))
    # quick return if no GN iterations requested
    gniters == 0 && (return m)
    # Gauss-Newton iterations to improve τ
    # NLS responses: obs.storage_n1 = res^2 - diag(Z L Lt Zt)
    for obs in m.data
        n = length(obs.y)
        # storage_qn = Lγ' * Zt
        mul!(obs.storage_qn, transpose(m.Lγ), obs.Zt)
        # storage_n1 = diag(rr' - Z * L * L' * Zt)
        @inbounds for j in 1:n
            obs.storage_n1[j] = obs.res2[j]
            for i in 1:q
                obs.storage_n1[j] -= abs2(obs.storage_qn[i, j])
            end
        end
    end
    # Gauss-Newton iterations
    for iter in 1:gniters
        # accumulate ∇ and FIM 
        fill!(m.∇τ, 0)
        fill!(m.Hττ, 0)
        for obs in m.data
            n = length(obs.y)
            mul!(obs.expwτ, transpose(obs.Wt), m.τ)
            obs.expwτ .= exp.(obs.expwτ)
            # storage_ln = Wt * diag(expwτ)
            copyto!(obs.storage_ln, obs.Wt)
            @inbounds for j in 1:n, i in 1:l
                obs.storage_ln[i, j] *= obs.expwτ[j]
            end
            # ∇i = Wt * diag(expwτ) * (ypseudo - expwτ)
            # expwτ = ypseudo - expwτ
            obs.expwτ .= obs.storage_n1 .- obs.expwτ
            BLAS.gemv!('N', T(1), obs.storage_ln, obs.expwτ, T(1), m.∇τ)
            # Hi = Wt * diag(expwτ) * diag(expwτ) * W
            BLAS.syrk!('U', 'N', T(1), obs.storage_ln, T(1), m.Hττ)
        end
        # Gauss-Newton update
        LAPACK.potrf!('U', m.Hττ)
        LAPACK.potrs!('U', m.Hττ, m.∇τ) # newton direction
        m.τ .+= m.∇τ
    end
    mul!(m.Σγ, m.Lγ, transpose(m.Lγ))
    m
end


# """
#     init_mom!(m::WSVarLmmModel, solver; init = init_ls!(m), parallel = false)

# Initialize `τ` and `Lγ` of a `VarLmmModel` object by method of moment (MoM) 
# using residulas in `m.obs[i].res`. It involves solving an unweighted NLS problem.

# # Position arguments
# - `m`: A `WSVarLmmModel` object.
# - `solver`: NLP solver. Default is `IpoptSolver(print_level=0, mehrotra_algorithm="yes", 
#     warm_start_init_point="yes", max_iter=100)`.

# # Keyword arguments
# - `init`: Initlizer for the NLS problem. Default is `init_ls!(m)`. If `init=m`, 
# then it uses the values provided in `m.τ` and `m.Lγ` as starting point.  
# - `parallel::Bool`: Multi-threading. Default is `false`.
# """
# function init_mom!(
#     m        :: WSVarLmmModel{T},
#     solver = Ipopt.IpoptSolver(print_level = 0, mehrotra_algorithm = "yes", 
#         warm_start_init_point = "yes",
#         warm_start_bound_push = 1e-6, max_iter = 100);
#     init     :: WSVarLmmModel = init_ls!(m),
#     parallel :: Bool = false
#     ) where T <: BlasReal
#     # set up NLP optimization problem
#     npar = m.l + ◺(m.q)
#     optm = MathProgBase.NonlinearModel(solver)
#     lb = fill(-Inf, npar)
#     ub = fill( Inf, npar)
#     MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Min, m)
#     # optimize unweighted obj function (MoM estimator)
#     m.iswtnls[1] = false
#     m.ismthrd[1] = parallel
#     par0 = zeros(npar)
#     modelpar_to_optimpar!(par0, m)
#     MathProgBase.setwarmstart!(optm, par0)
#     MathProgBase.optimize!(optm)
#     optimpar_to_modelpar!(m, MathProgBase.getsolution(optm))
#     optstat = MathProgBase.status(optm) 
#     optstat == :Optimal || 
#     @warn("Optimization unsuccesful; got $optstat")
#     mul!(m.Σγ, m.Lγ, transpose(m.Lγ))
#     m
# end

"""
    init_mom!(m::WSVarLmmModel, solver; init = init_ls!(m), parallel = false)

Initialize `τ` and `Lγ` of a `VarLmmModel` object by method of moment (MoM) 
using residulas in `m.obs[i].res`. It involves solving an unweighted NLS problem.

# Position arguments
- `m`: A `WSVarLmmModel` object.
- `solver`: NLP solver. Default is `IpoptSolver(print_level=0, mehrotra_algorithm="yes", 
    warm_start_init_point="yes", max_iter=100)`.

# Keyword arguments
- `init`: Initlizer for the NLS problem. Default is `init_ls!(m)`. If `init=m`, 
then it uses the values provided in `m.τ` and `m.Lγ` as starting point.  
- `parallel::Bool`: Multi-threading. Default is `false`.
"""
function init_mom!(
    m        :: WSVarLmmModel{T},
    solver   :: MOI.AbstractOptimizer = Ipopt.Optimizer();
    solver_config::Dict = 
        Dict("print_level"           => 0, 
             "mehrotra_algorithm"    => "yes",
             "warm_start_init_point" => "yes",
             "max_iter"              => 100),
    init     :: WSVarLmmModel = init_ls!(m),
    parallel :: Bool = false
    ) where T <: BlasReal
    # Pass options to solver
    config_solver(solver, solver_config)
    # set up NLP optimization problem
    npar = m.l + ◺(m.q)
    MOI.empty!(solver)
    lb = T[]
    ub = T[]

    NLPBlock = MOI.NLPBlockData(
        MOI.NLPBoundsPair.(lb, ub), m, true
    )
    
    par0 = Vector{T}(undef, npar)
    modelpar_to_optimpar!(par0, m)

    solver_pars = MOI.add_variables(solver, npar)
    for i in 1:npar
        MOI.set(solver, MOI.VariablePrimalStart(), solver_pars[i], par0[i])
    end

    MOI.set(solver, MOI.NLPBlock(), NLPBlock)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # optimize unweighted obj function (MoM estimator)
    m.iswtnls[1] = false
    m.ismthrd[1] = parallel
    
    MOI.optimize!(solver)
    
    # output
    optstat = MOI.get(solver, MOI.TerminationStatus())
    optstat in (MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED) || 
        @warn("Optimization unsuccessful; got $optstat")
    xsol = similar(par0)
    for i in eachindex(xsol)
        xsol[i] = MOI.get(solver, MOI.VariablePrimal(), MOI.VariableIndex(i))
    end
    optimpar_to_modelpar!(m, xsol)

    mul!(m.Σγ, m.Lγ, transpose(m.Lγ))
    m
end
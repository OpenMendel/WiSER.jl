export MixWildObs, MixWildModel
export loglikelihood!, setparameters!

"""
    MixWildObs
    MixWildObs(y, X, Z, W)

A realization of multiple location scale linear mixed model data instance.

# Positional Arguments

- `y`: the response vector
- `X`: the mean fixed effects covariate matrix 
- `Z`: the random location effects covariate matrix
- `W`: the within-subject variance fixed effects covariate matrix
"""
struct MixWildObs{T <: BlasReal}
    # data
    y                     :: Vector{T} # response 
    Xt                    :: Matrix{T} # X should include a column of 1's
    Zt                    :: Matrix{T} # Random effect covars
    Wt                    :: Matrix{T} # Covariates that affect WS variability
    # working arrays
    obj                   :: Vector{T} # instance objective value
    res                   :: Vector{T} # residual vector
    expwτinv              :: Vector{T} # hold exp.(W * τ)
    ztz                   :: Matrix{T} # hold Z'Z
    storage_q_n           :: Matrix{T}
    storage_q2_n          :: Matrix{T}
end

function MixWildObs(
    y::AbstractVector{T},
    X::AbstractMatrix{T},
    Z::AbstractMatrix{T},
    W::AbstractMatrix{T}
    ) where T <: BlasReal
    n, p, q, l  = size(X, 1), size(X, 2), size(Z, 2), size(W, 2)
    # working arrays
    obj                   = Vector{T}(undef, 1) 
    res                   = Vector{T}(undef, n)
    expwτinv              = Vector{T}(undef, n)
    ztz                   = transpose(Z) * Z
    storage_q_n           = Matrix{T}(undef, q, n)
    storage_q2_n          = Matrix{T}(undef, abs2(q), n)
    # constructor
    MixWildObs{T}(
        y, transpose(X), transpose(Z), transpose(W), 
        obj, res, expwτinv, ztz, 
        storage_q_n, storage_q2_n
        )
end

"""
    MixWildModel

Multiple location scale linear mixed model, which contains a vector of 
`MixWildObs` as data, model parameters, and working arrays.

    MixWildModel(obsvec; obswts, meannames, renames, wsvarnames)

# Positional arguments
- `obsvec`: Vector of WSVarLmmObs

# Keyword arguments
- `obswts`: Subject-level weight vector of observation weights, length of the `obsvec` object.
- `meannames`: Names of the mean fixed effects covariates
- `renames`: Names of the random location effects covariates
- `wsvarnames`: Names of the ws variance fixed effects covariates
"""
struct MixWildModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    data            :: Vector{MixWildObs{T}}
    respname        :: String
    meannames       :: Vector{String} # names of mean fixed effect variables
    renames         :: Vector{String} # names of random location effect variables
    wsvarnames      :: Vector{String} # names of ws var fixed effect variables
    meanformula     :: FormulaTerm
    reformula       :: FormulaTerm
    wsvarformula    :: FormulaTerm
    ids             :: Union{Vector{<:AbstractString}, Vector{<:Int}} # IDs of individuals/clusters in order
    obswts          :: Vector{T} # individual/cluster weights
    # dimenions
    p               :: Int       # number of mean parameters in linear regression
    q               :: Int       # number of random effects
    l               :: Int       # number of parameters for modeling WS variability
    m               :: Int       # number of individuals/clusters
    nis             :: Vector{Int}  # number of observations per cluster 
    nsum            :: Int       # number of observations (summed across individuals)
    # sufficient statistics
    xtx             :: Matrix{T} # sum_i Xi'Xi
    xty             :: Vector{T} # sum_i Xi'yi
    wtw             :: Matrix{T} # sum_i Wi'Wi
    ztz2            :: Matrix{T} # sum_i Zi'Zi ⊗ Zi'Zi
    ztz2od          :: Matrix{T} # sum_i (Zi'Zi ⊗ Zi'Zi - (Zi' ⊙ Zi')(Zi' ⊙ Zi')')
    # parameters
    β               :: Vector{T}  # p-vector of mean regression coefficients
    τ               :: Vector{T}  # l-vector of WS variability regression coefficients
    Lγω             :: Matrix{T}  # (q+1)x(q+1) lower triangular Cholesky factor of precision
    Σγω             :: Matrix{T}  # (q+1)x(q+1) covariance matrix of (γi, ωi)
    Λγω             :: Matrix{T}  # (q+1)x(q+1) covariance matrix of (γi, ωi)
    # working arrays
    λωωsc           :: Vector{T}
    Lγ⁻¹            :: Matrix{T}
    Lγ⁻¹ZtW⁻¹ZLγ⁻ᵀ  :: Matrix{T}
    storage_q       :: Vector{T}
    # Gauss-Hermite quadrature related
    gh_avec         :: Vector{T}
    gh_bvec         :: Vector{T}
    gh_c            :: Vector{T}
    gh_wts          :: Vector{T}
    gh_nodes        :: Vector{T}
    # model has been fit or not
    isfitted        :: Vector{Bool}
    # for variance estimator
    vcov            :: Matrix{T}
end

function MixWildModel(
    obsvec      :: Vector{MixWildObs{T}};
    obswts      :: Vector = [],
    respname    :: String = "y",
    meannames   :: Vector{String} = ["x$i" for i in 1:size(obsvec[1].Xt, 1)],
    renames     :: Vector{String} = ["z$i" for i in 1:size(obsvec[1].Zt, 1)],
    wsvarnames  :: Vector{String} = ["w$i" for i in 1:size(obsvec[1].Wt, 1)],
    meanformula :: FormulaTerm = FormulaTerm(term(Symbol(respname)), 
                        sum(term.(Symbol.(meannames)))),
    reformula   :: FormulaTerm = FormulaTerm(term(Symbol(respname)), 
                        sum(term.(Symbol.(renames)))),
    wsvarformula:: FormulaTerm = FormulaTerm(term(Symbol(respname)), 
                        sum(term.(Symbol.(wsvarnames)))),
    ids         :: Union{Vector{<:AbstractString}, Vector{Int}} = collect(1:length(obsvec)),
    ghpts       :: Integer = 10
    ) where T <: BlasReal
    # dimensions
    p            = size(obsvec[1].Xt, 1)
    q            = size(obsvec[1].Zt, 1)
    l            = size(obsvec[1].Wt, 1)
    m            = length(obsvec)
    nis          = map(o -> length(o.y), obsvec)
    nsum         = sum(nis)
    qp1◺         = ◺(q + 1)
    # sufficient statistics
    xtx          = zeros(T, p, p)
    xty          = zeros(T, p)
    wtw          = zeros(T, l, l)
    ztz2         = zeros(T, abs2(q), abs2(q))
    ztz2od       = zeros(T, abs2(q), abs2(q))
    for obs in obsvec
        # accumulate Xi'Xi
        BLAS.syrk!('U', 'N', T(1), obs.Xt, T(1), xtx)
        # accumulate Xi'yi
        BLAS.gemv!('N', T(1), obs.Xt, obs.y, T(1), xty)
        # accumulate Wi' * Wi
        BLAS.syrk!('U', 'N', T(1), obs.Wt, T(1), wtw)
        # accumulate Zi'Zi ⊗ Zi'Zi
        kron_axpy!(obs.ztz, obs.ztz, ztz2)
        # accumualte (Zi' ⊙ Zi')(Zi' ⊙ Zi')'
        # Ut_kr_Ut used as scratch space to store Zi' ⊙ Zi'
        kr_axpy!(obs.Zt, obs.Zt, fill!(obs.storage_q2_n, 0))
        BLAS.syrk!('U', 'N', T(1), obs.storage_q2_n, T(1), ztz2od)
    end
    ztz2od .= ztz2 .- ztz2od
    copytri!(   xtx, 'U')
    copytri!(   wtw, 'U')
    copytri!(  ztz2, 'U')
    copytri!(ztz2od, 'U')
    # parameters
    β        = Vector{T}(undef, p)
    τ        = Vector{T}(undef, l)
    Lγω      = Matrix{T}(undef, q + 1, q + 1) # cholesky of precision (optim. variables)
    Σγω      = Matrix{T}(undef, q + 1, q + 1) # covariance matrix
    Λγω      = Matrix{T}(undef, q + 1, q + 1) # precision matrix
    # working arrays
    λωωsc    = Vector{T}(undef, 1)            # Schur-complement: λωω - λγω' * Λγγ⁻¹ * λγω
    Lγ⁻¹     = Matrix{T}(undef, q, q)
    Lγ⁻¹ZtW⁻¹ZLγ⁻ᵀ = Matrix{T}(undef, q, q)
    storage_q = Vector{T}(undef, q)
    # Gauss-Hermite quadrature related
    gh_avec  = Vector{T}(undef, q)
    gh_bvec  = Vector{T}(undef, q)
    gh_c     = Vector{T}(undef, 1)
    gh_nodes, gh_wts = gausshermite(ghpts)
    # has been fit or not 
    isfitted = [false]
    # sandwich estimator
    vcov     = Matrix{T}(undef, p + qp1◺ + l, p + qp1◺ + l)
    # constructor
    MixWildModel{T}(
        obsvec, respname, meannames, renames, wsvarnames,
        meanformula, reformula, wsvarformula,
        ids, obswts, p, q, l, m, nis, nsum,
        xtx, xty, wtw, ztz2, ztz2od,
        β, τ, Lγω, Σγω, Λγω,
        λωωsc, Lγ⁻¹, Lγ⁻¹ZtW⁻¹ZLγ⁻ᵀ, storage_q,
        gh_avec, gh_bvec, gh_c, gh_wts, gh_nodes,
        isfitted, vcov)
end

function setparameters!(
    mwm  :: MixWildModel{T},
    β    :: AbstractVector{T},
    τ    :: AbstractVector{T},
    Lγω  :: AbstractMatrix{T} # lower cholesky factor of precision Λγω
    ) where T <: BlasReal
    q = mwm.q
    copy!(mwm.β, β)
    copy!(mwm.τ, τ)
    # lower cholesky factor of precision Λγω
    mwm.Lγω .= LowerTriangular(Lγω)
    # precision matrix
    mul!(mwm.Λγω, mwm.Lγω, transpose(mwm.Lγω))
    # inverse of lower cholesky of Λγγ
    @views copy!(mwm.Lγ⁻¹, mwm.Lγω[1:q, 1:q])
    LAPACK.trtri!('L', 'N', mwm.Lγ⁻¹)
    # covariance matrix = inverse of precision
    LAPACK.potrf!('U', copy!(mwm.Σγω, mwm.Λγω))
    LAPACK.potri!('U', mwm.Σγω)
    copytri!(mwm.Σγω, 'U')
    # (p+1, p+1) schur-complement of precision
    @views mwm.λωωsc[1] = mwm.Λγω[q + 1, q + 1] - 
        abs2(norm(mul!(mwm.storage_q, mwm.Lγ⁻¹, mwm.Λγω[1:q, q + 1])))
    mwm
end

"""
    loglikelihood!(i::Integer, m::MixWildModel, needgrad::Bool, needhess::Bool, updateres::Bool) 
    loglikelihood!(m::MixWildModel; needgrad::Bool, needhess::Bool, updateres::Bool)

Evaluate the loglikelihood of datum `i` or whole dataset using parameters 
`m.β`, `m.τ`, `m.Lγω`, ``. 
Gradient is calculated if `needgrad=true`. 
Expected Hessian is calculated if `needhess=true`.  
If `updateres=true`, update mean level residuals first.
"""
function loglikelihood!(
    obsidx    :: Integer,
    vlmm      :: MixWildModel{T},
    needgrad  :: Bool = false,
    needhess  :: Bool = false,
    updateres :: Bool = false
    ) where T <: BlasReal
    obs = vlmm.data[obsidx]
    q, l, n = vlmm.q, vlmm.l, size(obs.Zt, 2)
    λωω = vlmm.Λγω[q+1, q+1] # scalar
    ###########
    # objective
    ###########
    # update the residual vector ri = y_i - Xi β
    updateres && update_res!(obs, vlmm.β)
    # constant term
    obs.obj[1] = - (n + 1) * log(2π) + log(vlmm.λωωsc[1]) - log(λωω / 2)
    # expwτinv = W * τ for now
    mul!(obs.expwτinv, transpose(obs.Wt), vlmm.τ)
    obs.obj[1] = (obs.obj[1] - sum(obs.expwτinv)) / 2
    # expwτinv = exp(- W * τ)
    @inbounds for i in 1:n
        obs.expwτinv[i] = exp(-obs.expwτinv[i])
    end
    # assemble Lγ⁻¹ZtW⁻¹ZLγ⁻ᵀ
    rmul!(copy!(obs.storage_q_n, obs.Zt), Diagonal(obs.expwτinv)) # storage_q_n = ZtW⁻¹
    mul!(vlmm.storage_q, obs.storage_q_n, obs.res) # storage_q = ZtW⁻¹ * r, to be used by gh_avec
    mul!(vlmm.Lγ⁻¹ZtW⁻¹ZLγ⁻ᵀ, obs.storage_q_n, transpose(obs.Zt))
    BLAS.trmm!('L', 'L', 'N', 'N', T(1), vlmm.Lγ⁻¹, vlmm.Lγ⁻¹ZtW⁻¹ZLγ⁻ᵀ)
    BLAS.trmm!('R', 'L', 'T', 'N', T(1), vlmm.Lγ⁻¹, vlmm.Lγ⁻¹ZtW⁻¹ZLγ⁻ᵀ)
    # eigen-decompose Lγ⁻¹ZtW⁻¹ZLγ⁻ᵀ -> V * Diagonal(d) * V'
    # (d, U = Lγ⁻ᵀ V) is the generalized eigen-decomposiion of (ZtW⁻¹Z, Λγγ)
    d, V = eigen!(Symmetric(vlmm.Lγ⁻¹ZtW⁻¹ZLγ⁻ᵀ)) # this line allocates!
    # prepare constants for Gauss-Hermite quadrature
    BLAS.trmv!('L', 'N', 'N', vlmm.Lγ⁻¹, vlmm.storage_q)
    mul!(vlmm.gh_avec, transpose(V), vlmm.storage_q)
    mul!(vlmm.storage_q, vlmm.Lγ⁻¹, view(vlmm.Λγω, 1:q, q + 1))
    mul!(vlmm.gh_bvec, transpose(V), vlmm.storage_q)
    fill!(vlmm.gh_c, 0)
    @inbounds for i in 1:n
        vlmm.gh_c[1] += abs2(obs.res[i]) * obs.expwτinv[i]
    end
    # Gauss-Hermite quadrature
    gh_integral = zero(T)
    @inbounds for (i, ξ) in enumerate(vlmm.gh_nodes)
        ω = ξ / sqrt(λωω / 2)
        expωinv = exp(-ω)
        ep = n * ω + expωinv * vlmm.gh_c[1]
        for k in 1:q
            ep += log(expωinv * d[k] + 1) - 
                abs2(expωinv * vlmm.gh_avec[k] - ω * vlmm.gh_bvec[k]) / 
                (expωinv * d[k] + 1)
        end
        gh_integral += vlmm.gh_wts[i] * exp(- (1//2) * ep)
    end
    obs.obj[1] += log(gh_integral)
    ###########
    # gradient
    ###########
    if needgrad
        # TODO
    end
    ###########
    # hessian
    ###########
    if needhess
        # TODO
    end
    obs.obj[1]
end

"""
    loglikelihood!(m::MixWildModel, needgrad::Bool, needhess:Bool, updateres::Bool)

Calculate the objective function of a `WSVarLmmModel` object and optionally the 
gradient and hessian.
"""
function loglikelihood!(
    m         :: MixWildModel{T},
    needgrad  :: Bool = false,
    needhess  :: Bool = false,
    updateres :: Bool = false
    ) where T <: BlasReal
    # accumulate obj and gradient
    obj = zero(T)
    for i in 1:length(m.data)
        wtobs = isempty(m.obswts) ? one(T) : T(m.obswts[i])
        obj += wtobs * loglikelihood!(i, m, needgrad, needhess, updateres)
        if needgrad
            # TODO
        end
        if needhess
            # TODO
        end
    end
    obj
end

"""
    update_res!(obs::MixWildObs, β)

Update the residual vector of `obs::WSVarLmmObs` according to `β`.
"""
function update_res!(
    obs :: MixWildObs{T}, 
    β   :: Vector{T}
    ) where T <: BlasReal
    BLAS.gemv!('T', T(-1), obs.Xt, β, T(1), copyto!(obs.res, obs.y))
    obs.res
end

"""
    update_res!(m::MixWildModel)

Update residual vector of each observation in `m` according to `m.β`.
"""
function update_res!(m::MixWildModel{T}) where T <: BlasReal
    for obs in m.data
        update_res!(obs, m.β)
    end
    nothing
end

"""
    fit!(
        m::MixWildModel, 
        solver=IpoptSolver(print_level=0, mehrotra_algorithm="yes", max_iter=100)
    )

Fit a `MixWildModel` object using MLE.

# Positional arguments
- `m::MixWildModel`: Model to fit.
- `solver`: Nonlinear programming solver to use. Common choices include:  
    - `Ipopt.IpoptSolver(print_level=0, mehrotra_algorithm="yes", warm_start_init_point="yes", max_iter=100)`.
    - `Ipopt.IpoptSolver(print_level=0, watchdog_shortened_iter_trigger=3, max_iter=100)`.
    - `Ipopt.IpoptSolver(print_level=0, max_iter=100)`.
    - `KNITRO.KnitroSolver(outlev=3)`. (Knitro is commercial software)
    - `NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000)`.  
    - `NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxeval=4000)`.

# Keyword arguments
- `verbose::Bool`: Verbose display or not, Default is `true`.
"""
function fit!(
    mwm      :: MixWildModel,
    solver = NLopt.NLoptSolver(algorithm = :LN_BOBYQA, 
        ftol_rel = 1e-12, ftol_abs = 1e-8, maxeval = 10000);
    verbose  :: Bool = true)
    # set up NLP optimization problem
    npar = mwm.p + mwm.l + ◺(mwm.q + 1)
    optm = MathProgBase.NonlinearModel(solver)
    lb   = fill(-Inf, npar)
    ub   = fill( Inf, npar)
    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Max, mwm)
    par0 = Vector{Float64}(undef, npar)
    # optimize
    verbose && println("loglik at initial point = $(loglikelihood!(mwm, false, false, true))")
    tic = time() # start timing 
    modelpar_to_optimpar!(par0, mwm)
    MathProgBase.setwarmstart!(optm, par0)
    MathProgBase.optimize!(optm)
    optimpar_to_modelpar!(mwm, MathProgBase.getsolution(optm))
    toc = time()
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || 
        @warn("Optimization unsuccesful; got $optstat")
    verbose && @printf("status = %s, time(s) = %f\n", optstat, toc - tic)
    mwm.isfitted[1] = true
    # refresh objective, gradient, and Hessian
    verbose && println("loglik at solution = $(loglikelihood!(mwm, false, false, true))")
    # sandwich estimator
    # sandwich!(m)
    mwm
end

"""
    modelpar_to_optimpar!(par, m)

Translate model parameters in `m` to optimization variables in `par`.
"""
function modelpar_to_optimpar!(
    par :: Vector,
    mwm :: MixWildModel{T}
    ) where T <: BlasReal
    p, q, l = mwm.p, mwm.q, mwm.l
    # β
    copyto!(par, mwm.β)
    # τ
    copyto!(par, p + 1, mwm.τ, 1, l)
    # Lγω
    offset = p + l + 1
    @inbounds for j in 1:(q + 1), i in j:(q + 1)
        par[offset] = i == j ? log(max(mwm.Lγω[i, j], floatmin(T))) : mwm.Lγω[i, j]
        offset += 1
    end
    par
end

"""
    optimpar_to_modelpar!(m, par)

Translate optimization variables in `par` to the model parameters in `m`.
"""
function optimpar_to_modelpar!(
        mwm :: MixWildModel, 
        par :: Vector
    )
    p, q, l = mwm.p, mwm.q, mwm.l
    # Lγω
    offset = p + l + 1
    fill!(mwm.Lγω, 0)
    @inbounds for j in 1:(q + 1), i in j:(q + 1)
        mwm.Lγω[i, j] = i == j ? exp(par[offset]) : par[offset]
        offset += 1
    end
    @views setparameters!(mwm, par[1:p], par[(p+1):(p+l)], mwm.Lγω)
    mwm
end

function MathProgBase.initialize(
                       :: MixWildModel, 
    requested_features :: Vector{Symbol}
    )
    for feat in requested_features
        if !(feat in [:Grad])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(::MixWildModel) = [:Grad]

function MathProgBase.eval_f(
    mwm :: MixWildModel, 
    par :: Vector
    )
    optimpar_to_modelpar!(mwm, par)
    loglikelihood!(mwm, false, false, true)
end

function MathProgBase.eval_grad_f(
    mwm  :: MixWildModel, 
    grad :: Vector, 
    par  :: Vector
    )
    optimpar_to_modelpar!(mwm, par)
    logl = loglikelihood!(mwm, false, false, true)
    fill!(grad, 0)
    logl
end

MathProgBase.eval_g(::MixWildModel, g, par) = nothing
MathProgBase.jac_structure(::MixWildModel) = Int[], Int[]
MathProgBase.eval_jac_g(::MixWildModel, J, par) = nothing

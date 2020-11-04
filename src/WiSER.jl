module WiSER

using DataFrames, Tables, LinearAlgebra, MathProgBase
using Printf, Reexport, Statistics
import LinearAlgebra: BlasReal, copytri!
import DataFrames: DataFrame
@reexport using Ipopt
@reexport using NLopt
@reexport using StatsModels
@reexport using Distributions 

export 
    # types 
    WSVarLmmObs, 
    WSVarLmmModel,
    # functions
    coef,
    coefnames,
    coeftable,
    confint,
    DataFrame,
    fit!,
    init_ls!,
    init_mom!,
    nlsv_obj!,
    nclusters,
    nobs,
    rand!,
    respdists,
    rvarlmm, 
    rvarlmm!,
    stderror,
    update_res!,
    update_wtmat!,
    vcov

"""
    WSVarLmmObs
    WSVarLmmObs(y, X, Z, W)

A realization of within-subject variance linear mixed model data instance.

# Positional Arguments:
    `y`: the response vector
    `X`: the mean fixed effects covariate matrix 
    `Z`: the random location effects covariate matrix
    `W`: the within-subject variance fixed effects covariate matrix

"""
struct WSVarLmmObs{T <: BlasReal}
    # data
    y                     :: Vector{T} # response 
    Xt                    :: Matrix{T} # X should include a column of 1's
    Zt                    :: Matrix{T} # Random effect covars
    Wt                    :: Matrix{T} # Covariates that affect WS variability
    # working arrays
    obj                   :: Vector{T} # instance objective value
    ∇β                    :: Vector{T} # gradient
    ∇τ                    :: Vector{T}
    ∇Lγ                   :: Matrix{T}
    Hββ                   :: Matrix{T} # hessian
    Hττ                   :: Matrix{T} 
    HτLγ                  :: Matrix{T}
    HLγLγ                 :: Matrix{T}
    res                   :: Vector{T} # residual vector
    res2                  :: Vector{T} # residual vector.^2
    resnrm2               :: Vector{T} # sum of residual squares
    expwτ                 :: Vector{T} # hold exp.(W * τ)
    ztz                   :: Matrix{T} # Z'Z
    ztres                 :: Vector{T} # Z'res
    zlltzt_dg             :: Vector{T}
    storage_n1            :: Vector{T}
    storage_p1            :: Vector{T}
    storage_q1            :: Vector{T}
    storage_pn            :: Matrix{T}
    storage_qn            :: Matrix{T}
    storage_ln            :: Matrix{T}
    storage_pp            :: Matrix{T}    
    storage_qq            :: Matrix{T}
    storage_qp            :: Matrix{T}
    storage_q◺n           :: Matrix{T}
    # Woodbury structure for weight matrix Vinv = Dinv - U * U'
    Dinv                  :: Vector{T}
    Ut                    :: Matrix{T}
    # for weighted objective eval
    rt_Dinv_r             :: Vector{T}
    rt_UUt_r              :: Vector{T}
    rt_U                  :: Matrix{T}
    Dinv_r                :: Vector{T}
    rt_UUt                :: Matrix{T}
    Zt_Dinv_r             :: Vector{T}
    rt_UUt_Z              :: Matrix{T}
    diagUUt_Dinv          :: Vector{T}
    Dinv_Z_L              :: Matrix{T}
    UUt_Z_L               :: Matrix{T}
    Ut_D_U                :: Matrix{T}
    Zt_Dinv_Z             :: Matrix{T}
    Lt_Zt_Dinv_Z_L        :: Matrix{T}
    Zt_UUt_Z              :: Matrix{T}
    Lt_Zt_UUt_Z_L         :: Matrix{T}
    # for gradient wrt τ
    diagDVRV              :: Vector{T}
    # for gradient wrt Lγ
    Zt_Dinv               :: Matrix{T}
    Zt_UUt_rrt_Dinv_Z :: Matrix{T}
    Zt_UUt_rrt_UUt_Z      :: Matrix{T}
    Zt_UUt                :: Matrix{T}
    Lt_Zt_Dinv_r          :: Vector{T}
    Zt_Vinv_r             :: Vector{T}
    # for Hessian wrt τ
    Wt_D_Dinv             :: Matrix{T}
    sqrtDinv_UUt          :: Vector{T}
    Ut_kr_Ut              :: Matrix{T}
    Wt_D_Ut_kr_Utt        :: Matrix{T}
    Wt_D_sqrtdiagDinv_UUt :: Matrix{T}
    # for Hessian wrt Lγ
    Zt_Vinv_Z             :: Matrix{T}
    Zt_Vinv               :: Matrix{T}
end

function WSVarLmmObs(
    y::AbstractVector{T},
    X::AbstractMatrix{T},
    Z::AbstractMatrix{T},
    W::AbstractMatrix{T}
    ) where T <: BlasReal
    n, p, q, l  = size(X, 1), size(X, 2), size(Z, 2), size(W, 2)
    q◺ = ◺(q)
    # working arrays
    obj                   = Vector{T}(undef, 1) 
    ∇β                    = Vector{T}(undef, p)
    ∇τ                    = Vector{T}(undef, l)
    ∇Lγ                   = Matrix{T}(undef, q, q)
    Hββ                   = Matrix{T}(undef, p, p)
    Hττ                   = Matrix{T}(undef, l, l)
    HτLγ                  = Matrix{T}(undef, l, q◺)
    HLγLγ                 = Matrix{T}(undef, q◺, q◺)
    res                   = Vector{T}(undef, n)
    res2                  = Vector{T}(undef, n)
    resnrm2               = Vector{T}(undef, 1)
    expwτ                 = Vector{T}(undef, n)
    ztz                   = transpose(Z) * Z
    ztres                 = Vector{T}(undef, q)
    zlltzt_dg             = Vector{T}(undef, n)
    storage_n1            = Vector{T}(undef, n)
    storage_p1            = Vector{T}(undef, p)
    storage_q1            = Vector{T}(undef, q)
    storage_pn            = Matrix{T}(undef, p, n)
    storage_qn            = Matrix{T}(undef, q, n)
    storage_ln            = Matrix{T}(undef, l, n)
    storage_pp            = Matrix{T}(undef, p, p)
    storage_qq            = Matrix{T}(undef, q, q)
    storage_qp            = Matrix{T}(undef, q, p)
    storage_q◺n           = Matrix{T}(undef, q◺, n)
    # added for weighted estimating equations
    Dinv                  = Vector{T}(undef, n) #stores diag(exp(-wτ_0))
    Ut                    = Matrix{T}(undef, q, n)
    # denote, r as residual vector D as diagonal(exp(wτ))
    # added for weigthed estimating equations
    rt_Dinv_r             = Vector{T}(undef, 1)
    rt_UUt_r              = Vector{T}(undef, 1)
    rt_U                  = Matrix{T}(undef, 1, q) 
    Dinv_r                = Vector{T}(undef, n)
    rt_UUt                = Matrix{T}(undef, 1, n) 
    Zt_Dinv_r             = Vector{T}(undef, q)
    rt_UUt_Z              = Matrix{T}(undef, 1, q) 
    diagUUt_Dinv          = Vector{T}(undef, n)
    Dinv_Z_L              = Matrix{T}(undef, n, q)
    UUt_Z_L               = Matrix{T}(undef, n, q)
    Ut_D_U                = Matrix{T}(undef, q, q)
    Zt_Dinv_Z             = Matrix{T}(undef, q, q)
    Lt_Zt_Dinv_Z_L        = Matrix{T}(undef, q, q)
    Zt_UUt_Z              = Matrix{T}(undef, q, q)
    Lt_Zt_UUt_Z_L         = Matrix{T}(undef, q, q)
    # for gradient wrt τ
    diagDVRV              = Vector{T}(undef, n)
    # for gradient wrt Lγ
    Zt_Dinv               = Matrix{T}(undef, q, n) 
    Zt_UUt_rrt_Dinv_Z     = Matrix{T}(undef, q, q)
    Zt_UUt_rrt_UUt_Z      = Matrix{T}(undef, q, q)  
    Zt_UUt                = Matrix{T}(undef, q, n) 
    Lt_Zt_Dinv_r          = Vector{T}(undef, q)
    Zt_Vinv_r             = Vector{T}(undef, q)
    # for Hessian wrt τ
    Wt_D_Dinv             = Matrix{T}(undef, l, n)
    sqrtDinv_UUt          = Vector{T}(undef, n)
    Ut_kr_Ut              = Matrix{T}(undef, abs2(q), n)
    Wt_D_Ut_kr_Utt        = Matrix{T}(undef, l, abs2(q))
    Wt_D_sqrtdiagDinv_UUt = Matrix{T}(undef, l, n)
    # for Hessian wrt Lγ
    Zt_Vinv_Z             = Matrix{T}(undef, q, q)
    Zt_Vinv               = Matrix{T}(undef, q, n)
    # constructor
    WSVarLmmObs{T}(
        y, transpose(X), transpose(Z), transpose(W), obj,
        ∇β, ∇τ, ∇Lγ,
        Hββ, Hττ, HτLγ, HLγLγ,
        res, res2, resnrm2, expwτ, ztz, ztres, zlltzt_dg,
        storage_n1, storage_p1, storage_q1,
        storage_pn, storage_qn, storage_ln, 
        storage_pp, storage_qq, storage_qp, storage_q◺n,
        Dinv, Ut, rt_Dinv_r, rt_UUt_r, rt_U, Dinv_r,
        rt_UUt, Zt_Dinv_r, rt_UUt_Z, diagUUt_Dinv, 
        Dinv_Z_L, UUt_Z_L, Ut_D_U,  Zt_Dinv_Z, 
        Lt_Zt_Dinv_Z_L, Zt_UUt_Z, Lt_Zt_UUt_Z_L, 
        diagDVRV,
        Zt_Dinv, Zt_UUt_rrt_Dinv_Z, Zt_UUt_rrt_UUt_Z, 
        Zt_UUt, Lt_Zt_Dinv_r, Zt_Vinv_r, Wt_D_Dinv, 
        sqrtDinv_UUt, Ut_kr_Ut, Wt_D_Ut_kr_Utt, 
        Wt_D_sqrtdiagDinv_UUt, Zt_Vinv_Z, Zt_Vinv)
end

"""
    WSVarLmmModel

Within-subject variance linear mixed model, which contains a vector of 
`WSVarLmmObs` as data, model parameters, and working arrays.

    WSVarLmmModel(obsvec; obswts, meannames, renames, wsvarnames)
    
# Positional arguments  
- `obsvec`: Vector of WSVarLmmObs

# Keyword arguments  
- `obswts`: Subject-level weight vector of observation weights, length of the `obsvec` object.
- `meannames`: Names of the mean fixed effects covariates
- `renames`: Names of the random location effects covariates
- `wsvarnames`: Names of the ws variance fixed effects covariates

"""
struct WSVarLmmModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    data            :: Vector{WSVarLmmObs{T}}
    respname        :: String
    meannames       :: Vector{String} # names of mean fixed effect variables
    renames         :: Vector{String} # names of random location effect variables
    wsvarnames      :: Vector{String} # names of ws var fixed effect variables
    meanformula     :: FormulaTerm
    reformula       :: FormulaTerm
    wsvarformula    ::FormulaTerm
    ids             :: Union{Vector{AbstractString}, Vector{Int}} # IDs of individuals/clusters in order
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
    Lγ              :: Matrix{T}  # q by q lower triangular Cholesky factor of cov(γ)
    Σγ              :: Matrix{T}  # q by q covariance matrix of γ
    # working arrays
    ∇β              :: Vector{T}
    ∇τ              :: Vector{T}
    ∇Lγ             :: Matrix{T}
    ∇Σγ             :: Vector{T}
    Hββ             :: Matrix{T}
    Hττ             :: Matrix{T}
    HτLγ            :: Matrix{T}
    HLγLγ           :: Matrix{T}
    HΣγΣγ           :: Matrix{T}
    # weighted NLS or unweighted NLS
    iswtnls         :: Vector{Bool}
    # multi-threading or not
    ismthrd         :: Vector{Bool}
    # model has been fit or not
    isfitted        :: Vector{Bool}
    # for sandwich estimator
    ψ               :: Vector{T}
    Ainv            :: Matrix{T}
    B               :: Matrix{T}
    vcov            :: Matrix{T}
end

function WSVarLmmModel(
    obsvec      :: Vector{WSVarLmmObs{T}};
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
    ids         :: Union{Vector{AbstractString}, Vector{Int}} = collect(1:length(obsvec))
    ) where T <: BlasReal
    # dimensions
    p            = size(obsvec[1].Xt, 1)
    q            = size(obsvec[1].Zt, 1)
    l            = size(obsvec[1].Wt, 1)
    m            = length(obsvec)
    nis          = map(o -> length(o.y), obsvec)
    nsum         = sum(nis)
    q◺           = ◺(q)
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
        kr_axpy!(obs.Zt, obs.Zt, fill!(obs.Ut_kr_Ut, 0))
        BLAS.syrk!('U', 'N', T(1), obs.Ut_kr_Ut, T(1), ztz2od)
    end
    ztz2od .= ztz2 .- ztz2od
    copytri!(   xtx, 'U')
    copytri!(   wtw, 'U')
    copytri!(  ztz2, 'U')
    copytri!(ztz2od, 'U')
    # parameters
    β        = Vector{T}(undef, p)
    τ        = Vector{T}(undef, l)
    Lγ       = Matrix{T}(undef, q, q)
    Σγ       = Matrix{T}(undef, q, q)
    # gradients
    ∇β       = Vector{T}(undef, p)
    ∇τ       = Vector{T}(undef, l)
    ∇Lγ      = Matrix{T}(undef, q, q)
    ∇Σγ      = Vector{T}(undef, abs2(q))
    Hββ      = Matrix{T}(undef, p, p)
    Hττ      = Matrix{T}(undef, l, l)
    HτLγ     = Matrix{T}(undef, l, q◺)
    HLγLγ    = Matrix{T}(undef, q◺, q◺)
    HΣγΣγ    = Matrix{T}(undef, abs2(q), abs2(q))
    # weighted NLS fitting or not
    iswtnls = [false]
    # multi-threading or not
    ismthrd  = [false]
    # has been fit or not 
    isfitted = [false]
    # sandwich estimator
    ψ        = Vector{T}(undef, p + q◺ + l)
    Ainv     = Matrix{T}(undef, p + q◺ + l, p + q◺ + l)
    B        = Matrix{T}(undef, p + q◺ + l, p + q◺ + l)
    vcov     = Matrix{T}(undef, p + q◺ + l, p + q◺ + l)
    # constructor
    WSVarLmmModel{T}(
        obsvec, respname, meannames, renames, wsvarnames,
        meanformula, reformula, wsvarformula,
        ids, obswts, p, q, l, m, nis, nsum,
        xtx, xty, wtw, ztz2, ztz2od,
        β,  τ,  Lγ, Σγ,
        ∇β, ∇τ, ∇Lγ, ∇Σγ,
        Hββ, Hττ, HτLγ, HLγLγ, HΣγΣγ,
        iswtnls, ismthrd, isfitted, 
        ψ, Ainv, B, vcov)
end


coefnames(m::WSVarLmmModel) = [m.meannames; m.wsvarnames]
coef(m::WSVarLmmModel) = [m.β; m.τ]
nobs(m::WSVarLmmModel) = m.nsum
nclusters(m::WSVarLmmModel) = m.m
stderror(m::WSVarLmmModel) = [sqrt(m.vcov[i, i]) for i in 1:(m.p + m.l)]
vcov(m::WSVarLmmModel) = m.vcov # include variance parts of Lγ? 

confint(m::WSVarLmmModel, level::Real) = hcat(coef(m), coef(m)) +
    stderror(m) * quantile(Normal(), (1. - level) / 2.) * [1. -1.]
confint(m::WSVarLmmModel) = confint(m, 0.95)

function getformula(m::WSVarLmmModel, s::String)
    names = s == "mean" ? m.meannames : s == "var" ?
        m.wsvarnames : s == "re" ? m.renames : nothing
    lhs = join(map(x -> join(x[2:end], ": "),  (split.(names, ": "))), " + ")
    return m.respname * " ~ " * lhs
end

function coeftable(m::WSVarLmmModel)
    mstder = stderror(m)
    mcoefs = coef(m)
    wald = mcoefs ./ mstder
    pvals = 2 * Distributions.ccdf.(Normal(), abs.(wald))
    StatsModels.CoefTable(hcat(mcoefs, mstder, wald, pvals),
        ["Estimate", "Std. Error", "Z", "Pr(>|Z|)"],
        coefnames(m), 4, 3)
end

function Base.show(io::IO, m::WSVarLmmModel)
    if !m.isfitted[1]
        @warn("The model has not been fit.")
        return nothing
    end
    println(io)
    println(io, "Within-subject variance estimation by robust regression (WiSER)")
    println(io)
    println(io, "Mean Formula:")
    println(io, getformula(m, "mean"))
    println(io, "Random Effects Formula:")
    println(io, getformula(m, "re"))
    println(io, "Within-Subject Variance Formula:")
    println(io, getformula(m, "var"))
    println(io)
    println(io, "Number of individuals/clusters: $(m.m)")
    println(io, "Total observations: $(m.nsum)")
    println(io)
    println(io, "Fixed-effects parameters:")
    show(io, coeftable(m))
    println(io)
    println(io, "Random effects covariance matrix Σγ:")
    Base.print_matrix(IOContext(io, :compact => true), [m.renames m.Σγ])
    println(io)
    println(io)
end

include("nls.jl")
include("initialization.jl")
include("fit.jl")
include("df.jl")
include("rand.jl")
include("multivariate_calculus.jl")
include("sandwich.jl")

end

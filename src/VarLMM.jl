module VarLMM

using DataFrames, Distributions, LinearAlgebra, MathProgBase
using Permutations, Reexport, Statistics, StatsModels
using LoopVectorization, JuliaDB
import LinearAlgebra: BlasReal, copytri!
import DataFrames: DataFrame
@reexport using Ipopt
@reexport using NLopt
# @reexport using KNITRO
@reexport using StatsModels
@reexport using Distributions 

export 
    #types 
    VarLmmObs, 
    VarLmmModel,
    #functions
    coef,
    coefnames,
    coeftable,
    confint,
    DataFrame,
    fit!,
    init_ls!,
    init_wls!,
    mom_obj!,
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
    VarLmmObs
    VarLmmObs(y, X, Z, W)

A realization of variance linear mixed model data instance.
"""
struct VarLmmObs{T <: BlasReal}
    # data
    y                       :: Vector{T} # response 
    Xt                      :: Matrix{T} # X should include a column of 1's
    Zt                      :: Matrix{T} # Random effect covars
    Wt                      :: Matrix{T} # Covariates that affect WS variability
    # working arrays
    ∇β                      :: Vector{T} # gradient wrt β
    ∇τ                      :: Vector{T} # gradient wrt τ
    ∇Lγ                     :: Matrix{T} # gradient wrt L cholesky factor 
    Hττ                     :: Matrix{T} # hessian
    HτLγ                    :: Matrix{T}
    HLγLγ                   :: Matrix{T}
    res                     :: Vector{T} # residual vector
    res2                    :: Vector{T} # residual vector.^2
    resnrm2                 :: Vector{T} # sum of residual squares
    expwτ                   :: Vector{T} # hold exp.(W * τ)
    ztz                     :: Matrix{T} # Z'Z
    ztres                   :: Vector{T} # Z'res
    zlltzt_dg               :: Vector{T}
    storage_n1              :: Vector{T}
    storage_p1              :: Vector{T}
    storage_q1              :: Vector{T}
    storage_pn              :: Matrix{T}
    storage_qn              :: Matrix{T}
    storage_ln              :: Matrix{T}
    storage_pp              :: Matrix{T}    
    storage_qq              :: Matrix{T}
    storage_qp              :: Matrix{T}
    storage_q◺n             :: Matrix{T}

    #Woodbury structure for weight matrix Vinv = Dinv - U * U'
    Dinv                    :: Vector{T}
    Ut                      :: Matrix{T}

    #for weighted objective eval
    rt_Dinv_r               :: Vector{T}
    rt_UUt_r                :: Vector{T}
    rt_U                    :: Matrix{T}
    Dinv_r                  :: Vector{T}
    rt_UUt                  :: Matrix{T}
    Zt_Dinv_r               :: Vector{T}
    rt_UUt_Z                :: Matrix{T}
    diagUUt_Dinv            :: Vector{T}
    Dinv_Z_L                :: Matrix{T}
    UUt_Z_L                 :: Matrix{T}
    Ut_D_U                  :: Matrix{T}
    Zt_Dinv_Z               :: Matrix{T}
    Lt_Zt_Dinv_Z_L          :: Matrix{T}
    Zt_UUt_Z                :: Matrix{T}
    Lt_Zt_UUt_Z_L           :: Matrix{T}

    #for gradient wrt τ
    diagDVRV                :: Vector{T}

    #for gradient wrt Lγ
    Zt_Dinv                 :: Matrix{T}
    Zt_UUt_rrt_Dinv_Z       :: Matrix{T}
    Zt_UUt_rrt_UUt_Z        :: Matrix{T}
    Zt_UUt                  :: Matrix{T}
    Lt_Zt_Dinv_r            :: Vector{T}
    Zt_Vinv_r               :: Vector{T}

    #for Hessian wrt τ
    Wt_D_Dinv               :: Matrix{T}
    sqrtDinv_UUt            :: Vector{T}
    Ut_kr_Ut                :: Matrix{T}
    Wt_D_Ut_kr_Utt          :: Matrix{T}
    Wt_D_sqrtdiagDinv_UUt   :: Matrix{T}

    #for Hessian wrt Lγ
    Zt_Vinv_Z               :: Matrix{T}
    Zt_Vinv                 :: Matrix{T}

    XtVinvres               :: Vector{T}
    obj                     ::Vector{T}

end

function VarLmmObs(
    y::AbstractVector{T},
    X::AbstractMatrix{T},
    Z::AbstractMatrix{T},
    W::AbstractMatrix{T}
    ) where T <: BlasReal
    n, p, q, l  = size(X, 1), size(X, 2), size(Z, 2), size(W, 2)
    q◺ = ◺(q)
    # working arrays
    ∇β                      = Vector{T}(undef, p)
    ∇τ                      = Vector{T}(undef, l)
    ∇Lγ                     = Matrix{T}(undef, q, q)
    Hττ                     = Matrix{T}(undef, l, l)
    HτLγ                    = Matrix{T}(undef, l, q◺)
    HLγLγ                   = Matrix{T}(undef, q◺, q◺)
    res                     = Vector{T}(undef, n)
    res2                    = Vector{T}(undef, n)
    resnrm2                 = Vector{T}(undef, n)
    expwτ                   = Vector{T}(undef, n)
    ztz                     = Z'Z
    ztres                   = Vector{T}(undef, q)
    zlltzt_dg               = Vector{T}(undef, n)
    storage_n1              = Vector{T}(undef, n)
    storage_p1              = Vector{T}(undef, p)
    storage_q1              = Vector{T}(undef, q)
    storage_pn              = Matrix{T}(undef, p, n)
    storage_qn              = Matrix{T}(undef, q, n)
    storage_ln              = Matrix{T}(undef, l, n)
    storage_pp              = Matrix{T}(undef, p, p)
    storage_qq              = Matrix{T}(undef, q, q)
    storage_qp              = Matrix{T}(undef, q, p)
    storage_q◺n             = Matrix{T}(undef, q◺, n)

    #added for weighted estimating equations
    Dinv                    = Vector{T}(undef, n) #stores diag(exp(-wτ_0))
    Ut                      = Matrix{T}(undef, q, n)

    #denote, r as residual vector D as diagonal(exp(wτ))
    #added for weigthed estimating equations
    rt_Dinv_r               = Vector{T}(undef, 1)
    rt_UUt_r                = Vector{T}(undef, 1)
    rt_U                    = Matrix{T}(undef, 1, q) 
    Dinv_r                  = Vector{T}(undef, n)
    rt_UUt                  = Matrix{T}(undef, 1, n) 
    Zt_Dinv_r               = Vector{T}(undef, q)
    rt_UUt_Z                = Matrix{T}(undef, 1, q) 
    diagUUt_Dinv            = Vector{T}(undef, n)
    Dinv_Z_L                = Matrix{T}(undef, n, q)
    UUt_Z_L                 = Matrix{T}(undef, n, q)
    Ut_D_U                  = Matrix{T}(undef, q, q)
    Zt_Dinv_Z               = Matrix{T}(undef, q, q)
    Lt_Zt_Dinv_Z_L          = Matrix{T}(undef, q, q)
    Zt_UUt_Z                = Matrix{T}(undef, q, q)
    Lt_Zt_UUt_Z_L           = Matrix{T}(undef, q, q)

    #for gradient wrt τ
    diagDVRV                = Vector{T}(undef, n)

    #for gradient wrt Lγ
    Zt_Dinv                 = Matrix{T}(undef, q, n) 
    Zt_UUt_rrt_Dinv_Z       = Matrix{T}(undef, q, q)
    Zt_UUt_rrt_UUt_Z        = Matrix{T}(undef, q, q)  
    Zt_UUt                  = Matrix{T}(undef, q, n) 
    Lt_Zt_Dinv_r            = Vector{T}(undef, q)
    Zt_Vinv_r               = Vector{T}(undef, q)

    #for Hessian wrt τ
    Wt_D_Dinv               = Matrix{T}(undef, l, n)
    sqrtDinv_UUt            = Vector{T}(undef, n)
    Ut_kr_Ut                = Matrix{T}(undef, abs2(q), n)
    Wt_D_Ut_kr_Utt          = Matrix{T}(undef, l, abs2(q))
    Wt_D_sqrtdiagDinv_UUt   = Matrix{T}(undef, l, n)

    #for Hessian wrt Lγ
    Zt_Vinv_Z               = Matrix{T}(undef, q, q)
    Zt_Vinv                 = Matrix{T}(undef, q, n)

    XtVinvres               = Vector{T}(undef, p)
    obj                     = Vector{T}(undef, 1) 

    # constructor
    VarLmmObs{T}(
        y, transpose(X), transpose(Z), transpose(W), 
        ∇β, ∇τ, ∇Lγ,
        Hττ, HτLγ, HLγLγ,
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
        Wt_D_sqrtdiagDinv_UUt, Zt_Vinv_Z, Zt_Vinv, XtVinvres,
        obj)
end

"""
    VarLmmModel

Variance linear mixed model, which contains a vector of 
`VarLmmObs` as data, model parameters, and working arrays.

TODO: function documentation
"""
struct VarLmmModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    data    :: Vector{VarLmmObs{T}}
    meannames :: Vector{String} # names of mean fixed effect variables
    renames :: Vector{String} # names of random location effect variables
    wsvarnames :: Vector{String} # names of ws var fixed effect variables
    wts     :: Vector{T}
    p       :: Int       # number of mean parameters in linear regression
    q       :: Int       # number of random effects
    l       :: Int       # number of parameters for modeling WS variability
    m       :: Int       # number of individuals/clusters
    obs     :: Int       # number of observations (summed across individuals)
    # parameters
    β       :: Vector{T}  # p-vector of mean regression coefficients
    τ       :: Vector{T}  # l-vector of WS variability regression coefficients
    Lγ      :: Matrix{T}  # q by q lower triangular cholesky factor of random effects covmatrix of γ
    Σγ      :: Matrix{T}  # q by q Covariance matrix of γ
    # working arrays
    ∇β      :: Vector{T}
    ∇τ      :: Vector{T}
    ∇Lγ     :: Matrix{T}
    Hττ     :: Matrix{T}
    HτLγ    :: Matrix{T}
    HLγLγ   :: Matrix{T}
    # weighted model or not
    weighted:: Vector{Bool}
    # Add for variance
    Ainv    :: Matrix{T}
    B       :: Matrix{T}
    AinvB   :: Matrix{T}
    V       :: Matrix{T}
    XtVinvX :: Matrix{T}
end

function VarLmmModel(obsvec::Vector{VarLmmObs{T}};
    meannames = ["β$i" for i in 1:size(obsvec[1].Xt, 1)],
    renames = ["γ$i" for i in 1:size(obsvec[1].Zt, 1)],
    wsvarnames = ["τ$i" for i in 1:size(obsvec[1].Wt, 1)],
    wts = []
    ) where T <: BlasReal
    # dimensions
    p     = size(obsvec[1].Xt, 1)
    q, l  = size(obsvec[1].Zt, 1), size(obsvec[1].Wt, 1)
    m = length(obsvec)
    obs = sum(length(obsvec[i].y) for i in 1:m)
    q◺    = ◺(q)
    # parameters
    β     = Vector{T}(undef, p)
    τ     = Vector{T}(undef, l)
    Lγ    = Matrix{T}(undef, q, q)
    Σγ    = Matrix{T}(undef, q, q)
    # gradients
    ∇β    = Vector{T}(undef, p)
    ∇τ    = Vector{T}(undef, l)
    ∇Lγ   = Matrix{T}(undef, q, q)
    Hττ   = Matrix{T}(undef, l, l)
    HτLγ  = Matrix{T}(undef, l, q◺)
    HLγLγ = Matrix{T}(undef, q◺, q◺)
    # weighted fitting or not

    # Add for variance   
    Ainv    = Matrix{T}(undef, p + q◺ + l, p + q◺ + l)
    B       = Matrix{T}(undef, p + q◺ + l, p + q◺ + l)
    AinvB   = Matrix{T}(undef, p + q◺ + l, p + q◺ + l)
    V       = Matrix{T}(undef, p + q◺ + l, p + q◺ + l)
    XtVinvX = Matrix{T}(undef, p, p)

    # constructor
    VarLmmModel{T}(
        obsvec, meannames, renames, wsvarnames, wts,
        p, q, l, m, obs,
        β,  τ,  Lγ, Σγ,
        ∇β, ∇τ, ∇Lγ,
        Hττ, HτLγ, HLγLγ, [false],
        Ainv, B, AinvB, V, XtVinvX)
end

coefnames(m::VarLmmModel) = [m.meannames; m.wsvarnames]
coef(m::VarLmmModel) = [m.β; m.τ]
nobs(m::VarLmmModel) = m.obs
nclusters(m::VarLmmModel) = m.m
stderror(m::VarLmmModel) = sqrt.(diag(m.V)[1:(m.p + m.l)] ./ m.m)
vcov(m::VarLmmModel) = m.V # include variance parts of Lγ? 
# weights(m::VarLmmModel) = m.wts

confint(m::VarLmmModel, level::Real) = hcat(coef(m), coef(m)) +
    stderror(m) * quantile(Normal(), (1. - level) / 2.) * [1. -1.]
confint(m::VarLmmModel) = confint(m, 0.95)

function coeftable(m::VarLmmModel)
    mstder = stderror(m)
    mcoefs = coef(m)
    wald = mcoefs ./ mstder
    pvals = 2 * Distributions.ccdf.(Normal(), abs.(wald))
    StatsModels.CoefTable(hcat(mcoefs, mstder, wald, pvals),
        ["Estimate", "Std. Error", "Z", "Pr(>|Z|)"],
        coefnames(m), 4, 3)
end

function Base.show(io::IO, m::VarLmmModel)
    #p, q, l = m.p, m.q, m.l
    println(io, "Variance linear mixed model fit by method of moments")
    #println(io, " ", m.formula)

    println(io, "Number of individuals/clusters: $(m.m) \nTotal observations: $(m.obs)")
    println(io, " ")
    println(io, "Fixed-effects parameters:")
    show(io, coeftable(m))
    println(io, " ")

    println(io, "Σγ : Random Effects Covariance Matrix")
    Base.print_matrix(IOContext(io, :compact => true), [m.renames m.Σγ])
end

include("mom.jl")
# include("mom_avx.jl")
# include("mom_nlp.jl")
include("nlp_unconstr.jl")
include("df.jl")
include("varlmm_rand.jl")
include("multivariate_calculus.jl")

end
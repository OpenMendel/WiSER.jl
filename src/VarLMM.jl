module VarLMM

using DataFrames, Distributions, LinearAlgebra, MathProgBase
using Permutations, Reexport, Statistics, StatsModels, WoodburyMatrices
import LinearAlgebra: BlasReal, copytri!
import DataFrames: DataFrame
@reexport using Ipopt
@reexport using NLopt

export VarLmmObs, VarLmmModel
export DataFrame, fit!, init_ls!, init_wls!, mom_obj!, update_res!, update_wtmat!

"""
    VarLmmObs
    VarLmmObs(y, X, Z, W)

A realization of variance linear mixed model data instance.
"""
struct VarLmmObs{T <: BlasReal}
    # data
    y           :: Vector{T} # response 
    Xt          :: Matrix{T} # X should include a column of 1's
    Zt          :: Matrix{T} # Random effect covars
    Wt          :: Matrix{T} # Covariates that affect WS variability
    # working arrays
    ∇β          :: Vector{T} # gradient wrt β
    ∇τ          :: Vector{T} # gradient wrt τ
    ∇Lγ         :: Matrix{T} # gradient wrt L cholesky factor 
    Hττ         :: Matrix{T} # hessian
    HτLγ        :: Matrix{T}
    HLγLγ       :: Matrix{T}
    res         :: Vector{T} # residual vector
    res2        :: Vector{T} # residual vector.^2
    resnrm2     :: Vector{T} # sum of residual squares
    expwτ       :: Vector{T} # hold exp.(W * τ)
    ztz         :: Matrix{T} # Z'Z
    ztres       :: Vector{T} # Z'res
    zlltzt_dg   :: Vector{T}
    storage_n1  :: Vector{T}
    storage_p1  :: Vector{T}
    storage_q1  :: Vector{T}
    storage_pn  :: Matrix{T}
    storage_qn  :: Matrix{T}
    storage_ln  :: Matrix{T}
    storage_pp  :: Matrix{T}    
    storage_qq  :: Matrix{T}
    storage_qp  :: Matrix{T}
    storage_q◺n :: Matrix{T}

    #for weighted formulation
    Dinv        :: Vector{T}
    Ut          :: Matrix{T}

    #added for weigthed estimating equation obj eval
    rt_Dinv_r   :: Vector{T}
    rt_UUt_r    :: Vector{T}
    rt_U        :: Matrix{T}
    Dinv_r      :: Vector{T}
    rt_UUt      :: Matrix{T}
    Zt_Dinv_r   :: Vector{T}
    rt_UUt_Z    :: Matrix{T}
    diagUUt_Dinv :: Vector{T}
    Dinv_Z      :: Matrix{T}
    Dinv_Z_L    :: Matrix{T}
    UUt_Z       :: Matrix{T}
    UUt_Z_L     :: Matrix{T}
    Ut_D_U   :: Matrix{T}
    Zt_Dinv_Z   :: Matrix{T}
    Lt_Zt_Dinv_Z_L :: Matrix{T}
    Zt_UUt_Z    :: Matrix{T}
    Lt_Zt_UUt_Z_L :: Matrix{T}

    diagDVRV     :: Vector{T}

    #extra from gradient of vechL
    Zt_Dinv_rrt_Dinv_Z :: Matrix{T}
    Zt_Dinv_rrt_UUt_Z :: Matrix{T}
    Zt_Dinv     :: Matrix{T}
    Zt_UUt_rrt_Dinv_Z :: Matrix{T}
    Zt_UUt_rrt_UUt_Z    ::Matrix{T}
    Zt_UUt      :: Matrix{T}
    Zt_UUt_D_UUt_Z :: Matrix{T}
    LLt_Zt_UUt_ZL :: Matrix{T}

    #for Hessian wrt τ
    Wt_D_Dinv    :: Matrix{T}
    sqrtDinv_UUt   :: Vector{T}
    Ut_kr_Ut      :: Matrix{T}
    Wt_D_Ut_kr_Utt  :: Matrix{T}

    #for Hessian wrt Lγ
    Zt_Vinv_Z   :: Matrix{T}
    Zt_Vinv     :: Matrix{T}

    #additional needed
    Lt_Zt_Dinv_r :: Vector{T}
    Zt_Dinv_Z_L :: Matrix{T}
    Zt_U :: Matrix{T}
    Ut_D_Dinv_Z :: Matrix{T}
    Zt_Dinv_D_Dinv_Z :: Matrix{T}
    Wt_D_sqrtdiagDinv_UUt :: Matrix{T}
    Zt_UUt_Z_L  :: Matrix{T}

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
    ∇β          = Vector{T}(undef, p)
    ∇τ          = Vector{T}(undef, l)
    ∇Lγ         = Matrix{T}(undef, q, q)
    Hττ         = Matrix{T}(undef, l, l)
    HτLγ        = Matrix{T}(undef, l, q◺)
    HLγLγ       = Matrix{T}(undef, q◺, q◺)
    res         = Vector{T}(undef, n)
    res2        = Vector{T}(undef, n)
    resnrm2     = Vector{T}(undef, n)
    expwτ       = Vector{T}(undef, n)
    ztz         = Z'Z
    ztres       = Vector{T}(undef, q)
    zlltzt_dg   = Vector{T}(undef, n)
    storage_n1  = Vector{T}(undef, n)
    storage_p1  = Vector{T}(undef, p)
    storage_q1  = Vector{T}(undef, q)
    storage_pn  = Matrix{T}(undef, p, n)
    storage_qn  = Matrix{T}(undef, q, n)
    storage_ln  = Matrix{T}(undef, l, n)
    storage_pp  = Matrix{T}(undef, p, p)
    storage_qq  = Matrix{T}(undef, q, q)
    storage_qp  = Matrix{T}(undef, q, p)
    storage_q◺n = Matrix{T}(undef, q◺, n)

    #added for weighted estimating equations
    Dinv        = Vector{T}(undef, n) #stores diag(exp(-wτ_0))
    Ut          = Matrix{T}(undef, q, n)

    #r is the residual vector
    #t is tranpose
    #denote Dinv as diagonal(exp(-wτ))
    #added for weigthed estimating equations
    rt_Dinv_r   = Vector{T}(undef, 1)
    rt_UUt_r    = Vector{T}(undef, 1)
    rt_U        = Matrix{T}(undef, 1, q) #*
    Dinv_r      = Vector{T}(undef, n)
    rt_UUt      = Matrix{T}(undef, 1, n) #*
    Zt_Dinv_r   = Vector{T}(undef, q)
    rt_UUt_Z    = Matrix{T}(undef, 1, q) #*
    diagUUt_Dinv = Vector{T}(undef, n)
    Dinv_Z      = Matrix{T}(undef, n, q)
    Dinv_Z_L    = Matrix{T}(undef, n, q)
    UUt_Z       = Matrix{T}(undef, n, q)
    UUt_Z_L     = Matrix{T}(undef, n, q)
    Ut_D_U   = Matrix{T}(undef, q, q)
    Zt_Dinv_Z   = Matrix{T}(undef, q, q)
    Lt_Zt_Dinv_Z_L = Matrix{T}(undef, q, q)
    Zt_UUt_Z    = Matrix{T}(undef, q, q)
    Lt_Zt_UUt_Z_L = Matrix{T}(undef, q, q)

    diagDVRV = Vector{T}(undef, n)

    #extra from gradient of vechL
    Zt_Dinv_rrt_Dinv_Z = Matrix{T}(undef, q, q)
    Zt_Dinv_rrt_UUt_Z = Matrix{T}(undef, q, q)
    Zt_Dinv     = Matrix{T}(undef, q, n) # need? have Dinv_Z
    Zt_UUt_rrt_Dinv_Z = Matrix{T}(undef, q, q)
    Zt_UUt_rrt_UUt_Z = Matrix{T}(undef, q, q) #added this 
    Zt_UUt      = Matrix{T}(undef, q, n) # need? same as UUt_Z
    Zt_UUt_D_UUt_Z = Matrix{T}(undef, q, q)
    LLt_Zt_UUt_ZL   = Matrix{T}(undef, q, q)

    #for Hessian wrt τ
    Wt_D_Dinv = Matrix{T}(undef, l, n)
    sqrtDinv_UUt   = Vector{T}(undef, n)
    Ut_kr_Ut      = Matrix{T}(undef, q^2, n)
    Wt_D_Ut_kr_Utt  = Matrix{T}(undef, l, q^2)
    

    #for Hessian wrt Lγ
    Zt_Vinv_Z   = Matrix{T}(undef, q, q)
    Zt_Vinv     = Matrix{T}(undef, q, n)

    #added under mom !!! need to add stil 
    Lt_Zt_Dinv_r = Vector{T}(undef, q)
    Zt_Dinv_Z_L = Matrix{T}(undef, q, q)
    Zt_U = Matrix{T}(undef, q, q)
    Ut_D_Dinv_Z = Matrix{T}(undef, q, q)
    Zt_Dinv_D_Dinv_Z = Matrix{T}(undef, q, q)
    Wt_D_sqrtdiagDinv_UUt = Matrix{T}(undef, l, n)
    Zt_UUt_Z_L = Matrix{T}(undef, q, q)

    # constructor
    VarLmmObs{T}(
        y, transpose(X), transpose(Z), transpose(W), 
        ∇β, ∇τ, ∇Lγ,
        Hττ, HτLγ, HLγLγ,
        res, res2, resnrm2, expwτ, ztz, ztres, zlltzt_dg,
        storage_n1, storage_p1, storage_q1,
        storage_pn, storage_qn, storage_ln, 
        storage_pp, storage_qq, storage_qp, storage_q◺n,
        Dinv, Ut, rt_Dinv_r, rt_UUt_r, rt_U, Dinv_r, rt_UUt, 
        Zt_Dinv_r, rt_UUt_Z, diagUUt_Dinv, Dinv_Z, Dinv_Z_L, UUt_Z,
        UUt_Z_L, Ut_D_U, Zt_Dinv_Z, Lt_Zt_Dinv_Z_L,
        Zt_UUt_Z, Lt_Zt_UUt_Z_L, diagDVRV, Zt_Dinv_rrt_Dinv_Z,
        Zt_Dinv_rrt_UUt_Z, Zt_Dinv, Zt_UUt_rrt_Dinv_Z, Zt_UUt_rrt_UUt_Z,
        Zt_UUt, Zt_UUt_D_UUt_Z, LLt_Zt_UUt_ZL, Wt_D_Dinv, 
        sqrtDinv_UUt, Ut_kr_Ut, Wt_D_Ut_kr_Utt, Zt_Vinv_Z, Zt_Vinv,
        Lt_Zt_Dinv_r, Zt_Dinv_Z_L, Zt_U, Ut_D_Dinv_Z, 
        Zt_Dinv_D_Dinv_Z, Wt_D_sqrtdiagDinv_UUt, Zt_UUt_Z_L)
end

"""
    VarLmmModel

Variance linear mixed model, which contains a vector of 
`VarLmmObs` as data, model parameters, and working arrays.

TODO: function documentation
"""
struct VarLmmModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    data :: Vector{VarLmmObs{T}}
    p    :: Int       # number of mean parameters in linear regression
    q    :: Int       # number of random effects
    l    :: Int       # number of parameters for modeling WS variability
    # parameters
    β    :: Vector{T}  # p-vector of mean regression coefficients
    τ    :: Vector{T}  # l-vector of WS variability regression coefficients
    Lγ   :: Matrix{T}  # q by q lower triangular cholesky factor of random effects  var-covar matrix pertaining to γ
    # working arrays
    ∇β   :: Vector{T}
    ∇τ   :: Vector{T}
    ∇Lγ  :: Matrix{T}
    Hττ  :: Matrix{T}
    HτLγ :: Matrix{T}
    HLγLγ:: Matrix{T}
    # weighted model or not
    weighted :: Vector{Bool}
end

function VarLmmModel(obsvec::Vector{VarLmmObs{T}}) where T <: BlasReal
    # dimensions
    p     = size(obsvec[1].Xt, 1)
    q, l  = size(obsvec[1].Zt, 1), size(obsvec[1].Wt, 1)
    q◺    = ◺(q)
    # parameters
    β     = Vector{T}(undef, p)
    τ     = Vector{T}(undef, l)
    Lγ    = Matrix{T}(undef, q, q)
    # gradients
    ∇β    = Vector{T}(undef, p)
    ∇τ    = Vector{T}(undef, l)
    ∇Lγ   = Matrix{T}(undef, q, q)
    Hττ   = Matrix{T}(undef, l, l)
    HτLγ  = Matrix{T}(undef, l, q◺)
    HLγLγ = Matrix{T}(undef, q◺, q◺)
    # weighted fitting or not
    # constructor
    VarLmmModel{T}(
        obsvec, p, q, l,
        β,  τ,  Lγ,
        ∇β, ∇τ, ∇Lγ,
        Hττ, HτLγ, HLγLγ, [false])
end

include("mom.jl")
include("mom_nlp.jl")
# include("mom_nlp_unconstr.jl")
include("df.jl")
include("multivariate_calculus.jl")

end

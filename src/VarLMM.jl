module VarLMM

using DataFrames, Distributions, LinearAlgebra, MathProgBase
using Permutations, Reexport, Statistics, StatsModels
import LinearAlgebra: BlasReal, copytri!
import DataFrames: DataFrame
@reexport using Ipopt
@reexport using NLopt

export VarLmmObs, VarLmmModel
export DataFrame, fit!, init_ls!, mom_obj!, update_res!

"""
    VarLmmObs
    VarLmmObs(y, X, Z, W)

A realization of variance linear mixed model data instance.
"""
struct VarLmmObs{T <: BlasReal}
    # data
    y          :: Vector{T} # response 
    Xt          :: Matrix{T} # X should include a column of 1's
    Zt          :: Matrix{T} # Random effect covars
    Wt          :: Matrix{T} # Covariates that affect WS variability
    # working arrays
    ∇β         :: Vector{T} # gradient wrt β
    ∇τ         :: Vector{T} # gradient wrt τ
    ∇Lγ        :: Matrix{T} # gradient wrt L cholesky factor 
    Hττ        :: Matrix{T} # hessian
    HτLγ       :: Matrix{T}
    HLγLγ      :: Matrix{T}
    res        :: Vector{T} # residual vector
    res2       :: Vector{T} # residual vector.^2
    resnrm2    :: Vector{T} # sum of residual squares
    expwτ      :: Vector{T} # hold exp.(W * τ)
    ztz        :: Matrix{T} # Z'Z
    ztres      :: Vector{T} # Z'res
    zlltzt_dg  :: Vector{T}
    storage_n1 :: Vector{T}
    storage_q1 :: Vector{T}
    storage_qn :: Matrix{T}
    storage_ln :: Matrix{T}    
    storage_qq :: Matrix{T}
    storage_q◺n :: Matrix{T}
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
    storage_q1  = Vector{T}(undef, q)
    storage_qn  = Matrix{T}(undef, q, n)
    storage_ln  = Matrix{T}(undef, l, n)
    storage_qq  = Matrix{T}(undef, q, q)
    storage_q◺n = Matrix{T}(undef, q◺, n)
    # constructor
    VarLmmObs{T}(
        y, transpose(X), transpose(Z), transpose(W), 
        ∇β, ∇τ, ∇Lγ,
        Hττ, HτLγ, HLγLγ,
        res, res2, resnrm2, expwτ, ztz, ztres, zlltzt_dg,
        storage_n1, storage_q1,
        storage_qn, storage_ln, storage_qq, storage_q◺n)
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
    # constructor
    VarLmmModel{T}(
        obsvec, p, q, l,
        β,  τ,  Lγ,
        ∇β, ∇τ, ∇Lγ,
        Hττ, HτLγ, HLγLγ)
end

include("mom.jl")
include("mom_nlp_constr.jl")
# include("mom_nlp_unconstr.jl")
include("df.jl")
include("multivariate_calculus.jl")

end

module VarLMM

using DataFrames, Distributions, KNITRO, LinearAlgebra, MathProgBase, MixedModels
using Reexport, Statistics, StatsModels
import LinearAlgebra: BlasReal, copytri!
import DataFrames: DataFrame
@reexport using Ipopt
@reexport using NLopt

export VarLmmObs, VarLmmModel
export DataFrame, fit!, mom_obj!, mgf_γω

"""
    VarLmmObs
    VarLmmObs(y, X, Z, W)

A realization of variance linear mixed model data instance.
"""
struct VarLmmObs{T <: BlasReal}
    # data
    y::Vector{T} # response 
    X::Matrix{T} # X should include a column of 1's
    Z::Matrix{T} # Random effect covars
    W::Matrix{T} # Covariates that affect WS variability
    # working arrays
    ∇β    ::Vector{T}     # gradient wrt β
    ∇τ    ::Vector{T}     # gradient wrt τ
    ∇Lγ   ::Matrix{T}     # gradient wrt L cholesky factor 
    ∇lγω  ::Vector{T} 
    ∇lω   ::Vector{T}  
    res   ::Vector{T}     # residual vector
    expwτ ::Vector{T}     # hold exp.(W * τ)
    R     ::Matrix{T}     # hold variance residuals
    storage_nn ::Matrix{T}
    storage_nq ::Matrix{T}
    storage_qq ::Matrix{T}
    storage_n1 ::Vector{T}
    storage_q1 ::Vector{T}
end

function VarLmmObs(
    y::AbstractVector{T},
    X::AbstractMatrix{T},
    Z::AbstractMatrix{T},
    W::AbstractMatrix{T}
    ) where T <: BlasReal
    n, p, q, l = size(X, 1), size(X, 2), size(Z, 2), size(W, 2)
    # working arrays
    ∇β          = Vector{T}(undef, p)
    ∇τ          = Vector{T}(undef, l)
    ∇Lγ         = Matrix{T}(undef, q, q)
    ∇lγω        = Vector{T}(undef, q)
    ∇lω         = Vector{T}(undef, 1)
    res         = Vector{T}(undef, n)
    expwτ       = Vector{T}(undef, n)
    R           = Matrix{T}(undef, n, n)
    storage_nn  = Matrix{T}(undef, n, n)
    storage_nq  = Matrix{T}(undef, n, q)
    storage_qq  = Matrix{T}(undef, q, q)
    storage_n1  = Vector{T}(undef, n)
    storage_q1  = Vector{T}(undef, q)
    # constructor
    VarLmmObs{T}(
        y, X, Z, W, 
        ∇β, ∇τ, ∇Lγ, ∇lγω, ∇lω,
        res, expwτ, R,
        storage_nn, storage_nq, storage_qq,
        storage_n1, storage_q1)
end

# TODO: work on function documentation
"""
    VarLmmModel

Variance linear mixed model, which contains a vector of 
`VarLmmObs` as data, model parameters, and working arrays.

**Function**
varlmmModel(f1::FormulaTerm,
f2::FormulaTerm,
df::DataFrame,
idvar::Union{String, Symbol})

This is the constructor function to create a varlmmModel object
That is fit initialized to be fit from a dataframe, where 
f1 specifies the formula for the mean response where effect sizes
will be estimated with β i.e. @formula(y ~ age). f2 specifices 
the formula for the within-subject variance where effect sizes 
will be  estimated with τ. the lhs of the formula can be anything
i.e. @formula(var ~ x). `df` is the dataframe with those variables.
`idvar` is the variable in the dataframe specifying the ID variable
for the subjects. 

Example
varlmmModel(@formula(Y ~ X + W), @formula(Var ~ X + H),
dataset, :PID)
"""
struct VarLmmModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{VarLmmObs{T}}
    p   ::Int       # number of mean parameters in linear regression
    q   ::Int       # number of random effects
    l   ::Int       # number of parameters for modeling WS variability
    npar::Int       # total number of parameters estimated
    # parameters
    β   ::Vector{T}  # p-vector of mean regression coefficients
    τ   ::Vector{T}  # l-vector of WS variability regression coefficients
    Lγ  ::Matrix{T}  # q by q lower triangular cholesky factor of random effects  var-covar matrix pertaining to γ
    lγω ::Vector{T}  # q by 1 cholesky factor of RE covar matrix for γ,ω
    lω  ::Vector{T}  # 1 x 1 cholesky factor of RE variance for ω
    # working arrays
    ∇β  ::Vector{T}
    ∇τ  ::Vector{T}
    ∇Lγ ::Matrix{T}
    ∇lγω::Vector{T} 
    ∇lω ::Vector{T}
    storage_q :: Vector{T}
end

function VarLmmModel(obsvec::Vector{VarLmmObs{T}}) where T <: BlasReal
    # dimensions
    n, p = length(obsvec), size(obsvec[1].X, 2)
    q, l = size(obsvec[1].Z, 2), size(obsvec[1].W, 2)
    npar = p + l + ((q + 1) * (q + 2)) >> 1
    # parameters
    β    = Vector{T}(undef, p)
    τ    = Vector{T}(undef, l)
    Lγ   = Matrix{T}(undef, q, q)
    lγω  = Vector{T}(undef, q)
    lω   = Vector{T}(undef, 1)
    # gradients
    ∇β   = Vector{T}(undef, p)
    ∇τ   = Vector{T}(undef, l)
    ∇Lγ  = Matrix{T}(undef, q, q)
    ∇lγω = Vector{T}(undef, q)
    ∇lω  = Vector{T}(undef, 1)
    storage_q = Vector{T}(undef, q)
    # constructor
    VarLmmModel{T}(
        obsvec, p, q, l, npar,
         β,  τ,  Lγ,  lγω,  lω,
        ∇β, ∇τ, ∇Lγ, ∇lγω, ∇lω,
        storage_q)
end

# TODO: constructor from dataframe as in previous code

include("mom.jl")
include("mom_nlp_constr.jl")
# include("mom_nlp_unconstr.jl")
include("df.jl")

end
module VarLMM

using DataFrames, Distributions, KNITRO, LinearAlgebra, MathProgBase, MixedModels
using Reexport, Statistics, StatsModels, LsqFit
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

function VarLmmModel(
    f1::FormulaTerm, # formula for mean with random effects 
    f2::FormulaTerm, # formula for WS variance
    df, # dataframe containing all variables
    idvar::Union{Symbol, String}) # contains ID var)

    # read in dataframe
    # fit model
    # create varLMMobs and model
    # set parameters to the fit model

    lmm = LinearMixedModel(f1, df)
    MixedModels.fit!(lmm, REML=true)
    Z = copy(transpose(first(lmm.reterms).z))

    if typeof(idvar) <: String
        idvar = Symbol(idvar)
    end

    ids = unique(df[!, idvar])
    npeople = length(ids)
    ntotal = size(Z, 1)
    obsvec = Vector{VarLmmObs{Float64}}(undef, npeople)
    W = modelmatrix(f2, df) 
    for i in eachindex(ids)
        pinds = findall(df[!, idvar] .== ids[i])
        obs = VarLmmObs(lmm.y[pinds], 
        lmm.X[pinds, :], 
        Z[pinds, :], 
        W[pinds, :])
        obsvec[i] = obs
    end
    model = VarLmmModel(obsvec)

    #update model params with LMM-fitted params
    model.β .= lmm.beta 
    update_res!(model)
    @inbounds for j in 1:model.q, i in j:model.q
        model.Lγ[i, j] = first(lmm.lambda)[i, j] * lmm.sigma
    end
    model.lγω .= zeros(Float64, model.q)


    ## set τ[1] to 2log(lmm.sigma), rest 0
    model.τ .= zeros(Float64, model.l)
    model.τ[1] = 2log(lmm.sigma)
    
    ## use NLS on d_is 
    
    Σ = model.Lγ * transpose(model.Lγ)
    d = Vector{Float64}(undef, ntotal)
    start = 1
    for i in 1:length(model.data)
        ni = length(model.data[i].y)
        stop = start + ni - 1
        d[start:stop] = model.data[i].res.^2 - 
           diag(model.data[i].Z * Σ * 
           transpose(model.data[i].Z))
        start = stop + 1
    end
    multimodel(W, τmod) = exp.(W * τmod)
    nls = curve_fit(multimodel, W, d, model.τ)
    copy!(model.τ, nls.param)
    
    #set lω to 0.0
    model.lω[1] = 0.0

    return model
end


include("mom.jl")
include("mom_nlp_constr.jl")
# include("mom_nlp_unconstr.jl")
include("df.jl")

end
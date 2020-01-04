module VarLMM

using LinearAlgebra, MathProgBase, Reexport, Random, Distributions#, MixedModels
using LinearAlgebra: BlasReal, copytri!
@reexport using Ipopt
@reexport using NLopt
@reexport using MixedModels



"""
varlmmObs
varlmmObs(y, X, Z, W, index)
A realization of Variance linear mixed model data instance.
"""
struct varlmmObs{T <: BlasReal}
    # data
    y::Vector{T} # response 
    X::Matrix{T} # X should include a column of 1's
    Z::Matrix{T} # Random effect covars
    W::Matrix{T} # Covariates that affect WS variability
    #index::AbstractArray{T} # Array containing subject level indicies
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇τ::Vector{T}   # gradient wrt τ
    ∇Lγ::Matrix{T}  
    ∇lγω::Vector{T} 
    ∇lω::Vector{T}  # gradient wrt L cholesky factor 
    res::Vector{T}  # residual vector
    #xtx::Matrix{T}  # Xi'Xi (p-by-p)
    #ztz::Matrix{T}  # Zi'Zi (q-by-q)
    #xtz::Matrix{T}  # Xi'Zi (p-by-q)
    storage_nn::Matrix{T}
    storage_qn::Matrix{T}
    storage_qq::Vector{T}
    V::Matrix{T}
end
#storage_q1::Vector{T}
#storage_q2::Vector{T}
# Vchol::Matrix{T}

function varlmmObs(
    y::Vector{T},
    X::Matrix{T},
    Z::Matrix{T},
    W::Matrix{T}
    ) where T <: BlasReal
    n, p, q, l = size(X, 1), size(X, 2), size(Z, 2), size(W, 2)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, l)
    ∇Lγ = Vector{T}(undef, Int((q + 1) * q / 2))
    ∇lγω = Vector{T}(undef, q)
    ∇lω = 0.0
    res = Vector{T}(undef, n)
    #not sure these are needed
    #xtx = transpose(X) * X
    #ztz = transpose(Z) * Z
    #xtz = transpose(X) * Z
    storage_nn = Vector{T}(undef, n, n)
    storage_qn = Matrix{T}(undef, q, n)
    storage_qq = Matrix{T}(undef, q, q)
    V = Matrix{T}(undef, n, n) 
    varlmmObs{T}(y, X, Z, W, 
        ∇β, ∇τ, ∇Lγ, ∇lγω, ∇lω,
        res, #xtx, ztz, xtz,
        storage_nn, storage_qn, storage_qq, V)
end



"""
varlmmModel
varlmmModel
var linear mixed model, which contains a vector of 
`var` as data, model parameters, and working arrays.
"""
struct varlmmModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{varlmmObs{T}}
    #w::Vector{T}    # a vector of weights from bootstraping the subset
    #ntotal::Int     # total number of clusters
    p::Int          # number of mean parameters in linear regression
    q::Int          # number of random effects
    l::Int          # number of parameters for modeling WS variability
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    τ::Vector{T}    # l-vector of WS variability regression coefficients
    Lγ::Matrix{T} # q by q lower triangular cholesky factor of random effects  var-covar matrix pertaining to γ
    lγω::Vector{T} # q by 1 cholesky factor of RE covar matrix for γ,ω
    lω::T        # 1 x 1 cholesky factor of RE variance for ω
    # working arrays
    ∇β::Vector{T}   # gradient from all observations
    ∇τ::Vector{T}
    ∇Lγ::Matrix{T}  
    ∇lγω::Vector{T} 
    ∇lω::T  # gradient wrt L cholesky factor 
    Wτ::Vector{T}   #holds Wτ vector elements
    storage_qq::Matrix{T}
    storage_nq::Matrix{T}
    storage_nn::Matrix{T} #to hold matrix to take frobenius norm of
end

function varlmmModel(obsvec::Vector{varlmmObs{T}}) where T <: BlasReal
    n, p, q, l = length(obsvec), size(obsvec[1].X, 2), size(obsvec[1].Z, 2), size(obsvec[1].W, 2)
    npar = p + l + abs2(q + 1) >> 1
    # since X includes a column of 1, p is the number of mean parameters
    # the cholesky factor for the q+1xq+1 random effect q+1^2 values
    ## the arithmetic shift right operation has the effect of division by 2^n, here n = 1
    ## then there is the error variance
    #w   = ones(T, n) # initialize weights to be 1
    β   = Vector{T}(undef, p)
    τ   = Vector{T}(undef, l)
    Lγ = Matrix{T}(undef, abs2(q)) 
    lγω = Vector{T}(undef, q)
    lω = 0.0
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, l)
    ∇Lγ = Matrix{T}(undef, abs2(q)) 
    ∇lγω = Vector{T}(undef, q)
    ∇lω = 0.0
    #Hβ  = Matrix{T}(undef, p, p)
    #Hτ  = Matrix{T}(undef, 1, 1)
    #HΣ  = Matrix{T}(undef, abs2(q), abs2(q))
    Wτ = Vector{T}(undef, n)
    #XtX = zeros(T, p, p) # sum_i xi'xi
    #ntotal = 0
    #for i in eachindex(obsvec)
    #    ntotal  += length(obsvec[i].y)
    #    XtX    .+= obsvec[i].xtx
    #end
    storage_qq = Matrix{T}(undef, q, q)
    storage_nq = Matrix{T}(undef, n, q)
    storage_nn = Matrix{T}(undef, n, n)
    
    varlmmModel{T}(obsvec, w, ntotal, p, q, 
        β, τ, Lγ, lγω, lω,
        ∇β, ∇τ, ∇Lγ, ∇lγω, ∇lω, #∇Σ, #Hβ, Hτ, HΣ, XtX,
        Wτ, storage_qq, storage_nq, storage_nn)
end


include("MoMEst.jl")

end
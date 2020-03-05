module VarLMM

using LinearAlgebra, MathProgBase, Reexport, Distributions, Statistics, MixedModels
using StatsModels
using LinearAlgebra: BlasReal, copytri!
@reexport using Ipopt
@reexport using NLopt
#@reexport using MixedModels

export varlmmObs, varlmmModel
export fit!, MoMobjf!, init_β_τ!, vech!, vec2ltri!


"""
varlmmObs
varlmmObs(y, X, Z, W, index)
A realization of Variance linear mixed model data instance.
"""
struct varlmmObs{T <: BlasReal}
    # data
    y::AbstractVector{T} # response 
    X::AbstractMatrix{T} # X should include a column of 1's
    Z::AbstractMatrix{T} # Random effect covars
    W::AbstractMatrix{T} # Covariates that affect WS variability
    #index::AbstractArray{T} # Array containing subject level indicies
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇τ::Vector{T}   # gradient wrt τ
    ∇Lγ::Vector{T}  
    ∇lγω::Vector{T} 
    ∇lω::Vector{T}  # gradient wrt L cholesky factor 
    res::Vector{T}  # residual vector
    #xtx::Matrix{T}  # Xi'Xi (p-by-p)
    #ztz::Matrix{T}  # Zi'Zi (q-by-q)
    #xtz::Matrix{T}  # Xi'Zi (p-by-q)
    Wτ::Vector{T}   #holds Wτ vector elements
    storage_nn::Matrix{T}
    storage_qn::Matrix{T}
    storage_qq::Matrix{T}
    storage_n1::Vector{T}
    storage_qq2::Matrix{T}
    V::Matrix{T}
end
#storage_q1::Vector{T}
#storage_q2::Vector{T}
# Vchol::Matrix{T}

function varlmmObs( #ri, Ri, Vi, res 
    y::AbstractVector{T},
    X::AbstractMatrix{T},
    Z::AbstractMatrix{T},
    W::AbstractMatrix{T}
    ) where T <: BlasReal
    n, p, q, l = size(X, 1), size(X, 2), size(Z, 2), size(W, 2)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, l)
    ∇Lγ = Vector{T}(undef, Int((q + 1) * q / 2))
    ∇lγω = Vector{T}(undef, q)
    ∇lω = Vector{T}(undef, 1)
    res = Vector{T}(undef, n)
    #not sure these are needed
    #xtx = transpose(X) * X
    #ztz = transpose(Z) * Z
    #xtz = transpose(X) * Z
    Wτ = Vector{T}(undef, n)
    storage_nn = Matrix{T}(undef, n, n)
    storage_qn = Matrix{T}(undef, q, n)
    storage_qq = Matrix{T}(undef, q, q)
    storage_n1 = Vector{T}(undef, n)
    storage_qq2 = Matrix{T}(undef, q, q)
    V = Matrix{T}(undef, n, n) 
    varlmmObs{T}(y, X, Z, W, 
        ∇β, ∇τ, ∇Lγ, ∇lγω, ∇lω,
        res, Wτ, #xtx, ztz, xtz,
        storage_nn, storage_qn, storage_qq,
        storage_n1, storage_qq2, V)
end



"""
varlmmModel
varlmmModel
var linear mixed model, which contains a vector of 
`var` as data, model parameters, and working arrays.

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
struct varlmmModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{varlmmObs{T}}
    #w::Vector{T}    # a vector of weights from bootstraping the subset
    #ntotal::Int     # total number of clusters
    p::Int          # number of mean parameters in linear regression
    q::Int          # number of random effects
    l::Int          # number of parameters for modeling WS variability
    npar::Int # total number of parameters estimated
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    τ::Vector{T}    # l-vector of WS variability regression coefficients
    Lγ::Matrix{T} # q by q lower triangular cholesky factor of random effects  var-covar matrix pertaining to γ
    lγω::Vector{T} # q by 1 cholesky factor of RE covar matrix for γ,ω
    lω::Vector{T}        # 1 x 1 cholesky factor of RE variance for ω
    # working arrays
    ∇β::Vector{T}   # gradient from all observations
    ∇τ::Vector{T}
    ∇Lγ::Vector{T}  
    ∇lγω::Vector{T} 
    ∇lω::Vector{T}  # gradient wrt L cholesky factor 
    #Wτ::Vector{T}   #holds Wτ vector elements
    #storage_qq::Matrix{T}
    #storage_nq::Matrix{T}
    #storage_nn::Matrix{T} #to hold matrix to take frobenius norm of
end

function varlmmModel(obsvec::Vector{varlmmObs{T}}) where T <: BlasReal
    n, p, q, l = length(obsvec), size(obsvec[1].X, 2), size(obsvec[1].Z, 2), size(obsvec[1].W, 2)
    npar = p + l + ((q + 1) * (q + 2)) >> 1
    # since X includes a column of 1, p is the number of mean parameters
    # the cholesky factor for the q+1xq+1 random effect q+1^2 values
    ## the arithmetic shift right operation has the effect of division by 2^n, here n = 1
    ## then there is the error variance
    #w   = ones(T, n) # initialize weights to be 1
    β   = Vector{T}(undef, p)
    τ   = Vector{T}(undef, l)
    #Lγ = Matrix{T}(undef, q, q) 
    Lγ = Matrix{T}(I, q, q)
    #lγω = Vector{T}(undef, q)
    lγω = ones(T, q)
    #lω = Vector{T}(undef, 1)
    lω = ones(T, 1)
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, l)
    ∇Lγ = Vector{T}(undef, Int((q + 1) * q / 2)) 
    ∇lγω = Vector{T}(undef, q)
    ∇lω = Vector{T}(undef, 1)
    #Hβ  = Matrix{T}(undef, p, p)
    #Hτ  = Matrix{T}(undef, 1, 1)
    #HΣ  = Matrix{T}(undef, abs2(q), abs2(q))
    #XtX = zeros(T, p, p) # sum_i xi'xi
    #ntotal = 0
    #for i in eachindex(obsvec)
    #    ntotal  += length(obsvec[i].y)
    #    XtX    .+= obsvec[i].xtx
    #end
    #storage_qq = Matrix{T}(undef, q, q)
    #storage_nq = Matrix{T}(undef, n, q)
    #storage_nn = Matrix{T}(undef, n, n)
    
    varlmmModel{T}(obsvec, p, q, l, npar,
        β, τ, Lγ, lγω, lω,
        ∇β, ∇τ, ∇Lγ, ∇lγω, ∇lω)#, #∇Σ, #Hβ, Hτ, HΣ, XtX,
        #Wτ, storage_qq, storage_nq, storage_nn)
end


function varlmmModel(
    f1::FormulaTerm, # formula for mean with random effects 
    f2::FormulaTerm, # formula for WS variance
    df, # dataframe containing all variables
    idvar::Union{Symbol, String}) # contains ID var)

    # read in dataframe
    # fit model
    # create varLMMobs and model
    # set parameters to the fit model

    lmm = LinearMixedModel(f1, df)
    Z = copy(transpose(first(lmm.reterms).z))

    MixedModels.fit!(lmm, REML=true)

    if typeof(idvar) <: String
        idvar = Symbol(idvar)
    end
    ids = unique(df[!, idvar])
    npeople = length(ids)
    obsvec = Vector{varlmmObs{Float64}}(undef, npeople)
    W = modelmatrix(f2, df) 
    for i in eachindex(ids)
        pinds = findall(df[!, idvar] .== ids[i])
        obs = varlmmObs(view(lmm.y, pinds), 
        view(lmm.X, pinds, :), 
        view(Z, pinds, :), 
        view(W, pinds, :))
        obsvec[i] = obs
    end
    model = varlmmModel(obsvec)
    model.β .= lmm.beta 
    update_res!(model)
    #model.Lγ .= first(lmm.lambda) * lmm.sigma
    extractLγ!(model, first(lmm.lambda), lmm.sigma)
    #
    # model.τ .= zeros(Float64, model.l)
    # model.τ[1] = 2log(lmm.sigma)
    #model.lω[1] = lmm.sigma > 1 ? 2log(lmm.sigma) : 1
    # model.lω[1] = 10e-2
    model.lγω .= zeros(Float64, model.q)
    wtypseudo = zeros(Float64, model.l)
    wtw = zeros(Float64, model.l, model.l)
    Σ = model.Lγ * transpose(model.Lγ)
    Di = []
    for i in 1:length(model.data)
        #ypseudo = fill(2log(lmm.sigma), length(model.data[i].y))
        di = model.data[i].res.^2 - 
           diag(model.data[i].Z * Σ * 
           transpose(model.data[i].Z))
        #di = model.data[i].res.^2 - 0.001 * diag(model.data[i].Z *
        #    Σ * transpose(model.data[i].Z))
        #posinds = findall(di .> 0)
        #neginds = findall(di .<= 0)
        #di[neginds] .= 0.0001
        #wtypseudo += transpose(view(model.data[i].W, posinds, :)) * 
        #(log.(view(di, posinds)))
        #wtypseudo += transpose(model.data[i].W) * 
        #log.(di)
        # BLAS.syrk!('U', 'T', one(Float64),
        # model.data[i].W, one(Float64), wtw)
        #wtw += transpose(view(model.data[i].W, posinds, :)) *
        # view(model.data[i].W, posinds, :)
        wtw += transpose(model.data[i].W) * model.data[i].W
        #  wtypseudo += transpose(model.data[i].W) * ypseudo
        Di = vcat(Di, di)
    end
    Di_add = abs(minimum(Di)) + 0.00001
    for i in 1:length(model.data)
        di = model.data[i].res.^2 - 
           diag(model.data[i].Z * Σ * 
           transpose(model.data[i].Z)) .+ Di_add
        wtypseudo += transpose(model.data[i].W) * 
        log.(di)
    end
    ldiv!(model.τ, cholesky(Symmetric(wtw)), wtypseudo)
    #ntotal = size(W, 1)
    #rest = Di - W * model.τ
    #model.lω[1] = sqrt(sum(abs2, rest) / (ntotal - model.l))
    #model.lω[1] = 2 * sum(rest) / (ntotal)
    model.lω[1] = 0.0
    return model
end

include("MoMEst.jl")

end
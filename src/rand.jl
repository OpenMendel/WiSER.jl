"""
    rand!(m::WSVarLmmModel; respdist = MvNormal, γωdist = MvNormal, Σγω = [], kwargs...)

Replaces the responses `m.data[i].y` with a simulated response based on:
- The data in the model object's data `X, Z, W` matrices. 
- The parameter values in the model.
- The condistribution distribution of the response given the random effects.
- The distribution of the random effects.
- If simulating from MvTDistribution, you must specify the degrees of freedom via `df = x`.
"""
function rand!(
    m :: WSVarLmmModel; 
    respdist = MvNormal, 
    γωdist   = MvNormal, 
    Σγω      = [], 
    kwargs...)
    q = m.q
    isempty(Σγω) ? Σγω = [m.Lγ * transpose(m.Lγ) zeros(q); 
        zeros(1, q) 0.0] : Σγω
    Lγω = cholesky(Symmetric(Σγω), check = false).L
    Lγ    = Lγω[1:q, 1:q]
    lγω   = Lγω[q + 1, 1:q]
    lω    = Lγω[q + 1, q + 1]
    γω = Vector{Float64}(undef, q + 1)
    z  = similar(γω)
    for ob in m.data
        copyto!(ob.y, rvarlmm(transpose(ob.Xt), 
        transpose(ob.Zt), transpose(ob.Wt),
        m.β, m.τ, Lγω, Lγ, lγω, γω, z, 
        respdist, kwargs...))
    end
end

"""
    rvarlmm(Xs::Array{Matrix}, Zs::Array{Matrix},
    Ws::Array{Matrix}, β::Vector, τ::Vector;
    respdist = MvNormal, Σγ=[], Σγω=[])

Generate a simulated response from the `WSVarLmmModel` based on:
- `Xs`: array of each clusters `X`: mean fixed effects covariates
- `Zs`: array of each clusters `Z`: random location effects covariates
- `Ws`: array of each clusters `W`: within-subject variance fixed effects covariates
- `β`: mean fixed effects vector
- `τ`: within-subject variance fixed effects vector
- `respdist`: the distribution for response. Default is MvNormal. 
- `Σγ`: random location effects covariance matrix. 
- `Σγω`: joint random location and random scale effects covariance matrix (if generating from full model)

Output is an array of simulated responses that match the ordering in `Xs, Zs, Ws`.

-----------

    rvarlmm!(meanformula::FormulaTerm, reformula::FormulaTerm, 
    wsvarformula::FormulaTerm, idvar::Union{Symbol, String},
    datatable, β::Vector, τ::Vector; respdist = MvNormal, Σγ=[], Σγω=[],
    respname::Symbol = :y)

Generate a simulated response from the VarLMM model based on a DataFrame `datatable`. 
Note: **the datatable MUST be ordered by grouping variable for it to generate in the correct order.
This can be checked via `datatable == sort(datatable, idvar)`. The response is based on:

- `meanformula`: represents the formula for the mean fixed effects `β` (variables in X matrix)
- `reformula`: represents the formula for the mean random effects γ (variables in Z matrix)
- `wsvarformula`: represents the formula for the within-subject variance fixed effects τ (variables in W matrix)
- `idvar`: the id variable for groupings.
- `datatable`: DataFrame for the model. For this function it **must be in order**.
- `β`: mean fixed effects vector
- `τ`: within-subject variance fixed effects vector
- `respdist`: the distribution for response. Default is MvNormal. 
- `Σγ`: random location effects covariance matrix. 
- `Σγω`: joint random location and random scale effects covariance matrix (if generating from full model)
- `respname`: symbol representing the simulated response variable name.

`rvarlmm!()` produces a column in the datatable representing the new response. 
It also returns the column `y` as a vector. 
The datatable **must** be ordered by the ID variable for generated responses to match.
"""
function rvarlmm(Xs::Array{Matrix{T}}, Zs::Array{Matrix{T}},
    Ws::Array{Matrix{T}}, β::Vector{T},
    τ::Vector{T}; respdist = MvNormal, Σγ=[], Σγω=[], kwargs...,
    ) where T <: BlasReal

    @assert length(Xs) == length(Zs) == length(Ws) "Number of provided X, Z, and W matrices do not match"
    isempty(Σγ) && isempty(Σγω) && error("Neither the covariance matrix for γ
    nor the covariance matrix for (γ, ω) have been specified. One must be.")

    q = size(Zs[1], 2)
    
    # Get Cholesky Factor
    isempty(Σγω) ? Σγω = [Σγ zeros(q); zeros(1, q) 0.0] : Σγω
    Lγω = cholesky(Symmetric(Σγω), check = false).L
    Lγ    = Lγω[1:q, 1:q]
    lγω   = Lγω[q + 1, 1:q]
    lω    = Lγω[q + 1, q + 1]

    γω = Vector{Float64}(undef, q + 1)
    z  = similar(γω)
    y = map(i -> rvarlmm(Xs[i], Zs[i], Ws[i], β, τ, Lγω, Lγ,
        lγω, γω, z, respdist),
        1:length(Xs))
    return y
end

function rvarlmm!(meanformula::FormulaTerm, reformula::FormulaTerm, 
    wsvarformula::FormulaTerm, idvar::Union{Symbol, String},
    datatable::DataFrame, β::Vector{T}, τ::Vector{T}; respdist = MvNormal, Σγ=[], Σγω=[],
    respname::Symbol = :y, kwargs...
    ) where T <: BlasReal

    isempty(Σγ) && isempty(Σγω) && error("Neither the covariance matrix for γ
    nor the covariance matrix for (γ, ω) have been specified. One must be.")

    if typeof(idvar) <: String
        idvar = Symbol(idvar)
    end

    function rvarlmmob(f1, f2, f3, subdata,
        β, τ, Lγω, Lγ, lγω, γω, z, respdist)
        X = modelmatrix(meanformula, subdata)
        Z = modelmatrix(reformula, subdata)
        W = modelmatrix(wsvarformula, subdata)
        return rvarlmm(X, Z, W, β, τ, Lγω, Lγ, lγω, γω, z, respdist)
    end

    #apply df-wide schema
    meanformula = apply_schema(meanformula, schema(meanformula, datatable))
    reformula = apply_schema(reformula, schema(reformula, datatable))
    wsvarformula = apply_schema(wsvarformula, schema(wsvarformula, datatable))
    
    q = length(StatsModels.coefnames(reformula.rhs))
    # Get Cholesky Factor
    isempty(Σγω) ? Σγω = [Σγ zeros(q); zeros(1, q) 0.0] : Σγω
    Lγω = cholesky(Symmetric(Σγω), check = false).L
    Lγ    = Lγω[1:q, 1:q]
    lγω   = Lγω[q + 1, 1:q]
    lω    = Lγω[q + 1, q + 1]

    γω = Vector{Float64}(undef, q + 1)
    z  = similar(γω)

    # Need sortperm of groupby var... 
    # if typeof(datatable) <: IndexedTable
    #     y = JuliaDB.groupby(x -> rvarlmmob(meanformula, reformula, wsvarformula,
    #         x, β, τ, Lγω, Lγ, lγω, γω, z, respdist, kwargs...), datatable, idvar) |> 
    #         x -> column(x, 2) |> x -> vcat(x...) 
    #     datatable = JuliaDB.transform(datatable, respname => y)
    # else
    #     y = JuliaDB.groupby(x -> rvarlmmob(meanformula, reformula, wsvarformula,
    #         x, β, τ, Lγω, Lγ, lγω, γω, z, respdist, kwargs...), table(datatable), idvar) |> 
    #         x -> column(x, 2) |> x -> vcat(x...)
    y = combine(x -> rvarlmmob(meanformula, reformula, wsvarformula,
        x, β, τ, Lγω, Lγ, lγω, γω, z, respdist, kwargs...),
        groupby(datatable, idvar)) |>
        x -> x[!, 2]
    datatable[!, respname] = y
    # end
    return datatable
end

function rvarlmm(X, Z, W, β, τ, Lγω, Lγ, lγω, γω, z, respdist, kwargs...)
    q = size(Lγ, 1)
    mul!(γω, Lγω, Distributions.rand!(Normal(), z))
    # generate y
    μy = X * β + Z * γω[1:q]
    @views vy = exp.(W * τ .+ dot(γω[1:q], lγω) .+ γω[end])
    y = eval_respdist(μy, vy, respdist; kwargs...)
    return y
end

function eval_respdist(μy, vy, respdist; df = [])
    if respdist == MvNormal || respdist == Normal
        return rand(MvNormal(μy, Diagonal(vy)))
    elseif respdist == MvTDist 
        isempty(df) ? error("degree of freedom for MvTDist not specified, use 'df' = x.") : 
        return rand(MvTDist(df, μy, Matrix(Diagonal(vy))))
    elseif respdist == Gamma
        θparam = vy ./ μy
        αparam = abs2.(μy) ./ vy
        all(θparam .> 0) && all(αparam .> 0) || 
            error("The current parameter/data does not allow for Gamma to be used. α, θ params must be > 0.")
        return map((α, θ) -> rand(Gamma(α, θ)), αparam, θparam)
    elseif respdist == InverseGaussian
        λparam = μy.^3 ./ vy
        return map((μ , λ) -> rand(InverseGaussian(μ , λ)), μy, λparam)
    elseif respdist == InverseGamma
        αparam = (abs2.(μy) .- 2) ./ vy
        θparam = μy .* (αparam .- 1)
        all(θparam .> 0) && all(αparam .> 0) || 
            error("The current parameter/data does not allow for InverseGamma to be used. α, θ params must be > 0.")
        return map((α, θ) -> rand(InverseGamma(α, θ)), αparam, θparam)    
    elseif respdist == Uniform
        bparams = μy .+ 0.5sqrt.(12vy)
        aparams = 2μy .- bparams
        return map((a, b) -> rand(InverseGamma(a, b)), aparams, bparams) 
    else
        error("Response distribution $respdist is not valid. Run respdists() to see available options.")
    end
end 

respdists() = [:MvNormal, :MvTDist, :Gamma, :InverseGaussian, :InverseGamma, :Uniform]

"""
    DataFrame(m::VarLmmModel)

Convert the data in `VarLmmModel` to a `DataFrame`.
"""
function DataFrame(m::VarLmmModel)
    n = sum(vlmmobs -> length(vlmmobs.y), m.data)
    p, q, l = m.p, m.q, m.l
    # preallocate arrays
    id = Vector{Int}(undef, n)
    y  = Vector{Float64}(undef, n)
    X  = Matrix{Float64}(undef, n, p)
    Z  = Matrix{Float64}(undef, n, q)
    W  = Matrix{Float64}(undef, n, l)
    # gather data
    offset = 1
    for i in 1:length(m.data)
        vlmmobs = m.data[i]
        ni = length(vlmmobs.y)
        rangei = offset:(offset + ni - 1)
        id[rangei]   .= i
         y[rangei]    = vlmmobs.y
         X[rangei, :] = transpose(vlmmobs.Xt)
         Z[rangei, :] = transpose(vlmmobs.Zt)
         W[rangei, :] = transpose(vlmmobs.Wt)
        offset += ni
    end
    df = hcat(DataFrame(id = id, y = y), 
        DataFrame(X, [Symbol("x$i") for i in 1:p]), 
        DataFrame(Z, [Symbol("z$i") for i in 1:q]), 
        DataFrame(W, [Symbol("w$i") for i in 1:l]))
    categorical!(df, :id)
end


"""
    VarLmmModel(meanformula::FormulaTerm, reformula::FormulaTerm, 
    wsvarformula::FormulaTerm, idvar::Union{String, Symbol}, datatable)

Constructor of `VarLmmModel` from a `DataFrame`. `meanformula` represents the formula for
the mean fixed effects β (variables in X matrix), `reformula` represents the formula for 
the mean random effects γ (variables in Z matrix), `wsvarformula` represents the formula 
for the within-subject variance fixed effects τ (variables in W matrix). `idvar` is the
id variable for groupings. `data` is the data table holding all of the data for the model.
It can be a dataframe or column-based table. 

Example:
vlmm3 = VarLmmModel(@formula(y ~ 1 + x2 + x3 + x4 + x5),
    @formula(y ~ 1 + z2 + z3), @formula(y ~ 1 + w2 + w3 + w4 + w5), "id", df)
"""
function VarLmmModel(meanformula::FormulaTerm, reformula::FormulaTerm, 
    wsvarformula::FormulaTerm, idvar::Union{String, Symbol}, datatable)

    if typeof(idvar) <: String
        idvar = Symbol(idvar)
    end

    function varlmmobs(tab)
        y, X = modelcols(meanformula, tab)
        Z = modelmatrix(reformula, tab)
        W = modelmatrix(wsvarformula, tab)
        return VarLmmObs(y, X, Z, W)
    end

    #apply df-wide schema
    meanformula = apply_schema(meanformula, schema(meanformula, datatable))
    reformula = apply_schema(reformula, schema(reformula, datatable))
    wsvarformula = apply_schema(wsvarformula, schema(wsvarformula, datatable))
    
    meanname = "β: " .* coefnames(meanformula.rhs)
    rename = "γ: " .* coefnames(reformula.rhs)
    wsvarname = "τ: " .* coefnames(wsvarformula.rhs)

    #now form observations 
    if typeof(datatable) <: IndexedTable
        varlmm = JuliaDB.groupby(varlmmobs, datatable, idvar) |> 
                x->column(x, :varlmmobs) |> 
                x->VarLmmModel(x, meannames = meanname,
                renames = rename, wsvarnames = wsvarname)
    else
        varlmm = JuliaDB.groupby(varlmmobs, table(datatable), idvar) |> 
                x->column(x, :varlmmobs) |> 
                x->VarLmmModel(x, meannames = meanname,
                renames = rename, wsvarnames = wsvarname)
    end

    return varlmm
end

"""
    rvarlmm(Xs::Array{Matrix}, Zs::Array{Matrix},
    Ws::Array{Matrix}, β::Vector, τ::Vector;
    respdist = MvNormal, Σγ=[], Σγω=[])


Generate a simulated response from the VarLMM model based on:
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

Generate a simulated response from the VarLMM model based on a datatable. 
Note: **the datatable MUST be ordered by grouping variable for it to generate in the correct order.
This can be checked via `datatable == sort(datatable, idvar)`. The response is based on:

- `meanformula`: represents the formula for the mean fixed effects `β` (variables in X matrix)
- `reformula`: represents the formula for the mean random effects γ (variables in Z matrix)
- `wsvarformula`: represents the formula for the within-subject variance fixed effects τ (variables in W matrix)
- `idvar`: the id variable for groupings.
- `datatable`: the data table holding all of the data for the model. For this function it **must be in order**.
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
    τ::Vector{T}; respdist = MvNormal, Σγ=[], Σγω=[]
    ) where T <: BlasReal

    @assert length(Xs) == length(Zs) == length(Ws) "Number of provided X, Z, and W matrices do not match"
    isempty(Σγ) && isempty(Σγω) && error("Neither the covariance matrix for γ
    nor the covariance matrix for (γ, ω) have been specified. One must be.")

    y = []
    q = size(Zs[1], 2)
    
    # Get Cholesky Factor
    isempty(Σγω) ? Σγω = [Σγ zeros(q); zeros(1, q) 0.0] : nothing
    Lγω = cholesky(Symmetric(Σγω), check = false).L
    Lγ    = Lγω[1:q, 1:q]
    lγω   = Lγω[q + 1, 1:q]
    lω    = Lγω[q + 1, q + 1]

    γω = Vector{Float64}(undef, q + 1)
    z  = similar(γω)
    @inbounds for i in 1:length(Xs)
        push!(y, rvarlmm(Xs[i], Zs[i], Ws[i],  β, τ, Lγω, Lγ, lγω, γω, z, respdist))
    end
    return y
end

function rvarlmm!(meanformula::FormulaTerm, reformula::FormulaTerm, 
    wsvarformula::FormulaTerm, idvar::Union{Symbol, String},
    datatable, β::Vector{T}, τ::Vector{T}; respdist = MvNormal, Σγ=[], Σγω=[],
    respname::Symbol = :y
    ) where T <: BlasReal

    isempty(Σγ) && isempty(Σγω) && error("Neither the covariance matrix for γ
    nor the covariance matrix for (γ, ω) have been specified. One must be.")

    if typeof(idvar) <: String
        idvar = Symbol(idvar)
    end

    function rvarlmmob(f1, f2, f3, subdata,
        β, τ, Lγω, Lγ, lγω, γω, z, ydist)
        X = modelmatrix(meanformula, subdata)
        Z = modelmatrix(reformula, subdata)
        W = modelmatrix(wsvarformula, subdata)
        return rvarlmm(X, Z, W, β, τ, Lγω, Lγ, lγω, γω, z, ydist)
    end

    #apply df-wide schema
    meanformula = apply_schema(meanformula, schema(meanformula, datatable))
    reformula = apply_schema(reformula, schema(reformula, datatable))
    wsvarformula = apply_schema(wsvarformula, schema(wsvarformula, datatable))
    
    q = length(coefnames(reformula.rhs))
    # Get Cholesky Factor
    isempty(Σγω) ? Σγω = [Σγ zeros(q); zeros(1, q) 0.0] : nothing
    Lγω = cholesky(Symmetric(Σγω), check = false).L
    Lγ    = Lγω[1:q, 1:q]
    lγω   = Lγω[q + 1, 1:q]
    lω    = Lγω[q + 1, q + 1]

    γω = Vector{Float64}(undef, q + 1)
    z  = similar(γω)

    # Need sortperm of groupby var... 
    if typeof(datatable) <: IndexedTable
        y = JuliaDB.groupby(x -> rvarlmmob(meanformula, reformula, wsvarformula,
            x, β, τ, Lγω, Lγ, lγω, γω, z, respdist), datatable, idvar) |> 
            x -> column(x, 2) |> x -> vcat(x...) 
        transform(datatable, respname => y)
    else
        y = JuliaDB.groupby(x -> rvarlmmob(meanformula, reformula, wsvarformula,
            x, β, τ, Lγω, Lγ, lγω, γω, z, respdist), table(datatable), idvar) |> 
            x -> column(x, 2) |> x -> vcat(x...)
        datatable[!, respname] = y
    end
    return y
end

function rvarlmm(X, Z, W, β, τ, Lγω, Lγ, lγω, γω, z, ydist)
    q = size(Lγ, 1)
    mul!(γω, Lγω, Distributions.rand!(Normal(), z))
    # generate y
    μy = X * β + Z * γω[1:q]
    @views vy = exp.(W * τ .+ dot(γω[1:q], lγω) .+ γω[end])
    # TODO add more distributions
    return rand(ydist(μy, Diagonal(vy)))
end
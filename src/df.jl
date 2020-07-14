"""
    DataFrame(m::WSVarLmmModel)

Convert the data in `WSVarLmmModel` to a `DataFrame`.
"""
function DataFrame(m::WSVarLmmModel)
    p, q, l, n = m.p, m.q, m.l, m.nsum
    # preallocate arrays
    id = Vector{Int}(undef, n)
    y  = Vector{Float64}(undef, n)
    X  = Matrix{Float64}(undef, n, p)
    Z  = Matrix{Float64}(undef, n, q)
    W  = Matrix{Float64}(undef, n, l)
    addweight = !isempty(m.obswts)
    if addweight
        weights = Vector{Float64}(undef, n)
    end
    # gather data
    offset = 1
    for (i, vlmmobs) in enumerate(m.data)
        ni            = length(vlmmobs.y)
        rangei        = offset:(offset + ni - 1)
        id[rangei]   .= i
         y[rangei]    = vlmmobs.y
         X[rangei, :] = transpose(vlmmobs.Xt)
         Z[rangei, :] = transpose(vlmmobs.Zt)
         W[rangei, :] = transpose(vlmmobs.Wt)
         if addweight
            weights[rangei] .= m.obswts[i]
         end
        offset       += ni
    end
    df = hcat(DataFrame(id = id, y = y), 
        DataFrame(X, [Symbol("x$i") for i in 1:p]), 
        DataFrame(Z, [Symbol("z$i") for i in 1:q]), 
        DataFrame(W, [Symbol("w$i") for i in 1:l]))
    categorical!(df, :id)
    if addweight
        df[!, :obswts] = weights
    end
    df
end


"""
    WSVarLmmModel(meanformula::FormulaTerm, reformula::FormulaTerm, 
    wsvarformula::FormulaTerm, idvar::Union{String, Symbol}, datatable)

Constructor of `WSVarLmmModel` from a `DataFrame` or `IndexedTable`. 

# Positional arguments  
- `meanformula`: formula for the mean fixed effects β (variables in X matrix).  
- `reformula`: formula for the mean random effects γ (variables in Z matrix).  
- `wsvarformula`: formula for the within-subject variance effects τ (variables in W matrix). 
- `idvar`: id variable for groupings. 
- `datatable:` data table holding all of the data for the model. It can be a 
`DataFrame` or column-based table such as an `IndexedTable` from JuliaDB. 

# Keyword arguments
- `wtvar`: variable name corresponding to the observation weights in the datatable.

# Example
```
vlmm3 = WSVarLmmModel(@formula(y ~ 1 + x2 + x3 + x4 + x5),
    @formula(y ~ 1 + z2 + z3), @formula(y ~ 1 + w2 + w3 + w4 + w5), "id", df)
```
"""
function WSVarLmmModel(
    meanformula  :: FormulaTerm, 
    reformula    :: FormulaTerm, 
    wsvarformula :: FormulaTerm, 
    idvar        :: Union{String, Symbol}, 
    datatable;
    wtvar        :: Union{String, Symbol} = ""
    )
    idvar = Symbol(idvar)
    function varlmmobs(tab)
        tab = StatsModels.columntable(tab)
        tab, _ = StatsModels.missing_omit(tab, alltermformula)

        y, X = modelcols(meanformula, tab)
        Z    = modelmatrix(reformula, tab)
        W    = modelmatrix(wsvarformula, tab)
        return WSVarLmmObs(y, X, Z, W)
    end
    #for schema in missing values of y, otherwise it will dummy-encode.
    ydict = Dict(Symbol(meanformula.lhs) => ContinuousTerm)

    # apply df-wide schema
    meanformula  = apply_schema(meanformula, schema(meanformula, datatable, ydict))
    reformula    = apply_schema(reformula, schema(reformula, datatable, ydict))
    wsvarformula = apply_schema(wsvarformula, schema(wsvarformula, datatable, ydict))

    # collect all terms to perform missing_omit properly
    alltermformula = meanformula.lhs ~ sum(term.(union(terms(meanformula.rhs),
      terms(reformula.rhs), terms(wsvarformula.rhs))))
    alltermformula = apply_schema(alltermformula, schema(alltermformula, datatable, ydict))


    # variable names
    meanname  = StatsModels.coefnames(meanformula.rhs)
    meanname  = ["β$i: " for i in 1:length(meanname)] .* meanname
    rename    = StatsModels.coefnames(reformula.rhs)
    rename    = ["γ$i: " for i in 1:length(rename)] .* rename
    wsvarname = StatsModels.coefnames(wsvarformula.rhs)
    wsvarname = ["τ$i: " for i in 1:length(wsvarname)] .* wsvarname
    if isempty(string(wtvar)) 
        wts = []
    else
        cnames = colnames(table(datatable))
        wtvar  = Symbol(wtvar)
        wtvar in cnames || 
            error("weight variable $wtvar not in datatable $datatable")
        wts = JuliaDB.groupby((wts = wtvar => first, ),
                table(datatable), idvar) |> 
                x -> JuliaDB.select(x, :wts)
    end
    # now form observations 
    if typeof(datatable) <: IndexedTable
        varlmm = JuliaDB.groupby(varlmmobs, datatable, idvar) |> 
                x -> column(x, :varlmmobs) |> 
                x -> WSVarLmmModel(x, meannames = meanname,
                renames = rename, wsvarnames = wsvarname, obswts = wts)
    else
        varlmm = JuliaDB.groupby(varlmmobs, table(datatable), idvar) |> 
                x -> column(x, :varlmmobs) |> 
                x -> WSVarLmmModel(x, meannames = meanname,
                renames = rename, wsvarnames = wsvarname, obswts = wts)
    end

    return varlmm
end

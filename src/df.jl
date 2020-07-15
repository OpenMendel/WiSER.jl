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
- `tbl:` data table holding all of the data for the model. It can be a 
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
    tbl          :: Union{IndexedTable, DataFrame};
    wtvar        :: Union{String, Symbol} = ""
    )
    idvar = Symbol(idvar)
    function varlmmobs(tab)
        y, X = modelcols(meanformula, tab)
        Z    = modelmatrix(reformula, tab)
        W    = modelmatrix(wsvarformula, tab)
        return WSVarLmmObs(y, X, Z, W)
    end
    isdf = typeof(tbl) <: DataFrame
    datatable = isdf ? deepcopy(tbl) : DataFrame(tbl) #don't want to modify original df
    # collect all terms to perform dropping properly
    alltermformula = meanformula.lhs ~ sum(term.(union(terms(meanformula.rhs),
      terms(reformula.rhs), terms(wsvarformula.rhs))))

    # accumulate terms to keep in sub-df
    dropsmissing = Symbol.(filter!(x -> x != term(1), terms(alltermformula)))
    dropsmissing = isempty(string(wtvar)) ? [dropsmissing; idvar] :
         [dropsmissing; Symbol.(wtvar); idvar]

    # subset df and drop missing entries of used variables
    datatable = datatable[!, dropsmissing]
    dropmissing!(datatable, dropsmissing)

    # apply df-wide schema
    meanformula  = apply_schema(meanformula, schema(meanformula, datatable))#, ydict))
    reformula    = apply_schema(reformula, schema(reformula, datatable))#, ydict))
    wsvarformula = apply_schema(wsvarformula, schema(wsvarformula, datatable))#, ydict))

    # variable names
    meannames = StatsModels.coefnames(meanformula.rhs)
    # either array{Names} or string of one variable
    meannames = typeof(meannames) <: Array ? ["β$i: " for i in 1:length(meannames)] .*
        meannames : ["β1: " * meannames]
    renames = StatsModels.coefnames(reformula.rhs)
    renames = typeof(renames) <: Array ? ["γ$i: " for i in 1:length(renames)] .*
        renames : ["γ1: " * renames]
    wsvarnames = StatsModels.coefnames(wsvarformula.rhs)
    wsvarnames = typeof(wsvarnames) <: Array ?  ["τ$i: " for i in 1:length(wsvarnames)] .*
        wsvarnames : ["τ1: " * wsvarnames]

    if isempty(string(wtvar)) 
        wts = []
    else
        cnames = Symbol.(names(datatable))
        wtvar  = Symbol(wtvar)
        wtvar in cnames || 
            error("weight variable $wtvar not in datatable $datatable")
        wts = combine(DataFrames.groupby(datatable, idvar), wtvar => first)[!, 2]
    end
    obsvec = combine(varlmmobs, DataFrames.groupby(datatable, idvar))[!, 2]
    varlmm = WSVarLmmModel(obsvec, meannames = meannames, renames = renames,
    wsvarnames = wsvarnames, obswts = wts)
    datatable = nothing #get rid of memory when gc is called
    return varlmm
end

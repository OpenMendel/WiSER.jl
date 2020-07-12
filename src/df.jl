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
    addweight = !isempty(m.obswts)
    if addweight
        weights = Vector{Float64}(undef, n)
    end
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
         if addweight
            weights[rangei] .= m.obswts[i]
         end
        offset += ni
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
    VarLmmModel(meanformula::FormulaTerm, reformula::FormulaTerm, 
    wsvarformula::FormulaTerm, idvar::Union{String, Symbol}, datatable)

Constructor of `VarLmmModel` from a `DataFrame` or `IndexedTable`. `meanformula` represents the formula for
the mean fixed effects β (variables in X matrix), `reformula` represents the formula for 
the mean random effects γ (variables in Z matrix), `wsvarformula` represents the formula 
for the within-subject variance fixed effects τ (variables in W matrix). `idvar` is the
id variable for groupings. `data` is the data table holding all of the data for the model.
It can be a `DataFrame` or column-based table such as an `IndexedTable` from JuliaDB. 

Example:
vlmm3 = VarLmmModel(@formula(y ~ 1 + x2 + x3 + x4 + x5),
    @formula(y ~ 1 + z2 + z3), @formula(y ~ 1 + w2 + w3 + w4 + w5), "id", df)
"""
function VarLmmModel(meanformula::FormulaTerm, reformula::FormulaTerm, 
    wsvarformula::FormulaTerm, idvar::Union{String, Symbol}, datatable;
    wtvar::Union{String, Symbol} = "")

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
    
    meanname = StatsModels.coefnames(meanformula.rhs)
    meanname = ["β$i: " for i in 1:length(meanname)] .* meanname
    rename = StatsModels.coefnames(reformula.rhs)
    rename = ["γ$i: " for i in 1:length(rename)] .* rename
    wsvarname = StatsModels.coefnames(wsvarformula.rhs)
    wsvarname = ["τ$i: " for i in 1:length(wsvarname)] .* wsvarname

    if isempty(string(wtvar)) 
        wts = []
    else
        cnames = colnames(table(datatable))
        wtvar = Symbol(wtvar)
        wtvar in cnames || 
            error("weight variable $wtvar not in datatable $datatable")
        wts = JuliaDB.groupby((wts = wtvar => first, ),
                table(datatable), idvar) |> 
                x -> JuliaDB.select(x, :wts)
    end

    #now form observations 
    if typeof(datatable) <: IndexedTable
        varlmm = JuliaDB.groupby(varlmmobs, datatable, idvar) |> 
                x->column(x, :varlmmobs) |> 
                x->VarLmmModel(x, meannames = meanname,
                renames = rename, wsvarnames = wsvarname, obswts = wts)
    else
        varlmm = JuliaDB.groupby(varlmmobs, table(datatable), idvar) |> 
                x->column(x, :varlmmobs) |> 
                x->VarLmmModel(x, meannames = meanname,
                renames = rename, wsvarnames = wsvarname, obswts = wts)
    end

    return varlmm
end


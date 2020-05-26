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
    VarLmmModel2(meanformula::FormulaTerm, reformula::FormulaTerm, 
    wsvarformula::FormulaTerm, idvar::Union{String, Symbol}, df)

Constructor of `VarLmmModel` from a `DataFrame`. `meanformula` represents the formula for
the mean fixed effects β (variables in X matrix), `reformula` represents the formula for 
the mean random effects γ (variables in Z matrix), `wsvarformula` represents the formula 
for the within-subject variance fixed effects τ (variables in W matrix). `idvar` is the
id variable for groupings. df is the dataframe holding all of the data for the model. 

Example:
vlmm3 = VarLmmModel(@formula(y ~ x1 + x2 + x3 + x4 + x5),
    @formula(y ~ z1 + z2 + z3), @formula(y ~ w1 + w2 + w3 + w4 + w5), "id", df)
"""
function VarLmmModel(meanformula::FormulaTerm, reformula::FormulaTerm, 
    wsvarformula::FormulaTerm, idvar::Union{String, Symbol}, df)

    if typeof(idvar) <: String
        idvar = Symbol(idvar)
    end
    ids = unique(df[!, idvar])
    m = length(ids)

    function varlmmobs(df2)
        y, X = modelcols(meanformula, df2)
        Z = modelmatrix(reformula, df2)
        W = modelmatrix(wsvarformula, df2)
        return VarLmmObs(y, X, Z, W)
    end

    #apply df-wide schema
    meanformula = apply_schema(meanformula, schema(meanformula, df))
    reformula = apply_schema(reformula, schema(reformula, df))
    wsvarformula = apply_schema(wsvarformula, schema(wsvarformula, df))

    #now form observations 
    obsvec = combine(varlmmobs, groupby(df, :id))[!, 2]

    VarLmmModel(obsvec)
end
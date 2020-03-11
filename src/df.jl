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
         X[rangei, :] = vlmmobs.X
         Z[rangei, :] = vlmmobs.Z
         W[rangei, :] = vlmmobs.W
        offset += ni
    end
    df = hcat(DataFrame(id = id, y = y), 
        DataFrame(X, [Symbol("x$i") for i in 1:p]), 
        DataFrame(Z, [Symbol("z$i") for i in 1:q]), 
        DataFrame(W, [Symbol("w$i") for i in 1:l]))
    categorical!(df, :id)
end

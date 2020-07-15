export ◺, commutation, CopyMatrix, kron_axpy!, kr_axpy!, kr, mul!, vech,
    Ct_At_kron_A_KC, Ct_At_kron_A_KC!, 
    Ct_A_kron_B_C, Ct_A_kron_B_C!,
    Ct_A_kr_B, Ct_A_kr_B!
import LinearAlgebra: mul!

"""
    ◺(n::Integer)

Triangular number `n * (n + 1) / 2`.
"""
@inline ◺(n::Integer) = (n * (n + 1)) >> 1

"""
    kron_axpy!(A, X, Y)

Overwrite `Y` with `A ⊗ X + Y`. Same as `Y += kron(A, X)` but
more memory efficient.
"""
function kron_axpy!(
    A::AbstractVecOrMat{T},
    X::AbstractVecOrMat{T},
    Y::AbstractVecOrMat{T}
) where T <: Real
    m, n = size(A, 1), size(A, 2)
    p, q = size(X, 1), size(X, 2)
    @assert size(Y, 1) == m * p
    @assert size(Y, 2) == n * q
    @inbounds for j in 1:n
        coffset = (j - 1) * q
        for i in 1:m
            a = A[i, j]
            roffset = (i - 1) * p            
            for l in 1:q
                r = roffset + 1
                c = coffset + l
                for k in 1:p                
                    Y[r, c] += a * X[k, l]
                    r += 1
                end
            end
        end
    end
    Y
end

"""
    kr_axpy!(A, X, Y)

Overwrite `Y` with `A ⊙ X + Y`, where `⊙` stands for the Khatri-Rao (columnwise 
Kronecker) product. `A` and `X` need to have same number of columns.
"""
function kr_axpy!(
    A::AbstractVecOrMat{T},
    X::AbstractVecOrMat{T},
    Y::AbstractVecOrMat{T}
) where T <: Real
    @assert size(A, 2) == size(X, 2) == size(Y, 2)
    m, n, p = size(A, 1), size(A, 2), size(X, 1)
    @inbounds for j in 1:n
        r = 1        
        for i in 1:m
            aij = A[i, j]
            for k in 1:p
                Y[r, j] += aij * X[k, j]
                r += 1
            end
        end
    end
    Y
end

kr(A::AbstractVecOrMat{T}, X::AbstractVecOrMat{T}) where T <: Real = 
    kr_axpy!(A, X, zeros(T, size(A, 1) * size(X, 1), size(A, 2)))

struct CopyMatrix <: AbstractMatrix{Int}
    n::Int
end

Base.size(C::CopyMatrix) = (abs2(C.n), (C.n * (C.n + 1)) >> 1)

Base.IndexStyle(::Type{<:CopyMatrix}) = IndexCartesian()

function Base.getindex(C::CopyMatrix, i::Int, j::Int)
    r, c = CartesianIndices((1:C.n, 1:C.n))[i].I
    if r ≥ c && j == (c - 1) * C.n - ((c - 2) * (c - 1)) >> 1 + r - c + 1
        return 1
    else
        return 0
    end
end

"""
    mul!(result, A, C::CopyMatrix)

Right-multiplying a matrix `A` by a copying matrix is equivalent to keeping 
the columns of `A` corresponding to the lower triangular indices.
"""
function LinearAlgebra.mul!(
    result :: AbstractVecOrMat,
    A      :: AbstractVecOrMat,
    C      :: CopyMatrix
)
    n = isqrt(size(A, 2))
    m = size(A, 1)
    @assert size(result, 1) == m
    @assert size(result, 2) == (n * (n + 1)) >> 1
    ac, rc = 0, 0
    @inbounds for j in 1:n, i in 1:n
        ac += 1
        i < j && continue
        rc += 1
        for k in 1:m
            result[k, rc] = A[k, ac]
        end
    end
    result
end

"""
    mul!(result, Ct::Transpose{Int, CopyMatrix}, A)

Left-multiplying a matrix `A` by transpose of a copying matrix is equivalent to 
keeping the rows of `A` corresponding to the lower triangular indices.
"""
LinearAlgebra.mul!(
    result :: AbstractVecOrMat,
    Ct     :: Transpose{Int, CopyMatrix},
    A      :: AbstractVecOrMat
) = mul!(transpose(result), transpose(A), Ct.parent)

"""
    vech!(v::AbstractVector, A::AbstractVecOrMat)

Overwrite vector `v` by the entries from lower triangular part of `A`. 
"""
function vech!(v::AbstractVector, A::AbstractVecOrMat)
    m, n = size(A, 1), size(A, 2)
    idx = 1
    @inbounds for j in 1:n, i in j:m
        v[idx] = A[i, j]
        idx += 1
    end
    v
end

function commutation(m::Integer, n::Integer)
    K = zeros(Int, m * n, m * n)
    colK = 1
    @inbounds for j in 1:n, i in 1:m
        rowK          = n * (i - 1) + j
        K[rowK, colK] = 1
        colK         += 1
    end
    K
end

"""
    vech(A::AbstractVecOrMat) -> AbstractVector

Return the entries from lower triangular part of `A` as a vector.
"""
function vech(A::AbstractVecOrMat)
    m, n = size(A, 1), size(A, 2)
    vech!(similar(A, n * m - (n * (n - 1)) >> 1), A)
end

"""
    Ct_At_kron_A_KC!(H, A, B)

Overwrite `H` by `H + C'(A'⊗A)KC`, where `K` is the commutation matrix and 
`C` is the copying matrix.
"""
function Ct_At_kron_A_KC!(H::AbstractMatrix, A::AbstractMatrix)
    q = size(A, 1)
    @assert size(A, 2) == q
    @assert size(H, 1) == size(H, 2) == (q * (q + 1)) >> 1
    j = 1
    @inbounds for w in 1:q, s in w:q
        i = 1
        for r in 1:q, v in r:q
            H[i, j] += A[s, r] * A[v, w]
            i += 1
        end
        j += 1
    end
    H
end

function Ct_At_kron_A_KC(A)
    n◺ = ◺(size(A, 1))
    H = zeros(eltype(A), n◺, n◺)
    Ct_At_kron_A_KC!(H, A)
end

"""
    Ct_A_kron_B_C!(H, A, B)

Overwrite `H` by `H + C'(A⊗B)C`, where `C` is the copying matrix.
"""
function Ct_A_kron_B_C!(
    H::AbstractMatrix, 
    A::AbstractMatrix,
    B::AbstractMatrix,
    )
    q = size(A, 1)
    @assert size(A, 2) == size(B, 1) == size(B, 2) == q
    j = 1
    @inbounds for s in 1:q, w in s:q
        i = 1
        for r in 1:q, v in r:q
            H[i, j] += A[r, s] * B[v, w]
            i += 1
        end
        j += 1
    end
    H
end

function Ct_A_kron_B_C(A, B)
    n◺ = ◺(size(A, 1))
    H = zeros(eltype(A), n◺, n◺)
    Ct_A_kron_B_C!(H, A, B)
end

"""
    Ct_A_kr_B!(H, A, B)

Overwrite `H` by `H + C'(A⊙B)`, where `C` is the copying matrix and `⊙` is the 
Khatri-Rao (column-wise Kronecker) product.
"""
function Ct_A_kr_B!(H, A, B)
    @assert size(A) == size(B)
    (q, n) = size(A)    
    @inbounds for c in 1:n
        r = 1
        for ia in 1:q
            a = A[ia, c]
            for ib in ia:q
                H[r, c] += a * B[ib, c]
                r += 1
            end
        end
    end
    H
end

function Ct_A_kr_B(A, B)
    @assert size(A) == size(B)
    (q, n) = size(A)
    H = zeros(eltype(A), ◺(q), n)
    Ct_A_kr_B!(H, A, B)
end

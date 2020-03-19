export CopyMatrix, kron_axpy!, kr_axpy!, kr, mul!, vech
import LinearAlgebra: mul!

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
    @assert size(A, 2) == size(X, 2)
    m, n, p = size(A, 1), size(A, 2), size(X, 1)
    @inbounds for j in 1:n
        r = 1        
        for i in 1:m
            aij = A[i, j]
            for k in 1:p
                Y[r, j] = aij * X[k, j]
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

"""
    vech(A::AbstractVecOrMat) -> AbstractVector

Return the entries from lower triangular part of `A` as a vector.
"""
function vech(A::AbstractVecOrMat)
    m, n = size(A, 1), size(A, 2)
    vech!(similar(A, n * m - (n * (n - 1)) >> 1), A)
end

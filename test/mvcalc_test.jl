using BenchmarkTools, LinearAlgebra, Random, Test, VarLMM

@testset "kron_axpy!" begin
Random.seed!(123)
m, n = 100, 50
p, q = 30, 20
A = randn(m, n)
X = randn(p, q)

Y1 = zeros(m * p, n * q)
kron_axpy!(A, X, Y1)
@test Y1 == kron(A, X)
bm = @benchmark kron_axpy!($A, $X, $Y1) setup=(fill!($Y1, 0))
display(bm); println()
@test allocs(bm) == 0

Y2 = zeros(n * p, m * q)
kron_axpy!(transpose(A), X, Y2)
@test Y2 == kron(transpose(A), X)
# bm = @benchmark kron_axpy!($(transpose(A)), $X, $Y2) setup=(fill!($Y2, 0))
bm = @benchmark kron_axpy!($(transpose(A)), $X, $Y2) setup=(fill!($Y2, 0))
display(bm); println()
@test allocs(bm) == 0

Y3 = zeros(m * q, n * p)
kron_axpy!(A, transpose(X), Y3)
@test Y3 == kron(A, transpose(X))
bm = @benchmark kron_axpy!($A, $(transpose(X)), $Y3) setup=(fill!($Y3, 0))
display(bm); println()
@test allocs(bm) == 0

Y4 = zeros(n * q, m * p)
kron_axpy!(transpose(A), transpose(X), Y4)
@test Y4 == kron(transpose(A), transpose(X))
bm = @benchmark kron_axpy!($(transpose(A)), $(transpose(X)), $Y4) setup=(fill!($Y4, 0))
display(bm); println()
@test allocs(bm) == 0
end

@testset "kr_axpy!" begin
Random.seed!(123)
m, n, p = 100, 50, 30
A = randn(m, n)
X = randn(p, n)

Y1 = zeros(m * p, n)
kr_axpy!(A, X, Y1)
@test Y1 == reshape([A[i1, j] * X[i2, j] for i2 in 1:p, i1 in 1:m, j in 1:n], m * p, n)
bm = @benchmark kr_axpy!($A, $X, $Y1) setup=(fill!($Y1, 0))
display(bm); println()
@test allocs(bm) == 0
end

@testset "CopyMatrix" begin
Random.seed!(123)
n = 50
C = CopyMatrix(n)

@test size(C) == (n * n, (n * (n + 1)) >> 1)
# display(C); println()
# display(Matrix{Float64}(C)); println()
M = randn(n, n)
@test vec(LowerTriangular(M)) == C * vech(LowerTriangular(M))
@test vec(M)' * C == vech(M)'
@test C' * vec(M) == vech(M)

result1 = similar(vec(M)' * C)
bm = @benchmark mul!($result1, $(transpose(vec(M))), $C)
display(bm); println()
@test allocs(bm) == 0

result2 = similar(C' * vec(M))
bm = @benchmark mul!($result2, $(transpose(C)), $(vec(M)))
display(bm); println()
@test_skip allocs(bm) == 0
end

@testset "Ct_At_kron_A_KC" begin
Random.seed!(123)
q = 5
q◺ = VarLMM.◺(q)
A = randn(q, q); A = A'A * LowerTriangular(randn(q, q))
H = Ct_At_kron_A_KC(A)
Cq  = CopyMatrix(q)
Kqq = commutation(q, q)
@test size(H) == (q◺, q◺)
@test issymmetric(H)
@test H == Cq' * kron(A', A) * Kqq * Cq
bm = @benchmark Ct_At_kron_A_KC!($H, $A) setup=(fill!($H, 0))
display(bm); println()
@test allocs(bm) == 0
end

@testset "Ct_A_kron_B_C" begin
Random.seed!(123)
q = 5
q◺ = VarLMM.◺(q)
A = randn(q, q); A = A'A
B = randn(q, q); B = B'B
H = Ct_A_kron_B_C(A, B)
Cq  = CopyMatrix(q)
@test size(H) == (q◺, q◺)
@test issymmetric(H)
@test all(eigvals(Symmetric(H)) .≥ 0)
@test H == Cq' * kron(A, B) * Cq
bm = @benchmark Ct_A_kron_B_C!($H, $A, $B) setup=(fill!($H, 0))
display(bm); println()
@test allocs(bm) == 0
end

@testset "Ct_A_kr_B" begin
Random.seed!(123)
q, n = 5, 10
q◺ = ◺(q)
A = randn(q, n)
B = randn(q, n)
H = Ct_A_kr_B(A, B)
Cq  = CopyMatrix(q)
@test size(H) == (q◺, n)
@test H == Cq' * kr(A, B)
bm = @benchmark Ct_A_kr_B!($H, $A, $B) setup=(fill!($H, 0))
display(bm); println()
@test allocs(bm) == 0
end

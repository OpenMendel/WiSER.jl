# Test for df.jl functions
using Test, VarLMM, JuliaDB, DataFrames, Random, LinearAlgebra

Random.seed!(123)
t = table((id = [1; 1; 2; 3; 3; 3; 4], y = randn(7),
x1 = ones(7), x2 = randn(7), x3 = randn(7), z1 = ones(7),
z2 = randn(7), w1 = ones(7), w2 = randn(7), w3 = randn(7)))
df = DataFrame(t)

f1 = @formula(y ~ 1 + x2 + x3)
f2 = @formula(y ~ 1 + z2)
f3 = @formula(y ~ 1 + w2 + w3)

vlma = VarLmmModel(f1, f2, f3, :id, t)
dfa = DataFrame(vlma)
vlmb = VarLmmModel(f1, f2, f3, :id, df)
dfb = DataFrame(vlmb)

β = zeros(3)
τ = zeros(3)
Σγ = Matrix{Float64}(I, 2, 2)

y = [randn(2) for i in 1:3]
Xs = [randn(2, 3) for i in 1:3]
Ws = [randn(2, 3) for i in 1:3]
Zs = [randn(2, 2) for i in 1:3]

@testset "VarLmmModel Constructor & Simulating" begin
    @test dfa == dfb == df
    @test rvarlmm!(f1, f2, f3, :id, df, β, τ;
        Σγ = Σγ, respname = :response)[1] ≈ 2.1830342035877
    @test rvarlmm!(f1, f2, f3, :id, t, β, τ;
         Σγ = Σγ, respname = :response)[1] ≈ -1.49863327329357
    @test rvarlmm(Xs, Zs, Ws, β, τ; Σγ = Σγ)[1][1] ≈ -0.992563667923
    @test :y in names(df)
    @test :y in colnames(t)
end


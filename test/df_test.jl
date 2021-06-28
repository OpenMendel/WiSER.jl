module DFTest
# Test for df.jl and rand.jl functions
using DataFrames, Tables, LinearAlgebra, Random, Test, WiSER

Random.seed!(123)
rng = MersenneTwister(123)
t = columntable((
    id     = [1; 1; 2; 3; 3; 3; 4], 
    y      = [missing; randn(rng, 6)],
    x1     = ones(7), 
    x2     = randn(rng, 7), 
    x3     = randn(rng, 7), 
    z1     = ones(7),
    z2     = randn(rng, 7), 
    w1     = ones(7), 
    w2     = randn(rng, 7), 
    w3     = randn(rng, 7),
    obswts = [2.0; 2.0; 1.0; 0.5; 0.5; 0.5; 1.0]))
df = DataFrame(t)

f1 = @formula(y ~ x1 + x2 + x3)
f2 = @formula(y ~ z1 + z2)
f3 = @formula(y ~ w1 + w2 + w3)

vlma = WSVarLmmModel(f1, f2, f3, :id, t; wtvar = :obswts)
dfa  = DataFrame(vlma)
vlmb = WSVarLmmModel(f1, f2, f3, :id, df; wtvar = :obswts)
dfb  = DataFrame(vlmb)

β  = zeros(3)
τ  = zeros(3)
Σγ = Matrix{Float64}(I, 2, 2)

y  = [randn(rng, 2) for i in 1:3]
Xs = [randn(rng, 2, 3) for i in 1:3]
Ws = [randn(rng, 2, 3) for i in 1:3]
Zs = [randn(rng, 2, 2) for i in 1:3]

@testset "WSVarLmmModel Constructor" begin
    @test dfa == dfb == DataFrames.dropmissing(df)
end

@testset "Simulating Response" begin
    @test rvarlmm!(rng, f1, f2, f3, :id, df, β, τ;
        Σγ = Σγ, respname = :response)[1, :response] ≈ -3.7193661312903674
    @test rvarlmm(rng, Xs, Zs, Ws, β, τ; Σγ = Σγ)[1][1] ≈ 2.3445518865974098
    @test "response" in names(df)

    vlma.β .= 0
    vlma.τ .= 0
    vlma.Lγ .= [1. 0; 0 1.]
    ytest = vlma.data[1].y[1]
    WiSER.rand!(rng, vlma) 
    @test vlma.data[1].y[1] != ytest
    ytest = vlma.data[1].y[1]
    WiSER.rand!(rng, vlma; respdist = MvNormal) 
    @test vlma.data[1].y[1] != ytest
    ytest = vlma.data[1].y[1]
    WiSER.rand!(rng, vlma; respdist = MvTDist, df = 10) 
    @test vlma.data[1].y[1] != ytest
    ytest = vlma.data[1].y[1]
    vlma.β[1] = 30.
    WiSER.rand!(rng, vlma; respdist = Gamma) 
    @test vlma.data[1].y[1] != ytest
    ytest = vlma.data[1].y[1]
    WiSER.rand!(rng, vlma; respdist = InverseGaussian) 
    @test vlma.data[1].y[1] != ytest
    ytest = vlma.data[1].y[1]
    WiSER.rand!(rng, vlma; respdist = InverseGamma) 
    @test vlma.data[1].y[1] != ytest
end

end
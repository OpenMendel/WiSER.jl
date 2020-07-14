module DFTest
# Test for df.jl and varlmm_rand.jl functions
using DataFrames, JuliaDB, LinearAlgebra, Random, Test, WiSER

Random.seed!(123)
t = table((
    id     = [1; 1; 2; 3; 3; 3; 4], 
    y      = [missing; randn(6)],
    x1     = ones(7), 
    x2     = randn(7), 
    x3     = randn(7), 
    z1     = ones(7),
    z2     = randn(7), 
    w1     = ones(7), 
    w2     = randn(7), 
    w3     = randn(7),
    obswts = [2.0; 2.0; 1.0; 0.5; 0.5; 0.5; 1.0]))
df = DataFrame(t)

f1 = @formula(y ~ 1 + x2 + x3)
f2 = @formula(y ~ 1 + z2)
f3 = @formula(y ~ 1 + w2 + w3)

vlma = WSVarLmmModel(f1, f2, f3, :id, t; wtvar = :obswts)
dfa  = DataFrame(vlma)
vlmb = WSVarLmmModel(f1, f2, f3, :id, df; wtvar = :obswts)
dfb  = DataFrame(vlmb)

β  = zeros(3)
τ  = zeros(3)
Σγ = Matrix{Float64}(I, 2, 2)

y  = [randn(2) for i in 1:3]
Xs = [randn(2, 3) for i in 1:3]
Ws = [randn(2, 3) for i in 1:3]
Zs = [randn(2, 2) for i in 1:3]

@testset "WSVarLmmModel Constructor" begin
    @test dfa == dfb == DataFrames.dropmissing(df)
end

@testset "Simulating Response" begin
    @test rvarlmm!(f1, f2, f3, :id, df, β, τ;
        Σγ = Σγ, respname = :response)[1, :response] ≈ -0.356588825151
    global t
    t = rvarlmm!(f1, f2, f3, :id, t, β, τ;
        Σγ = Σγ, respname = :response)
    @test t[1].response ≈ -0.2840619552
    @test rvarlmm(Xs, Zs, Ws, β, τ; Σγ = Σγ)[1][1] ≈ 0.4852541787188
    @test string(:response) in names(df)
    @test :response in colnames(t)

    vlma.β .= 0
    vlma.τ .= 0
    vlma.Lγ .= [1. 0; 0 1.]
    ytest = vlma.data[1].y[1]
    WiSER.rand!(vlma) 
    @test vlma.data[1].y[1] != ytest
    ytest = vlma.data[1].y[1]
    WiSER.rand!(vlma; respdist = MvNormal) 
    @test vlma.data[1].y[1] != ytest
    ytest = vlma.data[1].y[1]
    WiSER.rand!(vlma; respdist = MvTDist, df = 10) 
    @test vlma.data[1].y[1] != ytest
    ytest = vlma.data[1].y[1]
    vlma.β[1] = 30.
    WiSER.rand!(vlma; respdist = Gamma) 
    @test vlma.data[1].y[1] != ytest
    ytest = vlma.data[1].y[1]
    WiSER.rand!(vlma; respdist = InverseGaussian) 
    @test vlma.data[1].y[1] != ytest
    ytest = vlma.data[1].y[1]
    WiSER.rand!(vlma; respdist = InverseGamma) 
    @test vlma.data[1].y[1] != ytest
end

end
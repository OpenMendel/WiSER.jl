using Distributions, LinearAlgebra, Random, Test, VarLMM

@testset "VarLmmModel" begin
Random.seed!(123)
# dimensions
m  = 100  # number of individuals
ns = rand(8:8, m) # numbers of observations per individual
p  = 5    # number of fixed effects, including intercept
q  = 2    # number of random effects, including intercept
l  = 3    # number of WS variance covariates, including intercept
obsvec = Vector{VarLmmObs{Float64}}(undef, m)
# true parameter values
βtrue = [0.1; 6.5; -3.5; 1.0; 5]
τtrue = [-1.5; 1.5; -0.5]
Σγ    = [2.0 0.00; 0.00 1.2]
δγω   = [0.0; 0.0] # restrictive model
σω    = [0.0]      # restrictive model
Σγω   = [Σγ δγω; δγω' σω]
Lγω   = cholesky(Symmetric(Σγω), check = false).L
# data generation
γω = Vector{Float64}(undef, q + 1)
z  = similar(γω) # hold vector of iid std normal
for i in 1:m
    # first column intercept, remaining entries iid std normal
    X = Matrix{Float64}(undef, ns[i], p)
    X[:, 1] .= 1
    rand!(Normal(), X[:, 2:end])
    # first column intercept, remaining entries iid std normal
    Z = Matrix{Float64}(undef, ns[i], q)
    Z[:, 1] .= 1
    rand!(Normal(), Z[:, 2:end])
    # first column intercept, remaining entries iid std normal
    W = Matrix{Float64}(undef, ns[i], l)
    W[:, 1] .= 1
    rand!(Normal(), W[:, 2:end])
    # generate random effects
    mul!(γω, Lγω, rand!(Normal(), z))
    # generate y
    μy = X * βtrue + Z * γω[1:end-1]
    @views vy = exp.(W * τtrue .+ dot(γω[1:end-1], Lγω[end, 1:end-1]) .+ γω[end])
    y = rand(MvNormal(μy, vy))
    # form a VarLmmObs instance
    obsvec[i] = VarLmmObs(y, X, Z, W)
end
@test length(obsvec) == m
# form VarLmmModel
vlmm = VarLmmModel(obsvec)
end

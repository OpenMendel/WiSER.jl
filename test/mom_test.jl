using BenchmarkTools, Distributions, InteractiveUtils, KNITRO
using LinearAlgebra, Random, Test, VarLMM

@info "generate data"
Random.seed!(123)
# dimensions
m  = 6000 # number of individuals
ns = rand(8:8, m) # numbers of observations per individual
p  = 5    # number of fixed effects, including intercept
q  = 2    # number of random effects, including intercept
l  = 3    # number of WS variance covariates, including intercept
obsvec = Vector{VarLmmObs{Float64}}(undef, m)
# true parameter values
βtrue = [0.1; 6.5; -3.5; 1.0; 5]
τtrue = [-1.5; 1.5; -0.5]
Σγ    = [2.0 0.0; 0.0 1.2]
δγω   = [0.0; 0.0] # restrictive model
σω    = [0.0]      # restrictive model
Σγω   = [Σγ δγω; δγω' σω]
Lγω   = cholesky(Symmetric(Σγω), check = false).L
@test all(Lγω[end, :] .== 0)
# generate data
γω = Vector{Float64}(undef, q + 1)
z  = similar(γω) # hold vector of iid std normal
for i in 1:m
    # first column intercept, remaining entries iid std normal
    X = Matrix{Float64}(undef, ns[i], p)
    X[:, 1] .= 1
    @views Distributions.rand!(Normal(), X[:, 2:p])
    # first column intercept, remaining entries iid std normal
    Z = Matrix{Float64}(undef, ns[i], q)
    Z[:, 1] .= 1
    @views Distributions.rand!(Normal(), Z[:, 2:q])
    # first column intercept, remaining entries iid std normal
    W = Matrix{Float64}(undef, ns[i], l)
    W[:, 1] .= 1
    @views Distributions.rand!(Normal(), W[:, 2:l])
    # generate random effects: γω = Lγω * z
    mul!(γω, Lγω, Distributions.rand!(Normal(), z))
    @test γω[end] == 0
    # generate y
    μy = X * βtrue + Z * γω[1:q]
    @views vy = exp.(W * τtrue .+ dot(γω[1:q], Lγω[end, 1:q]) .+ γω[end])
    y = rand(MvNormal(μy, Diagonal(vy)))
    # form a VarLmmObs instance
    obsvec[i] = VarLmmObs(y, X, Z, W)
    # debug
    # display(X); println()
    # display(Z); println()
    # display(W); println()
    # display(y); println()
end
# form VarLmmModel
vlmm = VarLmmModel(obsvec)

@testset "mom_obj!" begin
# set parameter values to be the truth
@show copy!(vlmm.β, βtrue)
@show copy!(vlmm.τ, τtrue)
@show vlmm.Lγ   .= Lγω[1:end-1, 1:end-1]
@show vlmm.lγω  .= Lγω[    end, 1:end-1]
@show vlmm.lω[1] = Lγω[    end,     end]
# evaluate objective (at truth)
mgfγω = mgf_γω(vlmm)
@test mgfγω == 1 # restrictive model
@info "obj/grad at true parameter values"
@show mom_obj!(vlmm, true)
@show vlmm.∇β
@show vlmm.∇τ
@show vlmm.∇Lγ
@show vlmm.∇lγω
@show vlmm.∇lω
@info "type stability"
#@code_warntype mom_obj!(vlmm.data[1], vlmm.β, vlmm.τ, vlmm.Lγ, vlmm.lγω, vlmm.lω, mgfγω, true)
#@code_warntype mom_obj!(vlmm, true)
@info "benchmark"
# bm = @benchmark mom_obj!($vlmm.data[1], $vlmm.β, $vlmm.τ, $vlmm.Lγ, $vlmm.lγω, $vlmm.lω, $mgfγω, true)
# display(bm)
# @test allocs(bm) == 0
bm = @benchmark mom_obj!($vlmm, true)
display(bm)
@test allocs(bm) == 0
end

@testset "fit! (start from truth)" begin
# solver = KNITRO.KnitroSolver(outlev=KNITRO.KN_OUTLEV_ALL)
solver = Ipopt.IpoptSolver(print_level=3)
# solver = NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=4000)
# solver = NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000)
# solver = NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxeval=4000)
# solver = NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000)
@show solver
# re-set starting point to truth
copy!(vlmm.β, βtrue)
copy!(vlmm.τ, τtrue)
vlmm.Lγ   .= Lγω[1:q, 1:q]
vlmm.lγω  .= Lγω[end, 1:q]
vlmm.lω[1] = Lγω[end, end]
# fit
@info "obj at starting point"
@show mom_obj!(vlmm)
@show vlmm.β
@show vlmm.τ
@show vlmm.Lγ
@show vlmm.lγω
@show vlmm.lω
@time fit!(vlmm, solver)
@info "obj at solution"
@show mom_obj!(vlmm)
@show vlmm.β
@show vlmm.τ
@show vlmm.Lγ
@show vlmm.lγω
@show vlmm.lω
@show vlmm.∇β
@show vlmm.∇τ
@show vlmm.∇Lγ
@show vlmm.∇lγω
@show vlmm.∇lω
end

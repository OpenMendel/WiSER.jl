using BenchmarkTools, Distributions, InteractiveUtils, KNITRO
using LinearAlgebra, MixedModels, Random, Test, VarLMM

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
τtrue = [-1.5; 1.0; -0.5]
Σγ    = [2.0 0.0; 0.0 1.2]
δγω   = [0.0; 0.0] # restrictive model
σω    = [1.0]
Σγω   = [Σγ δγω; δγω' σω]
Lγω   = cholesky(Symmetric(Σγω), check = false).L
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
    # @test γω[end] == 0
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
vlmm.Lγ   .= Lγω[1:q, 1:q]
vlmm.lγω  .= Lγω[end, 1:q]
vlmm.lω[1] = Lγω[end, end]
# evaluate objective (at truth)
mgfγω = mgf_γω(vlmm)
# @test mgfγω == 1 # restrictive model
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
display(bm); println()
@test allocs(bm) == 0
end

@testset "fit! (start from truth)" begin
for solver in [
    # KNITRO.KnitroSolver(outlev=0) # outlev 0-6
    Ipopt.IpoptSolver(print_level=3)
    # NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=4000)
    # NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000)
    # NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxeval=4000)
    # NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000)
    ]
    @show solver
    # re-set starting point to truth
    copy!(vlmm.β, βtrue)
    # copy!(vlmm.τ, τtrue)
    fill!(vlmm.τ, 0)
    vlmm.Lγ   .= Lγω[1:q, 1:q]
    vlmm.lγω  .= Lγω[end, 1:q]
    vlmm.lω[1] = Lγω[end, end]
    # fit
    @info "obj at starting point"
    @show mom_obj!(vlmm)
    @time VarLMM.fit!(vlmm, solver)
    @info "obj at solution"
    @show mom_obj!(vlmm)
    @info "estimates at solution"
    println("β")
    display([βtrue vlmm.β]); println()
    println("τ")
    display([τtrue vlmm.τ]); println()
    println("Lγω")
    display([vlmm.Lγ zeros(q); vlmm.lγω' vlmm.lω[1]]); println()
    display(Lγω); println()
    @info "gradient at solution"
    @show vlmm.∇β
    @show vlmm.∇τ
    @show vlmm.∇Lγ
    @show vlmm.∇lγω
    @show vlmm.∇lω
end
end

@testset "fit! (start from LMM fit)" begin
@info "get dataframe"
df = DataFrame(vlmm)
@test size(df, 1) == sum(ns)
@test size(df, 2) == p + q + l + 2
@test all(df.id[1:ns[1]] .== 1)
@info "get LMM fit"
lmm = LinearMixedModel(@formula(y ~ 0 + x1 + x2 + x3 + x4 + x5 + (0 + z1 + z2 | id)), df)
MixedModels.fit!(lmm)
@show lmm
@show β_by_lmm = lmm.β
@show Σγ_by_lmm = Diagonal(collect(values(lmm.σρs.id.σ))) * 
    [1. lmm.σρs.id.ρ[1]; lmm.σρs.id.ρ[1] 1.] * 
    Diagonal(collect(values(lmm.σρs.id.σ)))
@show σ0_by_lmm = lmm.σ
for solver in [
    # KNITRO.KnitroSolver(outlev=0) # outlev 0-6
    Ipopt.IpoptSolver(print_level=3)
    # NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=4000)
    # NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000)
    # NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxeval=4000)
    # NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000)
    ]
    @show solver
    # re-set starting point to LMM fit
    copy!(vlmm.β, β_by_lmm)
    fill!(vlmm.τ, 0)
    vlmm.Lγ .= cholesky(Symmetric(Σγ_by_lmm)).L
    fill!(vlmm.lγω, 0)
    if σ0_by_lmm ≥ 1
        vlmm.lω[1] = 2sqrt(log(σ0_by_lmm))
    else
        vlmm.lω[1] = 0
        vlmm.τ[1] = 2log(σ0_by_lmm)
    end
    # fit
    @info "obj at starting point"
    @show mom_obj!(vlmm)
    @time VarLMM.fit!(vlmm, solver)
    @info "obj at solution"
    @show mom_obj!(vlmm)
    @info "estimates at solution"
    println("β")
    display([βtrue vlmm.β]); println()
    println("τ")
    display([τtrue vlmm.τ]); println()
    println("Lγω")
    display([vlmm.Lγ zeros(q); vlmm.lγω' vlmm.lω[1]]); println()
    display(Lγω); println()
    @info "gradient at solution"
    @show vlmm.∇β
    @show vlmm.∇τ
    @show vlmm.∇Lγ
    @show vlmm.∇lγω
    @show vlmm.∇lω
end
end

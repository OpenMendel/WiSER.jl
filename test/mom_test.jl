using BenchmarkTools, Distributions, InteractiveUtils
using LinearAlgebra, Profile, Random, Test, VarLMM

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
τtrue = [-1.5; 1.5; -0.5; zeros(l - 3)]
Σγ    = [2.0 0.0; 0.0 1.2]
δγω   = [0.2; 0.1]
σω    = [1.0]
Σγω   = [Σγ δγω; δγω' σω]
Lγω   = cholesky(Symmetric(Σγω), check = false).L
Lγ    = Lγω[1:q, 1:q]
lγω   = Lγω[q + 1, 1:q]
lω    = Lγω[q + 1, q + 1]
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
    @views vy = exp.(W * τtrue .+ dot(γω[1:q], lγω) .+ γω[end])
    y = rand(MvNormal(μy, Diagonal(vy)))
    # form a VarLmmObs instance
    obsvec[i] = VarLmmObs(y, X, Z, W)
end
# form VarLmmModel
vlmm = VarLmmModel(obsvec)

@testset "mom_obj!" begin
# set parameter values to be the truth
copy!(vlmm.β, βtrue)
copy!(vlmm.τ, τtrue)
vlmm.τ[1] = τtrue[1] + 0.5(abs2(lω) + abs2(norm(lγω + Lγ'lγω)))
vlmm.Lγ  .= Lγ
@show vlmm.β
@show vlmm.τ
@show vlmm.Lγ
# evaluate objective (at truth)
@info "obj/grad/hessian at true parameter values"
@show mom_obj!(vlmm, true, true, true)
@show vlmm.∇β
@show vlmm.∇τ
@show vlmm.∇Lγ
H = [vlmm.Hττ vlmm.HτLγ; vlmm.HτLγ' vlmm.HLγLγ]
# display(H); println()
@test norm(H - transpose(H)) < 1e-8
@test all(eigvals(Symmetric(H)) .≥ 0)
@info "type stability"
#@code_warntype mom_obj!(vlmm.data[1], vlmm.β, vlmm.τ, vlmm.Lγ, true)
#@code_warntype mom_obj!(vlmm, true)
@info "benchmark"
# bm = @benchmark mom_obj!($vlmm.data[1], $vlmm.β, $vlmm.τ, $vlmm.Lγ, true)
# display(bm)
# @test allocs(bm) == 0
bm = @benchmark mom_obj!($vlmm, true, true, true)
display(bm); println()
@test allocs(bm) == 0
# @info "profile"
# Profile.clear()
# @profile @btime mom_obj!($vlmm, true, true)
# Profile.print(format=:flat)
end

# @testset "fit! (start from truth)" begin
# println(); println(); println()
# @info "fit! (start from truth)"
# for solver in [
#     # KNITRO.KnitroSolver(outlev=3) # outlev 0-6
#     Ipopt.IpoptSolver(print_level=3)
#     # NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=10000)
#     # NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=10000)
#     # NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxeval=10000)
#     # NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000)
#     ]
#     println("----------")
#     @show solver
#     println("----------")
#     # re-set starting point to truth
#     copy!(vlmm.β, βtrue)
#     copy!(vlmm.τ, τtrue)
#     vlmm.τ[1] = τtrue[1] + 0.5(abs2(lω) + abs2(norm(lγω + Lγ'lγω)))
#     vlmm.Lγ  .= Lγ
#     @show vlmm.β
#     @show vlmm.τ
#     @show vlmm.Lγ
#     # fit
#     @info "obj at starting point"
#     @show mom_obj!(vlmm)
#     @time VarLMM.fit!(vlmm, solver)
#     @info "obj at solution"
#     @show mom_obj!(vlmm)
#     @info "estimates at solution"
#     println("β")
#     display([βtrue vlmm.β]); println()
#     println("τ")
#     display([τtrue vlmm.τ]); println()
#     println("Lγ")
#     display(vlmm.Lγ); println()
#     display(Lγ); println()
#     @info "gradient at solution"
#     @show vlmm.∇β
#     @show vlmm.∇τ
#     @show vlmm.∇Lγ
# end
# end

@testset "fit! (start from LS fit)" begin
println(); println(); println()
@info "fit! (start from LS fit)"
for solver in [
    # KNITRO.KnitroSolver(outlev=3) # outlev 0-6
    Ipopt.IpoptSolver(print_level = 3)
    # Ipopt.IpoptSolver(print_level = 3, obj_scaling_factor = 1 / m) # less accurae, grad at 10^{-1}
    # Ipopt.IpoptSolver(print_level = 3, mu_strategy = "adaptive") # same speek
    # Ipopt.IpoptSolver(print_level = 3, mehrotra_algorithm = "yes") # unstable
    # NLopt.NLoptSolver(algorithm = :LD_SLSQP, maxeval = 4000)
    # NLopt.NLoptSolver(algorithm = :LD_MMA, maxeval = 4000)
    # NLopt.NLoptSolver(algorithm = :LD_LBFGS, maxeval = 4000)
    # NLopt.NLoptSolver(algorithm = :LN_BOBYQA, maxeval = 10000)
    ]
    println("----------")
    @show solver
    println("----------")
    # re-set starting point to LS fit
    init_ls!(vlmm)
    @show vlmm.β
    @show vlmm.τ
    @show vlmm.Lγ
    # fit
    @info "obj at starting point"
    @show mom_obj!(vlmm)
    @time VarLMM.fit!(vlmm, solver)
    @info "obj at solution"
    @show mom_obj!(vlmm, true, true)
    @info "estimates at solution"
    println("β")
    display([βtrue vlmm.β]); println()
    println("τ")
    display([τtrue vlmm.τ]); println()
    println("Lγω")
    display(vlmm.Lγ); println()
    display(Lγ); println()
    @info "gradient at solution"
    @show vlmm.∇β
    @show vlmm.∇τ
    @show vlmm.∇Lγ
end
end

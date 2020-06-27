using BenchmarkTools, InteractiveUtils
using LinearAlgebra, Profile, Random, Test, VarLMM

@info "generate data"
Random.seed!(123)
# dimensions
m  = 1000 # number of individuals
ns = rand(10:10, m) # numbers of observations per individual
p  = 5    # number of fixed effects, including intercept
q  = 3    # number of random effects, including intercept
l  = 5    # number of WS variance covariates, including intercept
obsvec = Vector{VarLmmObs{Float64}}(undef, m)
# true parameter values
βtrue = [ 0.1; 6.5; -3.5; 1.0; 5  ]
τtrue = [-5.5; 1.5; -0.5; 0.0; 0.0]
Σγ    = [1.5 0.5 0.3;
         0.5 1.0 0.2;
         0.3 0.2 0.5];
δγω   = [0.2; 0.1; 0.05]
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
    @views randn!(X[:, 2:p])
    # first column intercept, remaining entries iid std normal
    Z = Matrix{Float64}(undef, ns[i], q)
    Z[:, 1] .= 1
    @views randn!(Z[:, 2:q])
    # first column intercept, remaining entries iid std normal
    W = Matrix{Float64}(undef, ns[i], l)
    W[:, 1] .= 1
    @views randn!(W[:, 2:l])
    # generate random effects: γω = Lγω * z
    mul!(γω, Lγω, randn!(z))
    # generate y
    μy = X * βtrue + Z * γω[1:q]
    @views ysd = exp.(0.5 .* (W * τtrue .+ dot(γω[1:q], lγω) .+ γω[end]))
    y = ysd .* randn(ns[i]) .+ μy
    # form a VarLmmObs instance
    obsvec[i] = VarLmmObs(y, X, Z, W)
end
# form VarLmmModel
vlmm = VarLmmModel(obsvec);

@testset "fit! (start from LS fit)" begin
println(); println(); println()
for solver in [
    # KNITRO.KnitroSolver(outlev=3), # outlev 0-6
    Ipopt.IpoptSolver(print_level = 0, mehrotra_algorithm = "yes", max_iter = 100)    
    # Ipopt.IpoptSolver(print_level=5, 
    # watchdog_shortened_iter_trigger=3, 
    # max_iter=100),# helped remedy, best number
    # Ipopt.IpoptSolver(print_level = 0)
    # Ipopt.IpoptSolver(print_level = 3, hessian_approximation = "limited-memory"),
    # Ipopt.IpoptSolver(print_level = 3, obj_scaling_factor = 1 / m) # less accurae, grad at 10^{-1}
    # Ipopt.IpoptSolver(print_level = 3, mu_strategy = "adaptive") # same speek    
    # NLopt.NLoptSolver(algorithm = :LD_SLSQP, maxeval = 4000)
    # NLopt.NLoptSolver(algorithm = :LD_MMA, maxeval = 4000),
    # NLopt.NLoptSolver(algorithm = :LD_LBFGS, maxeval = 4000)
    # NLopt.NLoptSolver(algorithm = :LN_BOBYQA, ftol_rel = 1e-12, ftol_abs = 1e-8, maxeval = 10000)
    ]
    println("----------")
    @show solver
    println("----------")
    @info "init_ls!"
    @time init_ls!(vlmm)
    println("β")
    display([βtrue vlmm.β]); println()
    println("τ")
    display([τtrue vlmm.τ]); println()
    println("Lγω")
    display(vlmm.Lγ); println()
    display(Lγ); println()

    @info "init_mom!"
    @time init_mom!(vlmm, solver)
    println("β")
    display([βtrue vlmm.β]); println()
    println("τ")
    display([τtrue vlmm.τ]); println()
    println("Lγω")
    display(vlmm.Lγ); println()
    display(Lγ); println()

    @info "fitting"
    @time VarLMM.fit!(vlmm, solver, runs=2)
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
    # @info "gradient at solution"
    # @show vlmm.∇β
    # @show vlmm.∇τ
    # @show vlmm.∇Lγ
    @test sqrt(abs2(norm(vlmm.∇τ)) + abs2(norm(vlmm.∇Lγ))) < 1e-5
    # @info "Hessian at solution"
    # @show vlmm.Hββ
    # @show vlmm.HLγLγ
    # @show vlmm.HτLγ
    # @show vlmm.Hττ

    @info "inference at solution"
    show(vlmm)
end

end

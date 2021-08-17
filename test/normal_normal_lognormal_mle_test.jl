# Test on a normal (conditional distribution of Y), normal (distribution of γ), 
# , lognormal (distribution of ω) simulation data example
module NormalNormalLognormalMLETest

using BenchmarkTools, InteractiveUtils, KNITRO
using LinearAlgebra, Profile, Random, Test, WiSER

@info "Normal Normal LogNormal Test"
@info "generate data"
rng = MersenneTwister(123)
# dimensions
m  = 1000  # number of individuals
ns = rand(rng, 10:10, m) # numbers of observations per individual
p  = 5    # number of fixed effects, including intercept
q  = 3    # number of random effects, including intercept
l  = 5    # number of WS variance covariates, including intercept
mwmdata = Vector{MixWildObs{Float64}}(undef, m)
wsdata = Vector{WSVarLmmObs{Float64}}(undef, m)
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
    @views randn!(rng, X[:, 2:p])
    # first column intercept, remaining entries iid std normal
    Z = Matrix{Float64}(undef, ns[i], q)
    Z[:, 1] .= 1
    @views randn!(rng, Z[:, 2:q])
    # first column intercept, remaining entries iid std normal
    W = Matrix{Float64}(undef, ns[i], l)
    W[:, 1] .= 1
    @views randn!(rng, W[:, 2:l])
    # generate random effects: γω = Lγω * z
    mul!(γω, Lγω, randn!(rng, z))
    # generate y
    μy = X * βtrue + Z * γω[1:q]
    @views ysd = exp.(0.5 .* (W * τtrue .+ γω[end]))
    y = ysd .* randn(rng, ns[i]) .+ μy
    # form a VarLmmObs instance
    mwmdata[i] = MixWildObs(y, X, Z, W)
    wsdata[i] = WSVarLmmObs(y, X, Z, W)
end
# form MixWildModel
mwm = MixWildModel(mwmdata, ghpts = 10);
wsm  = WSVarLmmModel(wsdata);

@testset "loglikelihood!" begin
    setparameters!(mwm, βtrue, τtrue, cholesky(Symmetric(inv(Σγω)), check=false).L)
    @test mwm.β ≈ βtrue
    @test mwm.τ ≈ τtrue
    @test mwm.Σγω ≈ Σγω
    @test mwm.Λγω ≈ inv(Σγω)
    @test istril(mwm.Lγω)
    @test mwm.Lγω * mwm.Lγω' ≈ inv(Σγω)
    @test istril(mwm.Lγ⁻¹)
    @test mwm.Lγ⁻¹' * mwm.Lγ⁻¹ ≈ inv(mwm.Λγω[1:end-1, 1:end-1])
    @test mwm.λωωsc[1] ≈ mwm.Λγω[end, end] - mwm.Λγω[1:end-1, end]' * 
    inv(mwm.Λγω[1:end-1, 1:end-1]) * mwm.Λγω[1:end-1, end]
    for i in 1:10
        @show loglikelihood!(i, mwm, false, false, true)
    end
    @show loglikelihood!(mwm, false, false, true)
    # @btime loglikelihood!($mwm, false, false, true)
end

@testset "fit! (start from LS fit)" begin
println(); println(); println()
for solver in [
    # KNITRO.KnitroSolver(outlev=3, gradopt=2), # outlev 0-6
    # Ipopt.IpoptSolver(print_level = 5, 
    #     mehrotra_algorithm = "yes", 
    #     warm_start_init_point = "yes",
    #     max_iter = 100),
    # Ipopt.IpoptSolver(print_level=0, watchdog_shortened_iter_trigger=3, max_iter=100),
    # Ipopt.IpoptSolver(print_level = 0)
    # Ipopt.IpoptSolver(print_level = 3, hessian_approximation = "limited-memory"),
    # Ipopt.IpoptSolver(print_level = 3, obj_scaling_factor = 1 / m) # less accurae, grad at 10^{-1}
    # Ipopt.IpoptSolver(print_level = 3, mu_strategy = "adaptive") # same speek    
    # NLopt.NLoptSolver(algorithm = :LD_SLSQP, maxeval = 4000)
    # NLopt.NLoptSolver(algorithm = :LD_MMA, maxeval = 4000)
    # NLopt.NLoptSolver(algorithm = :LD_LBFGS, maxeval = 4000)
    # NLopt.NLoptSolver(algorithm = :LN_BOBYQA, ftol_rel = 1e-12, ftol_abs = 1e-8, maxeval = 10000)
    # NLopt.NLoptSolver(algorithm = :LN_BOBYQA, ftol_rel = 1e-8, ftol_abs = 1e-6, maxeval = 10000),
    # NLopt.NLoptSolver(algorithm = :LN_BOBYQA), # 5 secs, obj=-1261
    # NLopt.NLoptSolver(algorithm = :LN_COBYLA) # slow
    # NLopt.NLoptSolver(algorithm = :LN_NELDERMEAD), # 50 secs, obj=686, 10 GH nodes
    NLopt.NLoptSolver(algorithm = :LN_SBPLX), # 18 secs, obj=686, 10 GH nodes
    # NLopt.NLoptSolver(algorithm = :LN_PRAXIS), # 35 secs, obj=686, 10 GH nodes
    ]
    println("----------")
    @show solver
    println("----------")
    @info "init_ls!"
    @time init_ls!(wsm)
    println("β")
    display([βtrue wsm.β]); println()
    println("τ")
    display([τtrue wsm.τ]); println()
    println("Σγ")
    display(wsm.Σγ); println()
    display(Σγ); println()

    @info "log-likelihood at init_ls"
    setparameters!(mwm, wsm.β, wsm.τ, 
    cholesky(Symmetric([inv(wsm.Σγ) zeros(q); zeros(q)' 1]), check=false).L
    )
    @show loglikelihood!(mwm, false, false, true)

    @info "MLE fitting"
    @time WiSER.fit!(mwm, solver)
    @info "obj at solution"
    @show loglikelihood!(mwm, false, false, false)
    @info "estimates at solution"
    println("β")
    display([βtrue mwm.β]); println()
    println("τ")
    display([τtrue mwm.τ]); println()
    println("Σγω")
    display(mwm.Σγω); println()
    display(Σγω); println()

    # @info "inference at solution"
    # show(mwm)
end

end
end

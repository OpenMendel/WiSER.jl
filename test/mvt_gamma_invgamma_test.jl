# Test on a student T (conditional distribution of Y), gamma (distribution of γ), 
# inv-gamma (distribution of ω) simulation data example
module MvtGammaInvgammaTest

using Distributions, InteractiveUtils, LinearAlgebra
using Random, Roots, SpecialFunctions, Test, WiSER

@info "MvT Gamma InvGamma Test"
@info "generate data"
Random.seed!(1234)
rng = MersenneTwister(1234)
# dimensions
m  = 1000 # number of individuals
ns = rand(rng, 20:20, m) # numbers of observations per individual
p  = 5    # number of fixed effects, including intercept
q  = 3    # number of random effects, including intercept
l  = 5    # number of WS variance covariates, including intercept
obsvec = Vector{WSVarLmmObs{Float64}}(undef, m)
# true parameter values
βtrue = [ 0.1; 6.5; -3.5; 1.0; 5  ]
τtrue = [-1.5; 1.5; -0.5; 0.0; 0.0]
Σγ    = [1.5 0.5 0.3;
         0.5 1.0 0.2;
         0.3 0.2 0.5];
δγω   = [0.0; 0.0; 0.0] # for easier generation of (γ, ω)
σω    = 1.0
Σγω   = [Σγ δγω; δγω' σω]
Lγω   = cholesky(Symmetric(Σγω), check = false).L
Lγ    = Lγω[1:q, 1:q]
lγω   = Lγω[q + 1, 1:q]
lω    = Lγω[q + 1, q + 1]
# parameters for Gamma random deviate ω
# If X ∼ Gamma(α, θ), then E[-ln(X)] = - ψ(α) - ln(θ), Var[-ln(X)] = ψ'(α).
# ωi = log(inv(X)) = - log(X)
# We want Var[ωi] = ψ'(α) = σω and E[ωi] = - ψ(α) - ln(θ) = 0
ωα    = Roots.find_zero(x -> trigamma(x) - σω, 1)
ωα > 1 || error("ωα needs to be >1 for the existence of mean of inverse-gamma")
ωθ    = exp(-digamma(ωα))
# parameters for Gamma random deviate γ
# If X ∼ Gamma(α, θ), then E[X] = αθ, Var[X] = αθ^2.
# We want Var[X] = 1 and don't care about mean (will shift it to 0)
γα    = 4             # shape parameter 
γθ    = sqrt(inv(γα)) # scale parameter
# degree of freedom for t
ν     = 6
# generate data
γ     = Vector{Float64}(undef, q)
z     = similar(γ) # hold vector of iid std normal
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
    # generate ω ∼ log-inv-gamma(ωα, ωθ)
    ω = -log(rand(rng, Gamma(ωα, ωθ)))
    # generate random effects: γ = Lγ * z
    # z is iid Gamma with variance 1 and shifted to have mean 0
    Distributions.rand!(rng, Gamma(γα, γθ), z)
    z .-= γα * γθ # shift to have mean 0
    mul!(γ, Lγ, z)
    # generate y from t distribution (ν, μy, σ2ϵ)
    μy  = X * βtrue + Z * γ
    σ2ϵ = W * τtrue .+ dot(γ, lγω) .+ ω
    ysd = exp.(0.5 .* (σ2ϵ))
    # note: variance of T(ν) is ν / (ν - 2)
    y = μy + sqrt(((ν - 2) / ν)) .* ysd .* rand(rng, TDist(ν), ns[i])
    # form a VarLmmObs instance
    obsvec[i] = WSVarLmmObs(y, X, Z, W)
end
# form WSVarLmmModel
vlmm = WSVarLmmModel(obsvec);

@testset "fit! (start from LS fit)" begin
println(); println(); println()
for solver in [
    # KNITRO.KnitroSolver(outlev=3), # outlev 0-6
    Ipopt.IpoptSolver(print_level = 5, 
        mehrotra_algorithm = "yes", 
        warm_start_init_point = "yes",
        max_iter=100),
    # Ipopt.IpoptSolver(print_level=0, watchdog_shortened_iter_trigger=3, max_iter=100),
    # Ipopt.IpoptSolver(print_level = 0)
    # Ipopt.IpoptSolver(print_level = 3, hessian_approximation = "limited-memory"),
    # Ipopt.IpoptSolver(print_level = 3, obj_scaling_factor = 1 / m) # instable
    # Ipopt.IpoptSolver(print_level = 5, mu_strategy = "adaptive") # same speed
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

    @info "unweighted obj/gradient/hessian at init_ls"
    @show nlsv_obj!(vlmm, true, true, true)
    # println("∇β")
    # display(vlmm.∇β); println()
    # println("∇τ")
    # display(vlmm.∇τ); println()
    # println("∇Lγ")
    # display(vlmm.∇Lγ); println()
    # println("Hββ")
    # display(vlmm.Hββ); println()
    # println("Hττ")
    # display(vlmm.Hττ); println()
    # println("HτLγ")
    # display(vlmm.HτLγ); println()
    # println("HLγLγ")
    # display(vlmm.HLγLγ); println()
    # println("∇:")
    # ∇ = [vlmm.∇τ; vech(vlmm.∇Lγ)]
    # display(∇); println()
    # println("FIM:")
    # FIM = [vlmm.Hττ vlmm.HτLγ; vlmm.HτLγ' vlmm.HLγLγ]
    # display(FIM); println()
    # @show eigvals(Symmetric(FIM))
    # @show Symmetric(FIM) \ ∇

    @info "init_mom!"
    @time init_mom!(vlmm, solver)
    println("β")
    display([βtrue vlmm.β]); println()
    println("τ")
    display([τtrue vlmm.τ]); println()
    println("Lγω")
    display(vlmm.Lγ); println()
    display(Lγ); println()

    @info "weighted obj/gradient/hessian at init_ls"
    init_ls!(vlmm)
    vlmm.iswtnls[1] = true
    update_wtmat!(vlmm)
    @show nlsv_obj!(vlmm, true, true, true)
    # println("∇β")
    # display(vlmm.∇β); println()
    # println("∇τ")
    # display(vlmm.∇τ); println()
    # println("∇Lγ")
    # display(vlmm.∇Lγ); println()
    # println("Hββ")
    # display(vlmm.Hββ); println()
    # println("Hττ")
    # display(vlmm.Hττ); println()
    # println("HτLγ")
    # display(vlmm.HτLγ); println()
    # println("HLγLγ")
    # display(vlmm.HLγLγ); println()
    println("∇:")
    ∇ = [vlmm.∇τ; vech(vlmm.∇Lγ)]
    display(∇); println()
    # println("FIM:")
    # FIM = [vlmm.Hττ vlmm.HτLγ; vlmm.HτLγ' vlmm.HLγLγ]
    # display(FIM); println()
    # @show eigvals(Symmetric(FIM))
    # @show Symmetric(FIM) \ ∇

    @info "WNLS fitting"
    @time WiSER.fit!(vlmm, solver, runs=2) # start from init_ls by default
    @info "obj at solution"
    @show nlsv_obj!(vlmm, true, true)
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

end
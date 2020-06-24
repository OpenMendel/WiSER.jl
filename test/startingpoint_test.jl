using VarLMM, DataFrames, Random, LinearAlgebra, Test

function createvlmm(t, k, j)
    p  = 5    # number of fixed effects, including intercept
    q  = 2    # number of random effects, including intercept
    l  = 5    # number of WS variance covariates, including intercept
    q◺ = ◺(q)

    # true parameter values
    global βtrue = [0.1; 6.5; -3.5; 1.0; 5]
    global τtrue = [0.0; 0.5; -0.2; 0.5; 0.0]
    Σγ    = Matrix(Diagonal([2.0; 1.2])) 
    δγω   = [0.2; 0.1]
    σω    = [1.0]

    Σγω   = [Σγ δγω; δγω' σω]
    Lγω   = cholesky(Symmetric(Σγω), check = false).L
    global Lγ    = Lγω[1:q, 1:q]
    lγω   = Lγω[q + 1, 1:q]
    lω    = Lγω[q + 1, q + 1]

    vechLγ = vech(Lγ)
    # generate data
    γω = Vector{Float64}(undef, q + 1)
    z  = similar(γω) # hold vector of iid std normal

    intervals = zeros(p + l, 2) #hold intervals
    curcoverage = zeros(p + l) #hold current coverage resutls
    trueparams = [βtrue; τtrue] #hold true parameters

    #simulation parameters
    samplesizes = collect(1000:1000:6000)
    ns = [10; 25; 50; 100; 1000]

    m = samplesizes[t]
    ni = ns[k] # number of observations per individual
    obsvec = Vector{VarLmmObs{Float64}}(undef, m)
    println("rep $j obs per person $ni samplesize $m")
    Random.seed!(j + 100000k + 1000t)
    for i in 1:m
        # first column intercept, remaining entries iid std normal
        X = Matrix{Float64}(undef, ni, p)
        X[:, 1] .= 1
        @views Distributions.rand!(Normal(), X[:, 2:p])
        # first column intercept, remaining entries iid std normal
        Z = Matrix{Float64}(undef, ni, q)
        Z[:, 1] .= 1
        @views Distributions.rand!(Normal(), Z[:, 2:q])
        # first column intercept, remaining entries iid std normal
        W = Matrix{Float64}(undef, ni, l)
        W[:, 1] .= 1
        @views Distributions.rand!(Normal(), W[:, 2:l])
        # generate random effects: γω = Lγω * z
        mul!(γω, Lγω, Distributions.rand!(Normal(), z))
        # generate y
        μy = X * βtrue + Z * γω[1:q]
        @views vy = exp.(W * τtrue .+ dot(γω[1:q], lγω) .+ γω[end])
        y = rand(MvNormal(μy, Diagonal(vy)))
        # form a VarLmmObs instance
        obsvec[i] = VarLmmObs(y, X, Z, W)
    end
    # form VarLmmModel
    vlmm = VarLmmModel(obsvec);
    return vlmm
end

# vlmm1 = createvlmm(1, 1, 203) # trouble case
vlmm1 = createvlmm(2, 1, 35)

solver = Ipopt.IpoptSolver(print_level=0)
# solver = NLopt.NLoptSolver(algorithm = :LN_BOBYQA, 
#     ftol_rel = 1e-12, ftol_abs = 1e-8, maxeval = 10000)
# solver = NLopt.NLoptSolver(algorithm = :LD_SLSQP, maxeval = 4000)
# solver = NLopt.NLoptSolver(algorithm = :LD_MMA, maxeval = 4000)
# solver = NLopt.NLoptSolver(algorithm = :LD_LBFGS, maxeval = 4000)

@info "starting point by LS init_ls()"
VarLMM.init_ls!(vlmm1)
println("β"); display(vlmm1.β); println()
println("τ"); display(vlmm1.τ); println()
println("Lγ"); display(vlmm1.Lγ); println()
@show norm(vlmm1.β - βtrue)
@show norm(vlmm1.τ[2:end] - τtrue[2:end])
@show norm(vlmm1.Lγ - Lγ)

@info "unweighted NLS fit (start from LS)"
@time init_mom!(vlmm1, solver) #will return error
println("β"); display(vlmm1.β); println()
println("τ"); display(vlmm1.τ); println()
println("Lγ"); display(vlmm1.Lγ); println()
@show norm(vlmm1.β - βtrue)
@show norm(vlmm1.τ[2:end] - τtrue[2:end])
@show norm(vlmm1.Lγ - Lγ)

# @info "weighted NLS fit (start from true τ and LS β, Σγ)"
# init_ls!(vlmm1)
# # vlmm1.τ .= [-0.1; 0.5; -0.2; 0.5; 0.0]
# vlmm1.τ .= [0.27, 0.14, -0.04, 0.13, -0.02] * 5
# fit!(vlmm1, solver; init = vlmm1, runs = 2) #should work now
# println("β"); display(vlmm1.β); println()
# println("τ"); display(vlmm1.τ); println()
# println("Lγ"); display(vlmm1.Lγ); println()
# show(vlmm1)

@info "weighted fit (start from LS)"
@time fit!(vlmm1, solver; init = init_ls!(vlmm1), runs = 2)
println("β"); display(vlmm1.β); println()
println("τ"); display(vlmm1.τ); println()
println("Lγ"); display(vlmm1.Lγ); println()
@show norm(vlmm1.β - βtrue)
@show norm(vlmm1.τ[2:end] - τtrue[2:end])
@show norm(vlmm1.Lγ - Lγ)

@info "weighted fit (start from MoM)"
@time fit!(vlmm1, solver; init = init_mom!(vlmm1, solver), runs = 2)
println("β"); display(vlmm1.β); println()
println("τ"); display(vlmm1.τ); println()
println("Lγ"); display(vlmm1.Lγ); println()
@show norm(vlmm1.β - βtrue)
@show norm(vlmm1.τ[2:end] - τtrue[2:end])
@show norm(vlmm1.Lγ - Lγ)

### If you revert back to initialize intercept only these cases will also fail:
# ts = [2, 2, 2, 4, 1, 3, 2, 1, 1, 2]
# ks = [1, 1, 1, 1, 2, 1, 2, 1, 1, 1]
# js = [35, 53, 109, 148, 168, 174, 100, 7, 203, 233]

# for i in 1:length(ts)
#     vlmmi = createvlmm(1, 1, 203)
#     fit!(vlmm1, fittype=:Weighted, weightedruns = 2) #should work now
# end

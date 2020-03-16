using BenchmarkTools, Distributions, InteractiveUtils
using LinearAlgebra, MixedModels, Random, Test, VarLMM

@info "generate data"
Random.seed!(123)
# dimensions
m  = 6000 # number of individuals
ns = rand(20:20, m) # numbers of observations per individual
p  = 5    # number of fixed effects, including intercept
q  = 2    # number of random effects, including intercept
l  = 3    # number of WS variance covariates, including intercept
obsvec = Vector{VarLmmObs{Float64}}(undef, m)
# true parameter values
βtrue = [0.1; 6.5; -3.5; 1.0; 5]
τtrue = [1.5; -1.0; 0.5]
Σγ    = [2.0 0.0; 0.0 1.2]
δγω   = [0.1; 0.2]
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
    @views vy = exp.(W * τtrue .+ dot(γω[1:q], Lγω[end, 1:q]) .+ γω[end])
    y = rand(MvNormal(μy, Diagonal(vy)))
    # form a VarLmmObs instance
    obsvec[i] = VarLmmObs(y, X, Z, W)
end
# form VarLmmModel
vlmm = VarLmmModel(obsvec)

# @testset "mom_obj!" begin
# # set parameter values to be the truth
# copy!(vlmm.β, βtrue)
# copy!(vlmm.τ, τtrue)
# vlmm.τ[1] = τtrue[1] + 0.5(abs2(lω) + abs2(norm(lγω + Lγ'lγω)))
# vlmm.Lγ  .= Lγ
# @show vlmm.β
# @show vlmm.τ
# @show vlmm.Lγ
# # evaluate objective (at truth)
# @info "obj/grad at true parameter values"
# @show mom_obj!(vlmm, true)
# @show vlmm.∇β
# @show vlmm.∇τ
# @show vlmm.∇Lγ
# @info "type stability"
# #@code_warntype mom_obj!(vlmm.data[1], vlmm.β, vlmm.τ, vlmm.Lγ, true)
# #@code_warntype mom_obj!(vlmm, true)
# @info "benchmark"
# # bm = @benchmark mom_obj!($vlmm.data[1], $vlmm.β, $vlmm.τ, $vlmm.Lγ, true)
# # display(bm)
# # @test allocs(bm) == 0
# bm = @benchmark mom_obj!($vlmm, true)
# display(bm); println()
# @test allocs(bm) == 0
# end

@testset "fit! (start from truth)" begin
@info "fit! (start from truth)"
for solver in [
    # KNITRO.KnitroSolver(outlev=3) # outlev 0-6
    Ipopt.IpoptSolver(print_level=3)
    # NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=10000)
    # NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=10000)
    # NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxeval=10000)
    # NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000)
    ]
    println("----------")
    @show solver
    println("----------")
    # re-set starting point to truth
    copy!(vlmm.β, βtrue)
    copy!(vlmm.τ, τtrue)
    vlmm.τ[1] = τtrue[1] + 0.5(abs2(lω) + abs2(norm(lγω + Lγ'lγω)))
    vlmm.Lγ  .= Lγ
    @show vlmm.β
    @show vlmm.τ
    @show vlmm.Lγ
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
    println("Lγ")
    display(vlmm.Lγ); println()
    display(Lγ); println()
    @info "gradient at solution"
    @show vlmm.∇β
    @show vlmm.∇τ
    @show vlmm.∇Lγ
end
end

@testset "fit! (start from LMM fit)" begin
@info "fit! (start from LMM fit)"
@info "get dataframe"
df = DataFrame(vlmm)
@test size(df, 1) == sum(ns)
@test size(df, 2) == p + q + l + 2
@test all(df.id[1:ns[1]] .== 1)
@info "get LMM fit"
# lmm_formula = Term(:y) ~ 
#     Term(Symbol(0)) + sum([Term(Symbol("x$i")) for i = 1:p]) +
#     (Term(Symbol(0)) + sum([Term(Symbol("z$i")) for i = 1:q]) | Term(Symbol("id")))
lmm = LinearMixedModel(@formula(y ~ 0 + x1 + x2 + x3 + x4 + x5 + (0 + z1 + z2 | id)), df)
MixedModels.fit!(lmm)
@show lmm
@show β_by_lmm = lmm.β
@show Σγ_by_lmm = Diagonal(collect(values(lmm.σρs.id.σ))) * 
    [1. lmm.σρs.id.ρ[1]; lmm.σρs.id.ρ[1] 1.] * 
    Diagonal(collect(values(lmm.σρs.id.σ)))
@show σ0_by_lmm = lmm.σ
for solver in [
    # KNITRO.KnitroSolver(outlev=3) # outlev 0-6
    Ipopt.IpoptSolver(print_level=3)
    # NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=4000)
    # NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000)
    # NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxeval=4000)
    # NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000)
    ]
    println("----------")
    @show solver
    println("----------")
    # re-set starting point to LMM fit
    copy!(vlmm.β, β_by_lmm)
    fill!(vlmm.τ, 0)
    vlmm.τ[1] = 2log(σ0_by_lmm)
    vlmm.Lγ .= cholesky(Symmetric(Σγ_by_lmm), check = false).L
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
    display(vlmm.Lγ); println()
    display(Lγ); println()
    @info "gradient at solution"
    @show vlmm.∇β
    @show vlmm.∇τ
    @show vlmm.∇Lγ
end
end

@testset "fit! (start from LS fit)" begin
@info "fit! (start from LS fit)"
# LS estimate for β
xtx = zeros(p, p)
xty = zeros(p)
for i in eachindex(vlmm.data)
    xtx .+= vlmm.data[i].X'vlmm.data[i].X
    xty .+= vlmm.data[i].X'vlmm.data[i].y
end
update_res!(vlmm)
# LS etimate for σ2ω
σ2ω_by_ls = 0.0
n   = 0
for i in eachindex(vlmm.data)
    σ2ω_by_ls += sum(abs2, vlmm.data[i].res)
    n   += length(vlmm.data[i].y)
end
σ2ω_by_ls /= n
# LS estimate for Σγ
ztz2 = zeros(q * q, q * q)
ztr2 = zeros(q * q)
for i in eachindex(vlmm.data)
    ztz    = vlmm.data[i].Z'vlmm.data[i].Z    
    ztz2 .+= kron(ztz, ztz)
    ztr    = vlmm.data[i].Z'vlmm.data[i].res
    ztr2 .+= kron(ztr, ztr)
end
Σγ_by_ls = reshape(ztz2 \ ztr2, (q, q))
for solver in [
    # KNITRO.KnitroSolver(outlev=3) # outlev 0-6
    Ipopt.IpoptSolver(print_level=3)
    # NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=4000)
    # NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000)
    # NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxeval=4000)
    # NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000)
    ]
    println("----------")
    @show solver
    println("----------")
    # re-set starting point to LS fit
    @show copy!(vlmm.β, cholesky(xtx) \ xty)
    fill!(vlmm.τ, 0)
    vlmm.τ[1] = log(σ2ω_by_ls)
    @show vlmm.τ
    vlmm.Lγ .= cholesky(Σγ_by_ls, check=false).L
    @show vlmm.Lγ
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
    display(vlmm.Lγ); println()
    display(Lγ); println()
    @info "gradient at solution"
    @show vlmm.∇β
    @show vlmm.∇τ
    @show vlmm.∇Lγ
end
end

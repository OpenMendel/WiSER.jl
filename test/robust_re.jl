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
τtrue = [1.5; -1.0; 0.5]
Σγ    = [2.0 0.0; 0.0 1.2]
Lγ    = cholesky(Symmetric(Σγ), check = false).L
αω    = 3.0 # ωi ∼ InvGamma(α, β) 
βω    = 1.0 # α needs >2 for variance to exist
ωdist = InverseGamma(αω, βω)
@show Distributions.mean(ωdist)
@show Distributions.var(ωdist)
# generate data
γ = Vector{Float64}(undef, q)
z  = similar(γ) # hold vector of iid std normal
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
    # generate random effects: γ = Lγ * z
    mul!(γ, Lγ, Distributions.rand!(Normal(), z))
    # @test γω[end] == 0
    # generate y
    μy = X * βtrue + Z * γ
    @views vy = exp.(W * τtrue) * rand(ωdist)
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

# @testset "mom_obj!" begin
# # set parameter values to be the truth
# @show copy!(vlmm.β, βtrue)
# @show copy!(vlmm.τ, τtrue)
# vlmm.Lγ   .= Lγω[1:q, 1:q]
# vlmm.lγω  .= Lγω[end, 1:q]
# vlmm.lω[1] = Lγω[end, end]
# # evaluate objective (at truth)
# mgfγω = mgf_γω(vlmm)
# @info "obj/grad at true parameter values"
# @show mom_obj!(vlmm, true)
# @show vlmm.∇β
# @show vlmm.∇τ
# @show vlmm.∇Lγ
# @show vlmm.∇lγω
# @show vlmm.∇lω
# @info "type stability"
# #@code_warntype mom_obj!(vlmm.data[1], vlmm.β, vlmm.τ, vlmm.Lγ, vlmm.lγω, vlmm.lω, mgfγω, true)
# #@code_warntype mom_obj!(vlmm, true)
# @info "benchmark"
# # bm = @benchmark mom_obj!($vlmm.data[1], $vlmm.β, $vlmm.τ, $vlmm.Lγ, $vlmm.lγω, $vlmm.lω, $mgfγω, true)
# # display(bm)
# # @test allocs(bm) == 0
# bm = @benchmark mom_obj!($vlmm, true)
# display(bm); println()
# @test allocs(bm) == 0
# end

# @testset "fit! (start from truth)" begin
# @info "fit! (start from truth)"
# for solver in [
#     KNITRO.KnitroSolver(outlev=3) # outlev 0-6
#     Ipopt.IpoptSolver(print_level=3)
#     # NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=4000)
#     # NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000)
#     # NLopt.NLoptSolver(algorithm=:LD_LBFGS, maxeval=4000)
#     # NLopt.NLoptSolver(algorithm=:LN_BOBYQA, maxeval=10000)
#     ]
#     @show solver
#     # re-set starting point to truth
#     copy!(vlmm.β, βtrue)
#     # copy!(vlmm.τ, τtrue)
#     fill!(vlmm.τ, 0)
#     vlmm.Lγ   .= Lγω[1:q, 1:q]
#     vlmm.lγω  .= Lγω[end, 1:q]
#     vlmm.lω[1] = Lγω[end, end]
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
#     println("Lγω")
#     display([vlmm.Lγ zeros(q); vlmm.lγω' vlmm.lω[1]]); println()
#     display(Lγω); println()
#     @info "gradient at solution"
#     @show vlmm.∇β
#     @show vlmm.∇τ
#     @show vlmm.∇Lγ
#     @show vlmm.∇lγω
#     @show vlmm.∇lω
# end
# end

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
    @show solver
    # re-set starting point to LMM fit
    copy!(vlmm.β, β_by_lmm)
    fill!(vlmm.τ, 0)
    vlmm.Lγ .= cholesky(Symmetric(Σγ_by_lmm), check = false).L
    fill!(vlmm.lγω, 0)
    vlmm.τ[1] = 2log(σ0_by_lmm)
    vlmm.lω[1] = 0
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
    @show vlmm.∇lγω
    @show vlmm.∇lω
end
# # try to esitmate \sigma_\omega
# Σγ = vlmm.Lγ * vlmm.Lγ'
# xtx = zeros(m + 1, m + 1)
# xty = zeros(m + 1)
# mgfγω = mgf_γω(vlmm)
# for i in 1:m
#     Ri = vlmm.data[i].res * vlmm.data[i].res'
#     Vi = vlmm.data[i].Z * Σγ * vlmm.data[i].Z'
#     ri = diag(Ri)
#     ei = mgfγω .* exp.(vlmm.data[i].W * vlmm.τ)
#     vi = diag(Vi)
#     # display([ri ei vi]); println()
#     xtx[1, 1]    += abs2(norm(Vi))
#     xty[1]       += dot(Ri, Vi)
#     xtx[1, i+1]   = dot(vi, ei)
#     xtx[i+1, 1]   = xtx[1, i+1]
#     xtx[i+1, i+1] = abs2(norm(ei))
#     xty[i+1]      = dot(ri, ei)
# end
# xtx = sparse(xtx)
# # multiplicative algorithm to solve NNLS
# c = ones(m + 1)
# for iter in 1:50
#     c ./= xtx * c
#     c .*= xty
# end
# @show c[1:10]
# @show var(log.(c[2:end]))
# try to esitmate \sigma_\omega
# Σγ = vlmm.Lγ * vlmm.Lγ'
# mgfγω = mgf_γω(vlmm)
# c = Vector{Float64}(undef, m)
# for i in 1:m
#     Ri = vlmm.data[i].res * vlmm.data[i].res'
#     Vi = vlmm.data[i].Z * Σγ * vlmm.data[i].Z'
#     ri = diag(Ri)
#     # ei = mgfγω .* exp.(vlmm.data[i].W * vlmm.τ)
#     ei = exp.(vlmm.data[i].W * vlmm.τ)
#     vi = diag(Vi)
#     c[i] = mean(log.(ri ./ ei))
# end
# @show c[1:10]
# @show var(c)
end

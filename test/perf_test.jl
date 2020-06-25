using BenchmarkTools, InteractiveUtils
using LinearAlgebra, Profile, Random, Test, VarLMM

@info "generate data"
Random.seed!(123)
# dimensions
m  = 6000 # number of individuals
ns = rand(5:11, m) # numbers of observations per individual
p  = 5    # number of fixed effects, including intercept
q  = 3    # number of random effects, including intercept
l  = 5    # number of WS variance covariates, including intercept
obsvec = Vector{VarLmmObs{Float64}}(undef, m)
# true parameter values
βtrue = [0.1; 6.5; -3.5; 1.0; 5]
τtrue = [-1.5; 1.5; -0.5; zeros(l - 3)]
Σγ    = Matrix(Diagonal([2.0; 1.2; rand(q - 2)])) # full rank case
δγω   = [0.2; 0.1; rand(q - 2) ./ 10]
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

@testset "init_ls!" begin
# least squares starting point
init_ls!(vlmm)
@show vlmm.β
@show vlmm.τ
@show vlmm.Lγ
# @info "type stability"
# @code_warntype init_ls!(vlmm)
@info "benchmark"
bm = @benchmark init_ls!($vlmm)
display(bm); println()
@test allocs(bm) == 0
# @info "profile"
# Profile.clear()
# @profile @btime init_ls!(vlmm)
# Profile.print(format=:flat)
end

@testset "update_wtmat!" begin
# set parameter values to be the truth
copy!(vlmm.β, βtrue)
copy!(vlmm.τ, τtrue)
vlmm.τ[1] = τtrue[1] + 0.5(abs2(lω) + abs2(norm(lγω + Lγ'lγω)))
vlmm.Lγ  .= Lγ
@show vlmm.β
@show vlmm.τ
@show vlmm.Lγ
# update weight matrix
update_wtmat!(vlmm)
@show vlmm.β
@show vlmm.∇β
# @show vlmm.Hββ
# @info "type stability"
# @code_warntype update_wtmat!(vlmm)
@info "benchmark"
bm = @benchmark update_wtmat!($vlmm)
display(bm); println()
@test allocs(bm) == 0
# @info "profile"
# Profile.clear()
# @profile @btime update_wtmat!(vlmm)
# Profile.print(format=:flat)
end

@testset "mom_obj! (unweighted)" begin
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
vlmm.iswtnls[1] = false
@show mom_obj!(vlmm, true, true, true)
@show vlmm.∇β
@show vlmm.∇τ
@show vlmm.∇Lγ
H = [vlmm.Hττ vlmm.HτLγ; vlmm.HτLγ' vlmm.HLγLγ]
# display(H); println()
@test norm(H - transpose(H)) / norm(H) < 1e-8
@test all(eigvals(Symmetric(H)) .≥ 0)
# @info "type stability"
# @code_warntype mom_obj!(vlmm.data[1], vlmm.β, vlmm.τ, vlmm.Lγ, true)
# @code_warntype mom_obj!(vlmm, true)
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

@testset "mom_obj! (weighted)" begin
# set parameter values to be the truth
copy!(vlmm.β, βtrue)
copy!(vlmm.τ, τtrue)
vlmm.τ[1] = τtrue[1] + 0.5(abs2(lω) + abs2(norm(lγω + Lγ'lγω)))
vlmm.Lγ  .= Lγ
@show vlmm.β
@show vlmm.τ
@show vlmm.Lγ
# update_wtmat then evaluate at the truth
update_wtmat!(vlmm)
vlmm.iswtnls[1] = true
@show mom_obj!(vlmm, true, true, true)
@show vlmm.∇β
@show vlmm.∇τ
@show vlmm.∇Lγ
H = [vlmm.Hττ vlmm.HτLγ; vlmm.HτLγ' vlmm.HLγLγ]
# display(H); println()
@test norm(H - transpose(H)) / norm(H) < 1e-8
@test all(eigvals(Symmetric(H)) .≥ 0)
# @info "type stability"
# @code_warntype mom_obj!(vlmm.data[1], vlmm.β, vlmm.τ, vlmm.Lγ, true)
# @code_warntype mom_obj!(vlmm, true)
@info "benchmark"
bm = @benchmark mom_obj!($vlmm, true, true, true)
display(bm); println()
@test allocs(bm) == 0
# @info "profile"
# Profile.clear()
# @profile @btime mom_obj!($vlmm, true, true, true)
# Profile.print(format=:flat)
end

@testset "mom_obj! (weighted) - parallel" begin
# set parameter values to be the truth
copy!(vlmm.β, βtrue)
copy!(vlmm.τ, τtrue)
vlmm.τ[1] = τtrue[1] + 0.5(abs2(lω) + abs2(norm(lγω + Lγ'lγω)))
vlmm.Lγ  .= Lγ
@show vlmm.β
@show vlmm.τ
@show vlmm.Lγ
# update_wtmat then evaluate at the truth
update_wtmat!(vlmm)
vlmm.iswtnls[1] = true
vlmm.ismthrd[1] = true
@show mom_obj!(vlmm, true, true, true)
@show vlmm.∇β
@show vlmm.∇τ
@show vlmm.∇Lγ
H = [vlmm.Hττ vlmm.HτLγ; vlmm.HτLγ' vlmm.HLγLγ]
# display(H); println()
@test norm(H - transpose(H)) / norm(H) < 1e-8
@test all(eigvals(Symmetric(H)) .≥ 0)
# @info "type stability"
# @code_warntype mom_obj!(vlmm.data[1], vlmm.β, vlmm.τ, vlmm.Lγ, true)
# @code_warntype mom_obj!(vlmm, true)
@info "benchmark"
vlmm.iswtnls[1] = true
vlmm.ismthrd[1] = true
bm = @benchmark mom_obj!($vlmm, true, true, true)
display(bm); println()
# @info "profile"
# Profile.clear()
# @profile @btime mom_obj!($vlmm, true, true, true)
# Profile.print(format=:flat)
end

using BenchmarkTools, Distributions, InteractiveUtils
using LinearAlgebra, Profile, Random, Test, VarLMM

@info "generate data"
Random.seed!(1234)
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
# Σγ    = Matrix(Diagonal([2.0; 1.2; zeros(q - 2)])) # singular case
# δγω   = zeros(q)
# σω    = [0.0]
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
    # generate y
    μy = X * βtrue + Z * γω[1:q]
    @views vy = exp.(W * τtrue .+ dot(γω[1:q], lγω) .+ γω[end])
    y = rand(MvNormal(μy, Diagonal(vy)))
    # form a VarLmmObs instance
    obsvec[i] = VarLmmObs(y, X, Z, W)
end
# form VarLmmModel
vlmm = VarLmmModel(obsvec);

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
# @info "obj/grad/hessian at true parameter values"
# @show mom_obj!(vlmm, true, true, true)
# @show vlmm.∇β
# @show vlmm.∇τ
# @show vlmm.∇Lγ
# H = [vlmm.Hττ vlmm.HτLγ; vlmm.HτLγ' vlmm.HLγLγ]
# # display(H); println()
# @test norm(H - transpose(H)) / norm(H) < 1e-8
# @test all(eigvals(Symmetric(H)) .≥ 0)
# # @info "type stability"
# # @code_warntype mom_obj!(vlmm.data[1], vlmm.β, vlmm.τ, vlmm.Lγ, true)
# # @code_warntype mom_obj!(vlmm, true)
# @info "benchmark"
# # bm = @benchmark mom_obj!($vlmm.data[1], $vlmm.β, $vlmm.τ, $vlmm.Lγ, true)
# # display(bm)
# # @test allocs(bm) == 0
# bm = @benchmark mom_obj!($vlmm, true, true, true)
# display(bm); println()
# @test allocs(bm) == 0
# # @info "profile"
# # Profile.clear()
# # @profile @btime mom_obj!($vlmm, true, true)
# # Profile.print(format=:flat)
# end

# @testset "fit! (start from LS fit)" begin
# println(); println(); println()
# @info "fit! (start from LS fit)"
# for solver in [
#     # KNITRO.KnitroSolver(outlev=3), # outlev 0-6
#     Ipopt.IpoptSolver(print_level = 3, watchdog_shortened_iter_trigger=3),# helped remedy, best number
#     # Ipopt.IpoptSolver(print_level = 3),
#     # Ipopt.IpoptSolver(print_level = 3, hessian_approximation = "limited-memory"),
#     # Ipopt.IpoptSolver(print_level = 3, obj_scaling_factor = 1 / m) # less accurae, grad at 10^{-1}
#     # Ipopt.IpoptSolver(print_level = 3, mu_strategy = "adaptive") # same speek
#     # Ipopt.IpoptSolver(print_level = 3, mehrotra_algorithm = "yes") # unstable
#     # NLopt.NLoptSolver(algorithm = :LD_SLSQP, maxeval = 4000)
#     # NLopt.NLoptSolver(algorithm = :LD_MMA, maxeval = 4000),
#     # NLopt.NLoptSolver(algorithm = :LD_LBFGS, maxeval = 4000)
#     # NLopt.NLoptSolver(algorithm = :LN_BOBYQA, ftol_rel = 1e-12, ftol_abs = 1e-8, maxeval = 10000)
#     ]
#     println("----------")
#     @show solver
#     println("----------")
#     # re-set starting point to LS fit
#     @info "initilize from LS estimate"
#     vlmm.weighted[1] = false
#     init_ls!(vlmm) # warm up
#     @time init_ls!(vlmm)
#     @show vlmm.β
#     @show vlmm.τ
#     println("vlmm.Lγ"); display(vlmm.Lγ); println()
#     # fit
#     @info "fittng ..."
#     @info "obj at starting point"
#     @show mom_obj!(vlmm)
#     @time VarLMM.fit!(vlmm, solver)
#     @info "obj at solution"
#     @show mom_obj!(vlmm, true, true)
#     @time mom_obj!(vlmm)
#     # bm = @benchmark mom_obj!($vlmm, true, true, true)
#     # display(bm); println()
#     @info "estimates at solution"
#     println("β")
#     display([βtrue vlmm.β]); println()
#     println("τ")
#     display([τtrue vlmm.τ]); println()
#     println("Lγω")
#     display(vlmm.Lγ); println()
#     display(Lγ); println()
#     @info "gradient at solution"
#     @show vlmm.∇β
#     @show vlmm.∇τ
#     @show vlmm.∇Lγ
#     # @info "res2 expwτ diag(ZLLtZt)"
#     # for i in 1:5
#     #     display([vlmm.data[i].res2 vlmm.data[i].expwτ vlmm.data[i].zlltzt_dg]); println()
#     # end
#     # re-fit β by weighted least squares
#     @info "re-fit by WLS"
#     init_wls!(vlmm) # warm up
#     @time init_wls!(vlmm)
#     update_wtmat!(vlmm) # warm up 
#     @time update_wtmat!(vlmm)
#     # Profile.clear()
#     # @profile @btime update_wtmat!($vlmm)
#     # Profile.print(format=:flat)
    
#     # display(vlmm.data[1].wtmat); println()
#     # @show eigvals(vlmm.data[1].wtmat)
#     @info "fittng WLS..."
#     vlmm.weighted[1] = true
#     @info "obj at starting point"
#     @show mom_obj!(vlmm)
#     @time mom_obj!(vlmm)
#     @time VarLMM.fit!(vlmm, solver)
#     @info "obj at solution"
#     @show mom_obj!(vlmm, true, true)
#     @info "estimates at solution"
#     println("β")
#     display([βtrue vlmm.β]); println()
#     println("τ")
#     display([τtrue vlmm.τ]); println()
#     println("Lγω")
#     display(vlmm.Lγ); println()
#     display(Lγ); println()
#     @info "gradient at solution"
#     @show vlmm.∇β
#     @show vlmm.∇τ
#     @show vlmm.∇Lγ
#     @show vlmm.HLγLγ
#     @show vlmm.HτLγ
#     @show vlmm.Hττ
#     @show mom_obj!(vlmm)
#     # BenchmarkTools.DEFAULT_PARAMETERS.seconds = 15
#     # under m = 800, ni = 800:1000
#     @test mom_obj!(vlmm, true, true, true) ≈ 3.9331213647326386e8
#     @test isapprox(vlmm.∇Lγ, [-2.942432139024648e-14 -8.771293373745995e-15 -1.731001191734574e-14;
#      -1.66797855963846e-14 1.368796656887832e-13 1.4264115589582018e-13;
#      -3.7197885639546623e-14 1.9760068098839214e-13 1.3381970751486514e-13], atol = 1e-9)
#     @test isapprox(vlmm.∇τ, [1.63140612130519e-11, -8.898693550918324e-6, 
#     8.625343451029721e-6, -1.2175564673810868e-5, 6.686529573585176e-6], atol = 1e-9)
#     @test vlmm.HLγLγ ≈ [1606.8212031880164 -1.0819248030650936 22.784856175380057 -0.11003556836724929 -0.1734625015321543 0.01887503882470629;
#      -1.0819248030664714 1360.0932662911655 102.50990599123048 -21.73085966115454 0.00868466357737379 0.08737846866823742; 
#      22.78485617537875 102.50990599122669 2821.014939401117 -0.7880012357814555 -22.60513282678334 2.4044485153821404;
#       -0.11003556836724737 -21.73085966115631 -0.7880012357816188 2691.319229652556 97.44562964890231 -0.43582707025984946;
#        -0.17346250153217233 0.008684663577271406 -22.605132826786953 97.44562964889863 2795.4163699159676 -11.99303109202686;
#         0.018875038824687215 0.0873784686681497 2.4044485153797663 -0.43582707026003936 -11.9930310920323 5637.172602524909]
#     @test vlmm.HτLγ ≈ [0.08061709850319179 -0.0008442092590189238 0.00326481833580463 0.17919870962733725 0.012645821355870546 0.5372198431221124;
#      -0.07663236550581361 0.0006977475632101063 -0.0019465597712954887 -0.16736277218087242 -0.011699788669026266 -0.5040026057826498; 0.036919585414603384 0.0004406785929035899 0.0013228101839111177 0.08144949059682167 0.0057235072103784534 0.24372520935333877; 0.0016268894405231718 0.00015276783041439499 -5.607528861476198e-5 0.002672489834683056 0.0005857851754622765 0.009032345832346383; -0.013251377781719333 -3.6558639768175196e-5 0.00016054956129225216 -0.029642707002518517 -0.0018825494928452783 -0.08552102257965982]
#     @test vlmm.Hττ ≈ [270384.0094991878 141222.09367291332 -8217.083131009807 6282.077000013629 -44649.22033721553; 
#     141222.09367291332 343925.82167257956 -4682.034517776503 2718.9689339744064 -23296.71055040266; 
#     -8217.083131009807 -4682.034517776503 269242.2887936948 -285.69785992542654 1790.8346431986724; 
#     6282.077000013629 2718.9689339744064 -285.69785992542654 269958.7492758073 -1252.7672482838827; 
#     -44649.22033721553 -23296.71055040266 1790.8346431986724 -1252.7672482838827 277659.3121544409]
    
#     #for m = 6000 ni = 5:11
#     # @test mom_obj!(vlmm) ≈ 791226.9922616098
#     # @test isapprox(vlmm.∇τ, [1.91491267287347e-12, 7.859349229200063e-7, -8.247953231510508e-8, 6.274469528388948e-7, 9.676954570991825e-9], atol = 1e-6)
    
#     # @test isapprox(vlmm.∇Lγ, [-7.358558207215538e-13 -1.042943509332872e-12 -5.0177639820958575e-12; 2.887690087050032e-12 9.72333324966712e-13 -9.510725540451404e-13; -2.9167779302952113e-12 2.5307533846330443e-13 1.7472689961550714e-12], atol = 1e-9)
#     # @test vlmm.HLγLγ ≈ [10991.937106192028 -268.06955095398763 282.27465419294214 34.195298814610325 -3.707423184103097 61.08033296057797; -268.06955095398706 8878.750376129516 -5.495902664212522 -255.01667468037485 216.08240268674672 -2.6703680996921517; 282.2746541929416 -5.495902664211422 15253.825720383418 -0.6367875021323592 -212.9554587734356 569.9283750279841; 34.19529881461035 -255.01667468037414 -0.6367875021322729 17036.867248183946 198.48726893159358 118.62058596693457; -3.7074231841030425 216.0824026867464 -212.9554587734337 198.48726893159534 14596.670421746294 531.9271462112076; 61.08033296057808 -2.670368099692175 569.9283750279833 118.62058596693463 531.9271462112102 28819.00508808355]
#     # @test vlmm.HτLγ ≈ [252.46527872914322 -18.67403614864742 37.131726651917205 697.6477123219247 29.319602348149864 1444.8139967089444; 9.266871847914837 -1.9354519009171907 3.4089187263479066 71.14425928158474 -7.338744099325551 109.7435865456317; -2.5722969300898804 0.6728293138652596 7.113948955831991 -32.857368535663895 -6.798672061956019 -18.555565674594927; 34.6754941979307 -6.47698828574669 3.7428372802232768 78.33258551345322 10.710631173717756 183.85642388828467; 87.90322233482816 -4.625501078206309 10.018464674974416 231.17079987155535 7.50775181110256 488.3295481472321]
#     # @test vlmm.Hττ ≈ [37631.47724880125 19293.972998221176 -6154.394441195269 7178.524853730536 19297.145226314384; 19293.972998221176 38276.6319702 -1981.6494808910295 4468.047057536958 11995.928535901963; -6154.394441195269 -1981.6494808910295 32221.07485140185 -1159.1942172626075 -3667.8817492965672; 7178.524853730536 4468.047057536958 -1159.1942172626075 33345.235501947216 3665.469611470153; 19297.145226314384 11995.928535901963 -3667.8817492965672 3665.469611470153 41258.3024838933]

#     # Profile.clear()
#     # @profile @btime mom_obj!($vlmm, true, true, true)
#     # Profile.print(format=:flat)
#     bm = @benchmark mom_obj!($vlmm, true, true, true)
#     display(bm); println()
#     # Profile.clear()
#     # @profile @btime update_wtmat!(vlmm)
#     # Profile.print(format=:flat)
# end

@testset "fit! (start from LS fit)" begin
println(); println(); println()
for solver in [
    # KNITRO.KnitroSolver(outlev=3), # outlev 0-6
    # Ipopt.IpoptSolver(print_level = 5, watchdog_shortened_iter_trigger=3),# helped remedy, best number
    Ipopt.IpoptSolver(print_level = 0)
    # Ipopt.IpoptSolver(print_level = 3, hessian_approximation = "limited-memory"),
    # Ipopt.IpoptSolver(print_level = 3, obj_scaling_factor = 1 / m) # less accurae, grad at 10^{-1}
    # Ipopt.IpoptSolver(print_level = 3, mu_strategy = "adaptive") # same speek
    # Ipopt.IpoptSolver(print_level = 3, mehrotra_algorithm = "yes") # unstable
    # NLopt.NLoptSolver(algorithm = :LD_SLSQP, maxeval = 4000)
    # NLopt.NLoptSolver(algorithm = :LD_MMA, maxeval = 4000),
    # NLopt.NLoptSolver(algorithm = :LD_LBFGS, maxeval = 4000)
    # NLopt.NLoptSolver(algorithm = :LN_BOBYQA, ftol_rel = 1e-12, ftol_abs = 1e-8, maxeval = 10000)
    ]
    println("----------")
    @show solver
    println("----------")
    @info "fitting"
    @time VarLMM.fit!(vlmm, solver, runs=4)
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
    @info "Hessian at solution"
    @show vlmm.Hββ
    @show vlmm.HLγLγ
    @show vlmm.HτLγ
    @show vlmm.Hττ
    @info "inference at solution"
    show(vlmm)
    # BenchmarkTools.DEFAULT_PARAMETERS.seconds = 15
    # under m = 800, ni = 800:1000
    # @test mom_obj!(vlmm, true, true, true) ≈ 3.9331213647326386e8
    # @test isapprox(vlmm.∇Lγ, [-2.942432139024648e-14 -8.771293373745995e-15 -1.731001191734574e-14;
    #  -1.66797855963846e-14 1.368796656887832e-13 1.4264115589582018e-13;
    #  -3.7197885639546623e-14 1.9760068098839214e-13 1.3381970751486514e-13], atol = 1e-9)
    # @test isapprox(vlmm.∇τ, [1.63140612130519e-11, -8.898693550918324e-6, 
    # 8.625343451029721e-6, -1.2175564673810868e-5, 6.686529573585176e-6], atol = 1e-9)
    # @test vlmm.HLγLγ ≈ [1606.8212031880164 -1.0819248030650936 22.784856175380057 -0.11003556836724929 -0.1734625015321543 0.01887503882470629;
    #  -1.0819248030664714 1360.0932662911655 102.50990599123048 -21.73085966115454 0.00868466357737379 0.08737846866823742; 
    #  22.78485617537875 102.50990599122669 2821.014939401117 -0.7880012357814555 -22.60513282678334 2.4044485153821404;
    #   -0.11003556836724737 -21.73085966115631 -0.7880012357816188 2691.319229652556 97.44562964890231 -0.43582707025984946;
    #    -0.17346250153217233 0.008684663577271406 -22.605132826786953 97.44562964889863 2795.4163699159676 -11.99303109202686;
    #     0.018875038824687215 0.0873784686681497 2.4044485153797663 -0.43582707026003936 -11.9930310920323 5637.172602524909]
    # @test vlmm.HτLγ ≈ [0.08061709850319179 -0.0008442092590189238 0.00326481833580463 0.17919870962733725 0.012645821355870546 0.5372198431221124;
    #  -0.07663236550581361 0.0006977475632101063 -0.0019465597712954887 -0.16736277218087242 -0.011699788669026266 -0.5040026057826498; 0.036919585414603384 0.0004406785929035899 0.0013228101839111177 0.08144949059682167 0.0057235072103784534 0.24372520935333877; 0.0016268894405231718 0.00015276783041439499 -5.607528861476198e-5 0.002672489834683056 0.0005857851754622765 0.009032345832346383; -0.013251377781719333 -3.6558639768175196e-5 0.00016054956129225216 -0.029642707002518517 -0.0018825494928452783 -0.08552102257965982]
    # @test vlmm.Hττ ≈ [270384.0094991878 141222.09367291332 -8217.083131009807 6282.077000013629 -44649.22033721553; 
    # 141222.09367291332 343925.82167257956 -4682.034517776503 2718.9689339744064 -23296.71055040266; 
    # -8217.083131009807 -4682.034517776503 269242.2887936948 -285.69785992542654 1790.8346431986724; 
    # 6282.077000013629 2718.9689339744064 -285.69785992542654 269958.7492758073 -1252.7672482838827; 
    # -44649.22033721553 -23296.71055040266 1790.8346431986724 -1252.7672482838827 277659.3121544409]
    
    # # for m = 6000 ni = 5:11
    # @test mom_obj!(vlmm) ≈ 791226.9922616098
    # @test isapprox(vlmm.∇τ, [1.91491267287347e-12, 7.859349229200063e-7, -8.247953231510508e-8, 6.274469528388948e-7, 9.676954570991825e-9], atol = 1e-6)
    
    # @test isapprox(vlmm.∇Lγ, [-7.358558207215538e-13 -1.042943509332872e-12 -5.0177639820958575e-12; 2.887690087050032e-12 9.72333324966712e-13 -9.510725540451404e-13; -2.9167779302952113e-12 2.5307533846330443e-13 1.7472689961550714e-12], atol = 1e-9)
    # @test vlmm.HLγLγ ≈ [10991.937106192028 -268.06955095398763 282.27465419294214 34.195298814610325 -3.707423184103097 61.08033296057797; -268.06955095398706 8878.750376129516 -5.495902664212522 -255.01667468037485 216.08240268674672 -2.6703680996921517; 282.2746541929416 -5.495902664211422 15253.825720383418 -0.6367875021323592 -212.9554587734356 569.9283750279841; 34.19529881461035 -255.01667468037414 -0.6367875021322729 17036.867248183946 198.48726893159358 118.62058596693457; -3.7074231841030425 216.0824026867464 -212.9554587734337 198.48726893159534 14596.670421746294 531.9271462112076; 61.08033296057808 -2.670368099692175 569.9283750279833 118.62058596693463 531.9271462112102 28819.00508808355]
    # @test vlmm.HτLγ ≈ [252.46527872914322 -18.67403614864742 37.131726651917205 697.6477123219247 29.319602348149864 1444.8139967089444; 9.266871847914837 -1.9354519009171907 3.4089187263479066 71.14425928158474 -7.338744099325551 109.7435865456317; -2.5722969300898804 0.6728293138652596 7.113948955831991 -32.857368535663895 -6.798672061956019 -18.555565674594927; 34.6754941979307 -6.47698828574669 3.7428372802232768 78.33258551345322 10.710631173717756 183.85642388828467; 87.90322233482816 -4.625501078206309 10.018464674974416 231.17079987155535 7.50775181110256 488.3295481472321]
    # @test vlmm.Hττ ≈ [37631.47724880125 19293.972998221176 -6154.394441195269 7178.524853730536 19297.145226314384; 19293.972998221176 38276.6319702 -1981.6494808910295 4468.047057536958 11995.928535901963; -6154.394441195269 -1981.6494808910295 32221.07485140185 -1159.1942172626075 -3667.8817492965672; 7178.524853730536 4468.047057536958 -1159.1942172626075 33345.235501947216 3665.469611470153; 19297.145226314384 11995.928535901963 -3667.8817492965672 3665.469611470153 41258.3024838933]

    # Profile.clear()
    # @profile @btime mom_obj!($vlmm, true, true, true)
    # Profile.print(format=:flat)
    # bm = @benchmark mom_obj!($vlmm, true, true, true)
    # display(bm); println()
    # Profile.clear()
    # @profile @btime update_wtmat!(vlmm)
    # Profile.print(format=:flat)
end

end

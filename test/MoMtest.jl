module PolrfitTest

using Test, VarLMM


@testset "logit link" begin
    for solver in [IpoptSolver(print_level=0), NLoptSolver(algorithm=:LD_SLSQP)]
        @test 1 == 1
    end
end


end

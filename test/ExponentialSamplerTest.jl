using Rimu
using RimuRealTime
using Test

@testset "ExponentialSampler" begin
    H = MatrixHamiltonian([1;;])
    E = ExponentialSampler(H,1.0)
    v = DVec(1 => 1000.0; style=IsDynamicSemistochastic())
    names, values, wm, w = apply_operator!(working_memory(v), zerovector(v), v, E)
    @test w[1] ≈ 1000ℯ rtol=0.03

    add = BoseFS(2,0,0)
    H1 = HubbardReal1D(add; u=1.0)
    E = ExponentialSampler(H1, 1.0)
    @test E' == E

    E = ExponentialSampler(H1, 1.0im)
    @test E' == ExponentialSampler(H1, 0.0-1.0im)

    H2 = HubbardReal1D(add; u=1.0im)
    E = ExponentialSampler(H2, 1.0)
    @test E' == ExponentialSampler(H2', 1.0)

    E = ExponentialSampler(H2, 1.0im)
    @test E' == ExponentialSampler(H2', 0.0-1.0im)

    @test starting_address(E) == starting_address(H2)
    @test parent_operator(E) == H2

    @test !has_iterable_offdiagonals(ExponentialSampler)

    H3 = ExtendedHubbardReal1D(add; u=1.0, t = 0.0)
    @test LOStructure(ExponentialSampler(H1, 1.0)) == IsHermitian()
    @test LOStructure(ExponentialSampler(H2, 1.0)) == AdjointKnown()
    @test LOStructure(ExponentialSampler(H3, 1.0)) == IsDiagonal()

    E = ExponentialSampler(H1, 1.0)
    col = operator_column(E, add)
    @test parent_operator(col) == E
    @test starting_address(col) == add
 
    for _ in 1:1000
        a, p, v = random_offdiagonal(col)
        @test 0 < p <= 1
    end
end

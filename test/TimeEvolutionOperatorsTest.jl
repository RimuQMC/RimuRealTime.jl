using Rimu
using RimuRealTime
using Test

@testset "NthOrderTimeEvolution" begin
    add = BoseFS(2,0,0)
    H1 = HubbardReal1D(add)
    U1 = FirstOrderTimeEvolution(H1, 0.1)

    @test U1' == FirstOrderTimeEvolution(H1, -0.1)

    v = DVec(add => 1.0im)
    @test U1*v == v - im*0.1*(H1*v)

    H2 = HubbardReal1D(add; u=1.0im)
    U2 = FirstOrderTimeEvolution(H2, 0.1)
    @test U2' == FirstOrderTimeEvolution(H2', -0.1)

    U3 = FirstOrderTimeEvolution(H2, 0.1-0.01im)
    @test U3' == FirstOrderTimeEvolution(H2', -0.1-0.01im)

    U4 = NthOrderTimeEvolution(H1, 0.1, 2)
    v = DVec(add => 1.0im)
    @test U4*v ≈ v - im*0.1*(H1*v) - 0.5*0.01*(H1*(H1*v))

    H_diagonal = ExtendedHubbardReal1D(add; u=1.0, t=0.0)
    @test LOStructure(FirstOrderTimeEvolution(H_diagonal, 0.1)) == IsDiagonal()
    @test LOStructure(FirstOrderTimeEvolution{Rimu.AbstractHamiltonian}) == AdjointUnknown()
    @test LOStructure(U1) == AdjointKnown()

    @test NthOrderTimeEvolution(H1, 0.1, 0) == IdentityOperator()
    @test NthOrderTimeEvolution(H1, 0.1, -1) == ExponentialSampler(H1, -im * 0.1)
end

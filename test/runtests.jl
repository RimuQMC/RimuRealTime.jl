using Rimu
using RimuRealTime
using Test

@testset "RimuRealTime.jl" begin
    @testset "FirstOrderTimeEvolution" begin
        add = BoseFS(2,0,0)
        H1 = HubbardReal1D(add)
        U1 = FirstOrderTimeEvolution(H1, 0.1)

        @test U1' == FirstOrderTimeEvolution(H1, -0.1)

        v = DVec(add => 1.0im)
        @test apply_operator(U1, v) == v - im*0.1*apply_operator(H1, v)

        H2 = HubbardReal1D(add; u=1.0im)
        U2 = FirstOrderTimeEvolution(H2, 0.1)
        @test U2' == FirstOrderTimeEvolution(H2', -0.1)

    end

    @testset "Clock" begin
        add = BoseFS(2,0,0)
        H = HubbardReal1D(add)
        C = FirstOrderClock(H, 0.01, 10)

        @test LOStructure(C) == IsHermitian()
        @test starting_address(C) isa ClockAddress
        @test parent_operator(C) == H
        @test num_steps(C) == 10
        @test time_step(C) == 0.01
        @test starting_state(C) == DVec(add => 1.0)
        @test dimension(C) == 11*dimension(H)

        col = operator_column(C, ClockAddress(add, 0))
        @test diagonal_element(col) == 0.5

        ods = collect(offdiagonals(col))
        for _ in 1:10
            addr, prob, val = random_offdiagonal(col)
            @test (addr => val) in ods
        end
        
        @test diagonal_element(operator_column(C, ClockAddress(add, 5))) == 1
        @test diagonal_element(operator_column(C, ClockAddress(add, 10))) == 0.5

        p = ExactDiagonalizationProblem(C)
        result = solve(p)
        v = result.vectors[1]
        start = DVec(address(add) => v[add] for add in keys(v) if time_index(add) == 0)
        @test norm(start) ≈ abs(start[BoseFS(2, 0, 0)]) atol=10^-6

        U = FirstOrderTimeEvolution(H,0.01)
        @test start ≈ apply_operator(U', apply_operator(U, start)) atol=0.0002
    end

    @testset "ExponentialSampler" begin
        H = MatrixHamiltonian([1;;])
        E = ExponentialSampler(H,1.0)
        v = DVec(1 => 1000.0; style=IsDynamicSemistochastic())
        w = apply_operator(E, v)
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

    end
end

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
        @testset "Nth Order Clock" begin
            for N in 1:3
                add = BoseFS(2,0,0)
                H = HubbardReal1D(add)
                U = NthOrderTimeEvolution(H, 0.01, N)
                C = Clock(U, 10)

                @test LOStructure(C) == IsHermitian()
                @test starting_address(C) isa ClockAddress
                @test time_evolution_operator(C) == U
                @test num_steps(C) == 10
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
                vts = [DVec([address(add) => v[add] for add in keys(v) if time_index(add) == i]) for i in 0:10]
                start = vts[1]
                @test norm(start) ≈ abs(start[BoseFS(2, 0, 0)]) atol=10^-6

                @test start ≈ apply_operator(U', apply_operator(U, start)) atol=0.0002

                ops = [[ClockOperator(DensityMatrixDiagonal(1), t) for t in 0:10]; [ClockProjector(t) for t in 0:10]]
                replica_strategy = AllOverlaps(2; operator=ops)
                start_at = [DVec(100*v; style=IsDynamicSemistochastic{ComplexF64}()) for _ in 1:2]

                p = ProjectorMonteCarloProblem(C; target_walkers=2000, replica_strategy, last_step=10000, start_at)
                df = DataFrame(solve(p))

                e = shift_estimator(df; skip=5000, shift="shift_r1s1")
                @test e.mean ≈ result.values[1] atol=5*e.err

                density = rayleigh_replica_estimator(df; op_name="Op1", vec_name="Op12", skip=5000)
                @test density.f ≈ dot(start, DensityMatrixDiagonal(1), start)/dot(start, start) atol=5*abs(density.σ_f)
            end
        end
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

using Rimu
using RimuRealTime
using Test

@testset "Clock" begin
    @testset "ClockAddress" begin
        addc = BoseFS(2,0,0)
        addr = ClockAddress(addc, 5)
        addm = BoseFS{missing}(2,0,3)
        addrm = ClockAddress(addm, 5)
        @test fock_address(addr) == addc
        @test time_index(addr) == 5
        @test addr isa ClockAddress
        @test addr isa AbstractFockAddress
        @test allows_address_type(Clock(HubbardReal1D(addc), 10), typeof(addr))
        @test !allows_address_type(Clock(HubbardReal1D(addc), 10), BoseFS)
        @test num_components(addr) == num_components(addc)
        @test num_particles(addr) == num_particles(addc)
        @test ismissing(num_particles(typeof(addrm))) == true
        @test num_particles(addrm) == 5 == num_particles(addm)
        @test num_modes(addr) == num_modes(addc)
        @test num_modes_are_equal(addr)
        @test num_modes_check_equal(addr, addc) == num_modes_check_equal(addc) == 3
        @test maximum_mode_occupation(addr) == maximum_mode_occupation(addc)
    end
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
            vts = [DVec([fock_address(add) => v[add] for add in keys(v) if time_index(add) == i]) for i in 0:10]
            start = vts[1]
            @test norm(start) ≈ abs(start[BoseFS(2, 0, 0)]) atol=10^-6

            ops = [[ClockOperator(DensityMatrixDiagonal(1), t) for t in 0:10]; [clock_projector(t) for t in 0:10]]
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
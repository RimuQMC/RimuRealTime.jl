using Rimu
using RimuRealTime
using RimuRealTime: PECSingleState, RKSingleState, EulerSingleState, ProductSingleState
using SafeTestsets
using Test
using ExplicitImports: check_no_implicit_imports

@safetestset "ExplicitImports" begin
    using RimuRealTime
    using ExplicitImports
    @test check_no_implicit_imports(
        RimuRealTime; skip=(RimuRealTime, Base, Core)
    ) === nothing
end

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
end

@testset "QDStates" begin
    address = FermiFS(1,1,1,1,1,0,0,0,0,0)
    hamiltonian = ExtendedHubbardReal1D(address; v=-2)
    shift = solve(ExactDiagonalizationProblem(hamiltonian)).values[1]

    v = DVec(address => 1.0)
    PEC_state = PECSingleState(v, working_memory(v), "", hamiltonian, shift)
    @test PEC_state.state_vector == v
    @test PEC_state.state_vector !== v
    @test PEC_state.H_vector == hamiltonian*v - shift*v

    RK_state = RKSingleState(v, working_memory(v), "", hamiltonian, 0.01)
    @test RK_state.state_vector == v
    @test RK_state.state_vector !== v

    Euler_state = EulerSingleState(v, working_memory(v), "", hamiltonian, 0.01)
    @test Euler_state.state_vector == v
    @test Euler_state.state_vector !== v

    product_state = ProductSingleState(v, working_memory(v), "", hamiltonian, 0.01, 2)
    @test product_state.state_vector == v
    @test product_state.state_vector !== v
end

@testset "QuantumDynamicsProblem" begin
    address = FermiFS(1,1,1,1,0,0,0,0)
    hamiltonian = ExtendedHubbardReal1D(address; v=-2)
    shift = solve(ExactDiagonalizationProblem(hamiltonian)).values[1]
    replica_strategy = AllOverlaps(3)
    post_step_strategy = Projector(:G, DVec(address => 1.0))
    initial_walkers = 1000
    maximum_time = 1.0
    time_step = 0.01

    for evolution_strategy in [PEC(), Runge_Kutta(), Euler(), Product(2)]
        for alpha in [0.0, 0.01]
            for scaling_strategy in [NoScaling(), DynamicScaling(initial_walkers), ConstantScaling(0.1)]
                problem = QuantumDynamicsProblem(
                    hamiltonian;
                    shift,
                    time_step,
                    alpha,
                    maximum_time,
                    initial_walkers,
                    evolution_strategy,
                    replica_strategy,
                    post_step_strategy,
                    scaling_strategy
                )

                @test problem.algorithm == DiscretizedEvolution(; time_step_strategy=ConstantTimeStep(), evolution_strategy, scaling_strategy)
                @test problem.hamiltonian == hamiltonian
                @test num_replicas(problem) == 3
                @test eval(Meta.parse(repr(problem.simulation_plan))) == problem.simulation_plan

                sim = init(problem)
                @test sim.modified[] == false == sim.aborted[] == sim.success[]
                state = sim.state
                @test num_replicas(state) == 3
                tsp = state.time_step_parameters
                @test tsp.alpha == alpha
                @test typeof(tsp.time) == (alpha == 0.0 ? Float64 : ComplexF64)
                @test tsp.prev_walkers == initial_walkers

                sim = solve(problem)
                @test sim.modified == true
                @test sim.success == true
                @test Rimu.is_finalized(sim.report) == true
                sim = solve!(sim; maximum_time=2.0)

                df = DataFrame(sim)
                @test typeof(df.G_r1[end]) == ComplexF64
                @test real(df.time[end]) >= 2.0
                @test typeof(df.time[end]) == (alpha == 0.0 ? Float64 : ComplexF64)
            end
        end
    end

    time_step_strategy = WalkerControl()
    problem = QuantumDynamicsProblem(
        hamiltonian;
        shift,
        time_step,
        time_step_strategy,
        last_step=100,
        initial_walkers,
        replica_strategy,
        post_step_strategy
    )
    sim = solve(problem)
    df = DataFrame(sim)
    @test 0.0 <= df.alpha[end] <= pi/2
    @test df.time[end] isa ComplexF64

    style = IsDeterministic{ComplexF64}()
    problem = QuantumDynamicsProblem(
        hamiltonian;
        shift,
        time_step,
        last_step=100,
        start_at=address,
        style,
        evolution_strategy=Runge_Kutta(5)
    )
    sim1 = solve(problem)
    df1 = DataFrame(sim1)

    problem = QuantumDynamicsProblem(
        hamiltonian;
        shift,
        time_step,
        last_step=100,
        start_at=address,
        style,
        evolution_strategy=Runge_Kutta()
    )
    sim2 = solve(problem)
    df2 = DataFrame(sim2)

    @test sim1.state[1].state_vector != sim2.state[1].state_vector

    problem = QuantumDynamicsProblem(
        hamiltonian;
        start_at=DVec(address => 1.0+0.0im; style=IsDeterministic{ComplexF64}()),
        evolution_strategy=Runge_Kutta()
    )
    sim = init(problem)
    @test StochasticStyle(sim.state[1].state_vector) isa IsDeterministic
    @test sim.state.algorithm.evolution_strategy isa Runge_Kutta

    @test_throws ArgumentError QuantumDynamicsProblem(hamiltonian; start_at=DVec(address=>1.0))

    for evolution_strategy in [Euler(), Product(1)]
        problem = QuantumDynamicsProblem(
            hamiltonian;
            time_step,
            last_step=1,
            initial_walkers=1,
            evolution_strategy,
            style=IsDeterministic{ComplexF64}()
        )
        sim = solve(problem)
        vec = DVec(address => 1.0+0.0im; style=IsDeterministic{ComplexF64}())
        U = FirstOrderTimeEvolution(hamiltonian, time_step)
        @test sim.state[1].state_vector == U*vec
    end
    
    for evolution_strategy in [PEC(), Runge_Kutta(), Product(2)]
        problem = QuantumDynamicsProblem(
            hamiltonian;
            time_step,
            last_step=1,
            initial_walkers=1,
            evolution_strategy,
            style=IsDeterministic{ComplexF64}()
        )
        sim = solve(problem)
        vec = DVec(address => 1.0+0.0im; style=IsDeterministic{ComplexF64}())
        U = NthOrderTimeEvolution(hamiltonian, time_step, 2)
        @test sim.state[1].state_vector ≈ U*vec
    end
end

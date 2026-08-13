using Rimu
using RimuRealTime
using RimuRealTime: LeapfrogSingleState, PECSingleState, RKSingleState, EulerSingleState, ProductSingleState, ExactSingleState
using Test

@testset "QuantumDynamicsProblem" begin
    address = FermiFS(1,1,1,1,0,0,0,0)
    hamiltonian = ExtendedHubbardReal1D(address; v=-2)
    shift = solve(ExactDiagonalizationProblem(hamiltonian)).values[1]
    replica_strategy = AllOverlaps(3)
    post_step_strategy = Projector(:G, DVec(address => 1.0))
    initial_walkers = 1000
    maximum_time = 1.0
    time_step = 0.01

    for evolution_strategy in [PEC(), RungeKutta(), Euler(), Product(2)]
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

                @test problem.algorithms == ntuple(Returns(DiscretizedEvolution(; time_step_strategy=ConstantTimeStep(), evolution_strategy, scaling_strategy)), num_replicas(problem))
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
        evolution_strategy=RungeKutta(5)
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
        evolution_strategy=RungeKutta()
    )
    sim2 = solve(problem)
    df2 = DataFrame(sim2)

    @test sim1.state[1].state_vector != sim2.state[1].state_vector

    problem = QuantumDynamicsProblem(
        hamiltonian;
        start_at=DVec(address => 1.0+0.0im; style=IsDeterministic{ComplexF64}()),
        evolution_strategy=RungeKutta()
    )
    sim = init(problem)
    @test StochasticStyle(sim.state[1].state_vector) isa IsDeterministic
    @test first(sim.state.algorithms).evolution_strategy isa RungeKutta

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

    for evolution_strategy in [PEC(), RungeKutta(), Product(2)]
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
    problem1 = QuantumDynamicsProblem(
        hamiltonian;
        time_step,
        last_step=1,
        initial_walkers=1,
        evolution_strategy=ExactEvolution(),
        style=IsDeterministic{ComplexF64}()
    )
    sim1 = solve(problem1)

    problem2 = QuantumDynamicsProblem(
        hamiltonian;
        time_step,
        last_step=1,
        initial_walkers=1,
        evolution_strategy=ExactEvolution(tol=1e-14),
        style=IsDeterministic{ComplexF64}()
    )
    sim2 = solve(problem2)

    @test sim1.state[1].state_vector ≈ sim2.state[1].state_vector atol=1e-9

    problem_default_style = QuantumDynamicsProblem(
    hamiltonian;
    time_step,
    last_step=1,
    initial_walkers=1,
    evolution_strategy=ExactEvolution()
    )
    @test problem_default_style.style isa IsDeterministic{ComplexF64}

    shown = sprint(show, problem_default_style)
    @test occursin("QuantumDynamicsProblem with 1 replica(s):", shown)
    @test occursin("algorithm =", shown)
    @test occursin("hamiltonian =", shown)
    @test occursin("start_at =", shown)
    @test occursin("style =", shown)
    @test occursin("initial_walkers =", shown)
    @test occursin("initiator =", shown)
    @test occursin("threading =", shown)
    @test occursin("simulation_plan =", shown)
    @test occursin("replica_strategy =", shown)
    @test occursin("reporting_strategy =", shown)
    @test occursin("post_step_strategy =", shown)
    @test occursin("metadata =", shown)
    @test occursin("random_seed =", shown)

    problem_without_seed = QuantumDynamicsProblem(hamiltonian; random_seed=false)
    @test problem_without_seed.random_seed === nothing

    problem_with_seed = QuantumDynamicsProblem(hamiltonian; random_seed=123)
    @test problem_with_seed.random_seed == UInt64(123)

    problem_all_overlaps = QuantumDynamicsProblem(
        hamiltonian;
        replica_strategy=AllOverlaps(3),
        random_seed=false,
    )
    @test num_overlaps(problem_all_overlaps) == 3
    @test num_overlaps(problem_without_seed) == 0
end

@testset "LeapfrogDynamicScaling" begin
    address = FermiFS(1,1,1,1,0,0,0,0)
    hamiltonian = ExtendedHubbardReal1D(address; v=-2)
    shift = solve(ExactDiagonalizationProblem(hamiltonian)).values[1]
    initial_walkers = 100

    problem = QuantumDynamicsProblem(
        hamiltonian;
        shift,
        time_step=0.01,
        last_step=10,
        start_at=DVec(address => 1.0+0.0im; style=IsDeterministic{ComplexF64}()),
        evolution_strategy=Leapfrog(),
        scaling_strategy=DynamicScaling(initial_walkers)
    )
    sim = solve(problem)
    @test sim.success == true

    s_state = sim.state[1]
    @test s_state.current_scale[] != 1.0
    @test norm(s_state.state_vector, 1) ≈ initial_walkers rtol=1e-8
end

@testset "LeapfrogDeadPopulation" begin
    address = FermiFS(1,1,1,1,0,0,0,0)
    hamiltonian = ExtendedHubbardReal1D(address; v=-2)
    shift = solve(ExactDiagonalizationProblem(hamiltonian)).values[1]

    problem = QuantumDynamicsProblem(
        hamiltonian;
        shift,
        time_step=0.01,
        last_step=10,
        start_at=DVec(address => 1.0+0.0im; style=IsDeterministic{ComplexF64}()),
        evolution_strategy=Leapfrog()
    )
    sim = init(problem)
    s_state = sim.state[1]
    zerovector!(s_state.state_real)
    zerovector!(s_state.state_imag_staggered)
    zerovector!(s_state.state_imag_staggered_previous)

    @test_logs (:error, r"is dead\. Aborting\.") match_mode=:any solve!(sim)
end

@testset "LeapfrogComplex One Step" begin

    address = FermiFS(1,1,1,1,0,0,0,0)
    hamiltonian = ExtendedHubbardReal1D(address; v=-2)
    shift = solve(ExactDiagonalizationProblem(hamiltonian)).values[1]
    time_step = 0.01
    v = DVec(address => 1.0 + 0.5im; style=IsDeterministic{ComplexF64}())
    h_v = hamiltonian * v - shift * v

    problem = QuantumDynamicsProblem(
    	hamiltonian;
    	shift,
    	time_step,
    	last_step=1,
    	start_at=v,
    	evolution_strategy=LeapfrogComplex()
    )
    sim = solve(problem)
    @test sim.success == true
    s_state = sim.state[1]
    expected_staggered = v - im * time_step / 2 * h_v
    expected_vector = v - im * time_step * (hamiltonian * expected_staggered - shift * expected_staggered)

    @test s_state.state_vector_previous ≈ v
    @test s_state.state_staggered ≈ expected_staggered
    @test s_state.state_vector ≈ expected_vector
end
    
@testset "LeapfrogComplexDynamicScaling" begin
    address = FermiFS(1, 1, 1, 1, 0, 0, 0, 0)
    hamiltonian = ExtendedHubbardReal1D(address; v=-2)
    shift = solve(ExactDiagonalizationProblem(hamiltonian)).values[1]
    initial_walkers = 100
    problem = QuantumDynamicsProblem(
        hamiltonian;
        shift,
        time_step=0.01,
        last_step=10,
        start_at=DVec(address => 1.0 + 0.0im; style=IsDeterministic{ComplexF64}()),
        evolution_strategy=LeapfrogComplex(),
        scaling_strategy=DynamicScaling(initial_walkers),
    )
    sim = solve(problem)
    s_state = sim.state[1]
    @test sim.success == true
    @test s_state.current_scale[] != 1.0
    @test norm(s_state.state_vector, 1) ≈ initial_walkers rtol=1e-8
end

@testset "LeapfrogComplexDeadPopulation" begin
    address = FermiFS(1, 1, 1, 1, 0, 0, 0, 0)
    hamiltonian = ExtendedHubbardReal1D(address; v=-2)
    shift = solve(ExactDiagonalizationProblem(hamiltonian)).values[1]
    problem = QuantumDynamicsProblem(
        hamiltonian;
        shift,
        time_step=0.01,
        last_step=10,
        start_at=DVec(address => 1.0 + 0.0im; style=IsDeterministic{ComplexF64}()),
        evolution_strategy=LeapfrogComplex(),
    )
    sim = init(problem)
    s_state = sim.state[1]
    zerovector!(s_state.state_vector)
    zerovector!(s_state.state_staggered)
    zerovector!(s_state.state_vector_previous)

    @test_logs (:error, r"is dead\. Aborting\.") match_mode=:any solve!(sim)
end

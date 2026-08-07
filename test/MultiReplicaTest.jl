using Rimu
using RimuRealTime
using RimuRealTime: RKSingleState, PECSingleState
using Test

@testset "MultiReplicaAlgorithm" begin
    address = FermiFS(1,1,1,1,0,0,0,0)
    hamiltonian = ExtendedHubbardReal1D(address; v=-2)
    shift = solve(ExactDiagonalizationProblem(hamiltonian)).values[1]
    replica_strategy = AllOverlaps(2)
    evolution_strategy = (PEC(), RungeKutta())
    start_at = DVec(address => 1.0+0.0im; style=IsDeterministic{ComplexF64}())

    problem = QuantumDynamicsProblem(
        hamiltonian;
        shift,
        time_step=0.01,
        last_step=10,
        start_at,
        evolution_strategy,
        replica_strategy
    )

    @test length(problem.algorithms) == 2
    @test problem.algorithms[1].evolution_strategy isa PEC
    @test problem.algorithms[2].evolution_strategy isa RungeKutta

    sim = init(problem)
    @test sim.state[1] isa PECSingleState
    @test sim.state[2] isa RKSingleState
    @test sim.state[1].state_vector !== sim.state[2].state_vector
    @test first(sim.state.algorithms).evolution_strategy isa PEC

    sim = solve(problem)
    @test sim.success == true

    single_algorithm = DiscretizedEvolution(; time_step_strategy=ConstantTimeStep(), evolution_strategy=Euler(), scaling_strategy=NoScaling())
    problem2 = QuantumDynamicsProblem(
        hamiltonian;
        shift,
        time_step=0.01,
        last_step=1,
        start_at,
        algorithm=single_algorithm,
        replica_strategy
    )
    @test problem2.algorithms == ntuple(Returns(single_algorithm), 2)

    single_algorithm = DiscretizedEvolution(; time_step_strategy=ConstantTimeStep(), evolution_strategy=Euler(), scaling_strategy=NoScaling())
    problem2 = QuantumDynamicsProblem(
        hamiltonian;
        shift,
        time_step=0.01,
        last_step=1,
        start_at,
        algorithm=single_algorithm,
        replica_strategy
    )
    @test problem2.algorithms == ntuple(Returns(single_algorithm), 2)

    sim2 = init(problem2)
    @test sim2.algorithm isa NTuple{2, DiscretizedEvolution}

    problem_collection = QuantumDynamicsProblem(
        hamiltonian;
        shift,
        time_step=0.01,
        last_step=1,
        start_at=[deepcopy(start_at), deepcopy(start_at)],
        replica_strategy=AllOverlaps(2),
        evolution_strategy=Euler(),
    )
    sim_collection = init(problem_collection)
    @test length(sim_collection.state) == 2
    @test sim_collection.state[1].state_vector ≈ start_at

    @test_throws ArgumentError init(QuantumDynamicsProblem(
        hamiltonian; start_at=42, last_step=1, shift, time_step=0.01
    ))

    tsp = RimuRealTime.TimeStepParameters{Float64}(0.0, 100.0, 0.0, 0.05, 0.05)
    problem_tsp = QuantumDynamicsProblem(
        hamiltonian;
        shift,
        last_step=1,
        initial_walkers=1,
        evolution_strategy=Euler(),
        style=IsDeterministic{ComplexF64}(),
        initial_time_step_parameters=tsp,
    )
    sim_tsp = solve(problem_tsp)
    @test sim_tsp.state.time_step_parameters.time_step == 0.05
end

@testset "MultiReplicaStyle" begin
    address = FermiFS(1,1,1,1,0,0,0,0)
    hamiltonian = ExtendedHubbardReal1D(address; v=-2)
    shift = solve(ExactDiagonalizationProblem(hamiltonian)).values[1]
    replica_strategy = AllOverlaps(2)
    style = (IsDeterministic{ComplexF64}(), IsDynamicSemistochastic{ComplexF64}())

    problem = QuantumDynamicsProblem(
        hamiltonian;
        shift,
        time_step=0.01,
        last_step=1,
        initial_walkers=100,
        evolution_strategy=Euler(),
        replica_strategy,
        style
    )

    @test problem.style == style

    sim = init(problem)
    @test StochasticStyle(sim.state[1].state_vector) isa IsDeterministic
    @test StochasticStyle(sim.state[2].state_vector) isa IsDynamicSemistochastic
    @test sim.state[1].state_vector !== sim.state[2].state_vector
end
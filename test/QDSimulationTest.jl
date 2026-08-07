using Rimu
using RimuRealTime
using Test

@testset "QDSimulation" begin
    address = FermiFS(1, 1, 1, 1, 0, 0, 0, 0)
    hamiltonian = ExtendedHubbardReal1D(address; v=-2)
    shift = solve(ExactDiagonalizationProblem(hamiltonian)).values[1]
    start_at = DVec(address => 1.0 + 0.0im; style=IsDeterministic{ComplexF64}())
    sim = init(QuantumDynamicsProblem(
        hamiltonian;
        shift,
        start_at,
        time_step=0.01,
        last_step=1,
        evolution_strategy=Euler(),
    ))

    output = sprint(show, sim)
    @test occursin("QDSimulation with 1 replica(s).", output)
    @test occursin("Algorithm:", output)
    @test occursin("Hamiltonian:", output)
    @test occursin("Step:", output)
    @test occursin("Time:", output)
    @test sim.algorithm === sim.state.algorithms
    @test sim.hamiltonian === hamiltonian
    @test sim.df isa DataFrame

    post_step_strategy = Projector(:G, start_at)
    reporting_strategy = sim.state.reporting_strategy
    solve!(sim;
        maximum_time=0.1,
        last_step=0,
        wall_time=10.0,
        post_step_strategy,
        reporting_strategy,
        empty_report=true,
    )

    @test sim.state.simulation_plan.maximum_time == 0.1
    @test sim.state.simulation_plan.last_step == 0
    @test sim.state.simulation_plan.wall_time == 10.0
    @test num_replicas(sim.state.replica_strategy) == 1
    @test sim.state.post_step_strategy == (post_step_strategy,)
    @test sim.state.reporting_strategy === reporting_strategy
end
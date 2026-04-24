"""
    QDAlgorithm
Abstract type for quantum dynamics algorithms, for use with
[`QuantumDynamicsProblem`](@ref). Implemented algorithms: [`DiscretizedEvolution`](@ref).
"""
abstract type QDAlgorithm end

"""
    QDSimulationPlan(; starting_step = 0, last_step = Inf, wall_time = Inf, maximum_time = 1.0)
Defines the duration of the simulation. The simulation ends when the `last_step` is
reached, `wall_time` is exceeded, or `maximum_time` (length of real time evolution) is
reached.

See [`QuantumDynamicsProblem`](@ref), [`QDSimulation`](@ref).
"""
Base.@kwdef struct QDSimulationPlan
    starting_step::Int = 0
    last_step::Float64 = Inf
    wall_time::Float64 = Inf
    maximum_time::Float64 = 1.0
end
function Base.show(io::IO, plan::QDSimulationPlan)
    print(
        io, "QDSimulationPlan(starting_step=", plan.starting_step,
        ", last_step=", plan.last_step, ", wall_time=", plan.wall_time,
        ", maximum_time=", plan.maximum_time, ")"
    )
end

"""
    QuantumDynamicsProblem(hamiltonian; kwargs...)
Defines a problem for time evolution under the given `hamiltonian`.

# Keyword arguments:
- `time_step = 0.01`: Size of the time step. For complex time steps, this is the modulus;
    use keyword `alpha` to define the argument.
- `maximum_time = 1.0`: How long to evolve for in real time. Alternatively, set `last_step`
    to limit the number of time steps.
- `shift = 0.0`: Energy shift applied to the Hamiltonian. The state evolves under
    `H - shift*I`.
- `initial_walkers = 1000`: Initial walker population.
- `start_at = starting_address(hamiltonian)`: The initial state, as an address or an
    AbstractDVec.
- `style = IsDynamicSemistochastic{ComplexF64}()`: Stochastic style of the simulation.
- `initiator = false`: Whether to use initiators. Can be `true`, `false`, or a valid
    Rimu.InitiatorRule.
- `threading`: Default is to use multithreading and/or
    [MPI](https://juliaparallel.org/MPI.jl/latest/) if available. Set to
    `true` to force PDVec for the starting vector, `false` for serial computation;
    may be overridden by `start_at`.
- `evolution_strategy = PEC()`: Strategy for time evolution, see
    [`EvolutionStrategy`](@ref).
- `scaling_strategy = NoScaling`: Strategy for controlling walkers by scaling the vector,
    see [`ScalingStrategy`](@ref).
- `n_replicas = 1`: Number of synchronised independent simulations.
- `replica_strategy = NoStats(n_replicas)`: Which results to report from replica
    simulations. See Rimu.ReplicaStrategy.
- `reporting_strategy = ReportDFAndInfo()`: How and when to report results, see
    [`ReportingStrategy`](@ref).
- `post_step_strategy = ()`: Extract observables (e.g. Rimu.ProjectedEnergy), see
    Rimu.PostStepStrategy.
- `alpha = 0.0`: Initial phase angle of the time step.
- `time_step_strategy = ConstantTimeStep()`: Defines how the time step is updated during
    the simulation.
- `D = 0.1`: How strongly the time step phase angle is updated if the `time_step_strategy`
    used is [`WalkerControl`](@ref).
- `algorithm = DiscretizedEvolution(; time_step_strategy, evolution_strategy, scaling_strategy)`:
    The algorithm to use. Currently only [`DiscretizedEvolution`](@ref) is implemented.
- `starting_step = 0`: Starting step of the simulation.
- `wall_time = Inf`: Maximum time allowed for the simulation.
- `simulation_plan = QDSimulationPlan(; starting_step, last_step, wall_time, maximum_time)`:
    Defines the duration of the simulation. Takes precedence over `last_step`,
    `maximum_time`, and `wall_time`.
- `initial_time_step_parameters = (; abs_time_step, alpha, D)`.
- `display_name = "QDSimulation"`: Name displayed in progress bar (via `ProgressLogging`).
- `metadata`: User-supplied metadata to be added to the report. Must be an iterable of
    pairs or a `NamedTuple`, e.g. `metadata = ("key1" => "value1", "key2" => "value2")`.
    All metadata is converted to strings.
- `random_seed = true`: Provide and store a seed for the random number generator. If set to
    `true`, a new random seed is generated from `RandomDevice()`. If set to number, this
    number is used as the seed. This seed is used by `solve` (and `init`) to re-seed the
    default random number generator (consistently on each MPI rank) such that
    `solve`ing the same `QuantumDynamicsProblem` twice will yield identical results. If
    set to `false`, no seed is used and consecutive random numbers are used.
"""
struct QuantumDynamicsProblem{N}
    algorithm::QDAlgorithm
    hamiltonian::AbstractHamiltonian
    start_at
    shift::Union{Float64,ComplexF64}
    style::StochasticStyle
    initiator::InitiatorRule
    threading::Bool
    simulation_plan::QDSimulationPlan
    replica_strategy::ReplicaStrategy{N}
    initial_walkers::Float64
    initial_time_step_parameters
    reporting_strategy::ReportingStrategy
    post_step_strategy::Tuple
    metadata::LittleDict{String,String}
    random_seed::Union{Nothing,UInt64}
end

function Base.show(io::IO, p::QuantumDynamicsProblem)
    nr = num_replicas(p)
    println(io, "QuantumDynamicsProblem with $nr replica(s):")
    isnothing(p.algorithm) || println(io, "  algorithm = ", p.algorithm)
    println(io, "  hamiltonian = ", p.hamiltonian)
    println(io, "  start_at = ", p.start_at)
    println(io, "  style = ", p.style)
    println(io, "  initial_walkers = ", p.initial_walkers)
    println(io, "  initiator = ", p.initiator)
    println(io, "  threading = ", p.threading)
    println(io, "  simulation_plan = ", p.simulation_plan)
    println(io, "  replica_strategy = ", p.replica_strategy)
    print(io, "  reporting_strategy = ", p.reporting_strategy)
    println(io, "  post_step_strategy = ", p.post_step_strategy)
    println(io, "  metadata = ", p.metadata)
    print(io, "  random_seed = ", p.random_seed)
end

function QuantumDynamicsProblem(
    hamiltonian::AbstractHamiltonian;
    n_replicas = 1,
    start_at = starting_address(hamiltonian),
    shift = 0.0,
    style = IsDynamicSemistochastic{ComplexF64}(),
    initiator = false,
    threading = nothing,
    time_step = 0.01,
    starting_step = 0,
    last_step = Inf,
    maximum_time = 1.0,
    scaling_strategy = NoScaling(),
    wall_time = Inf,
    simulation_plan = nothing,
    replica_strategy = NoStats(n_replicas),
    initial_walkers = 1000.0,
    D = 0.1,
    alpha=0.0,
    time_step_strategy=ConstantTimeStep(),
    evolution_strategy=PEC(),
    algorithm=nothing,
    initial_time_step_parameters=nothing,
    reporting_strategy = ReportDFAndInfo(),
    post_step_strategy = (),
    metadata = nothing,
    display_name = "QDSimulation",
    random_seed = true
)
    if isnothing(simulation_plan)
        simulation_plan = QDSimulationPlan(
            starting_step,
            last_step,
            wall_time,
            maximum_time
        )
    end

    if isnothing(algorithm)
        algorithm = DiscretizedEvolution(;
            time_step_strategy,
            evolution_strategy,
            scaling_strategy
        )
    end

    n_replicas = num_replicas(replica_strategy)

    if random_seed == true
        random_seed = rand(RandomDevice(), UInt64)
    elseif random_seed == false
        random_seed = nothing
    elseif !isnothing(random_seed)
        random_seed = UInt64(random_seed)
    end

    if initiator isa Bool
        initiator = initiator ? Initiator() : NonInitiator()
    end

    if isnothing(threading)
        threading = Threads.nthreads() > 1
    end

    if isnothing(initial_time_step_parameters)
        abs_time_step = time_step
        initial_time_step_parameters = (; abs_time_step, alpha, D)
    end

    if scaling_strategy isa ConstantScaling
        shift += im*scaling_strategy.scale
    end

    report = Report()
    report_metadata!(report, "display_name", display_name)
    isnothing(metadata) || report_metadata!(report, metadata)
    metadata = report.meta::LittleDict{String, String}

    if post_step_strategy isa PostStepStrategy
        post_step_strategy = (post_step_strategy,)
    end

    if start_at isa AbstractDVec && valtype(start_at) <: Real || eltype(style) <: Real
        throw(ArgumentError(
            "The starting vector or stochastic style provided must allow complex values."
        ))
    end

    return QuantumDynamicsProblem{n_replicas}(
        algorithm,
        hamiltonian,
        start_at,
        shift,
        style,
        initiator,
        threading,
        simulation_plan,
        replica_strategy,
        initial_walkers,
        initial_time_step_parameters,
        reporting_strategy,
        post_step_strategy,
        metadata,
        random_seed,
    )
end

Rimu.num_replicas(::QuantumDynamicsProblem{N}) where {N} = N
function Rimu.num_overlaps(p::QuantumDynamicsProblem{N}) where {N}
    if p.replica_strategy isa AllOverlaps{N,<:Any,<:Any,true}
        return N*(N-1)÷2
    else
        return 0
    end
end

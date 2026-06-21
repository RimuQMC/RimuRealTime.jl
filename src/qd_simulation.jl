"""
    QDSimulation
Holds the state and the results of a quantum dynamics simulation.
Is returned by [`init(::QuantumDynamicsProblem)`](@ref) and solved with
[`solve!(::QDSimulation)`](@ref).

Obtain the results of a simulation `sm` as a DataFrame with `DataFrame(sm)`.

## Fields
- `problem::QuantumDynamicsProblem`: The problem that was solved
- `state::QDReplicaState`: The current state of the simulation
- `report::Report`: The report of the simulation
- `modified::Bool`: Whether the simulation has been modified
- `aborted::Bool`: Whether the simulation has been aborted
- `success::Bool`: Whether the simulation has been completed successfully
- `message::String`: A message about the simulation status
- `elapsed_time::Float64`: The wall time elapsed during the simulation

See also [`QuantumDynamicsProblem`](@ref), [`init`](@ref), [`solve!`](@ref).
"""
mutable struct QDSimulation
    problem::QuantumDynamicsProblem
    state::QDReplicaState
    report::Report
    modified::Bool
    aborted::Bool
    success::Bool
    message::String
    elapsed_time::Float64
end

function QDSimulation(problem::QuantumDynamicsProblem)

    @unpack algorithm, hamiltonian, start_at, style, threading, simulation_plan,
        replica_strategy, initial_time_step_parameters, initial_walkers, shift,
        reporting_strategy, post_step_strategy,
        metadata, initiator, random_seed = problem

    reporting_strategy = refine_reporting_strategy(reporting_strategy)

    n_replicas = num_replicas(replica_strategy)

    if !isnothing(random_seed)
        mpi_seed!(random_seed)
    end

    start_at = isnothing(start_at) ? starting_address(hamiltonian) : start_at
    if start_at isa AbstractDVec
        v = deepcopy(start_at)
    else
        v = default_starting_vector(start_at => initial_walkers; style, initiator, threading)
    end

    if initial_time_step_parameters isa NamedTuple
        @unpack abs_time_step, alpha = initial_time_step_parameters
        if !iszero(alpha) || algorithm.time_step_strategy isa WalkerControl
            K = ComplexF64
        else
            K = Float64
        end
        time = zero(K)
        time_step = K(abs_time_step*exp(-im*alpha))
        walkers = norm(v, 1)
        time_step_parameters = TimeStepParameters{K}(
            alpha,
            walkers,
            time,
            time_step,
            abs_time_step
        )
    end

    wm = working_memory(v)
    single_states = ntuple(n_replicas) do i
        id = if n_replicas == 1
            ""
        else
            "_r$(i)"
        end
        if algorithm.evolution_strategy isa Leapfrog
            LeapfrogSingleState(v, wm, id, hamiltonian, shift, time_step)
        elseif algorithm.evolution_strategy isa PEC
            PECSingleState(v, wm, id, hamiltonian, shift, algorithm.evolution_strategy.damping)
        elseif algorithm.evolution_strategy isa RungeKutta
            RKSingleState(v, wm, id, hamiltonian, time_step, algorithm.evolution_strategy.damping)
        elseif algorithm.evolution_strategy isa Euler
            EulerSingleState(v, wm, id, hamiltonian, time_step)
        elseif algorithm.evolution_strategy isa Product
            ProductSingleState(v, wm, id, hamiltonian, time_step, algorithm.evolution_strategy.order)
        end
    end
    @assert single_states isa NTuple{n_replicas, <:QDSingleState}

    state = QDReplicaState(
        single_states,
        time_step_parameters,
        shift,
        hamiltonian,
        algorithm,
        Ref(simulation_plan.starting_step),
        simulation_plan,
        reporting_strategy,
        post_step_strategy,
        replica_strategy
    )
    report = Report()
    report_default_metadata!(report, state)
    metadata!(report, metadata)

    return QDSimulation(
        problem, state, report, false, false, false, "", 0.0
    )
end

function Base.show(io::IO, sm::QDSimulation)
    print(io, "QDSimulation")
    st = sm.state
    print(io, " with ", num_replicas(st), " replica(s).")
    print(io, "\n  Algorithm:   ", sm.algorithm)
    print(io, "\n  Hamiltonian: ", sm.hamiltonian)
    print(io, "\n  Step:        ", st.step[])
    print(io, "\n  Time:        ", st.time_step_parameters.time)
    print(io, "\n  modified = $(sm.modified), aborted = $(sm.aborted), success = $(sm.success)")
    sm.message == "" || print(io, "\n  message: ", sm.message)
end

function report_simulation_status_metadata!(report::Report, sm::QDSimulation)
    @unpack modified, aborted, success, message, elapsed_time = sm

    metadata!(report, "modified", modified)
    metadata!(report, "aborted", aborted)
    metadata!(report, "success", success)
    metadata!(report, "message", message)
    metadata!(report, "elapsed_time", elapsed_time)
    return report
end

function Base.getproperty(sm::QDSimulation, key::Symbol)
    if key == :df
        return DataFrame(sm)
    elseif key == :algorithm
        return sm.state.algorithm
    elseif key == :hamiltonian
        return sm.state.hamiltonian
    else
        return getfield(sm, key)
    end
end

DataFrames.DataFrame(s::QDSimulation) = DataFrame(s.report)

"""
    init(problem::QuantumDynamicsProblem)::QDSimulation

Initialise a [`QDSimulation`](@ref).

See also [`QuantumDynamicsProblem`](@ref), [`solve!`](@ref), [`solve`](@ref),
[`step!`](@ref), [`QDSimulation`](@ref).
"""
function CommonSolve.init(problem::QuantumDynamicsProblem)
    return QDSimulation(problem)
end

"""
    step!(sm::QDSimulation)::QDSimulation

Advance the simulation by one step.

Calling [`solve!`](@ref) will advance the simulation until the maximum time is reached, the
last step is completed, or the wall time limit is reached. When completing the simulation
without calling [`solve!`](@ref), the simulation report needs to be finalised by calling
`Rimu.finalize_report!`.

See also [`QuantumDynamicsProblem`](@ref), [`init`](@ref), [`solve!`](@ref),
[`solve`](@ref), [`QDSimulation`](@ref).
"""
function CommonSolve.step!(sm::QDSimulation)
    @unpack state, report, algorithm = sm
    @unpack single_states, time_step_parameters, simulation_plan, step, reporting_strategy,
        replica_strategy = state
    @unpack time_step_strategy = algorithm

    if sm.aborted || sm.success
        @warn "Simulation is already aborted or finished."
        return sm
    end
    if step[] >= simulation_plan.last_step
        @warn "Simulation has already reached the last step."
        return sm
    end

    step[] += 1

    if step[] % reporting_interval(reporting_strategy) == 0
        report!(reporting_strategy, step[], report, :step, step[])
    end

    proceed = true
    for replica in single_states
        proceed &= advance!(report, state, replica)
    end
    sm.modified = true

    time_step_stats = update_time_step!(
        time_step_strategy,
        time_step_parameters,
        norm(single_states[1].state_vector, 1)
    )

    if step[] % reporting_interval(state.reporting_strategy) == 0
        report!(reporting_strategy, step[], report, time_step_stats)

        replica_names, replica_values = replica_stats(replica_strategy, single_states)
        report!(reporting_strategy, step[], report, replica_names, replica_values)
        report_after_step!(reporting_strategy, step[], report, state)
        ensure_correct_lengths(report)
    end

    if !proceed
        sm.aborted = true
        sm.message = "Aborted in step $(step[])."
        return sm
    end
    @unpack time = time_step_stats
    if step[] >= simulation_plan.last_step || real(time) >= simulation_plan.maximum_time
        sm.success = true
    end
    return sm
end

"""
    solve(::QuantumDynamicsProblem)::QDSimulation

Initialize and solve a [`QuantumDynamicsProblem`](@ref) until the maximum time is reached,
the last step is completed, or the wall time limit is reached.

See also [`init`](@ref), [`solve!`](@ref), [`step!`](@ref), [`QDSimulation`](@ref).
"""
CommonSolve.solve

"""
    solve!(sm::QDSimulation; kwargs...)::QDSimulation

Solve a [`QDSimulation`](@ref) until the maximum time is reached, the last step is
completed, or the wall time limit is reached.

To continue a previously completed simulation, set a new `maximum_time`, `last_step`, or
`wall_time` using the keyword arguments. Optionally, changes can be made to the
`replica_strategy`, the `post_step_strategy`, or the `reporting_strategy`.

# Optional keyword arguments:
* `maximum_time = nothing`: Set the maximum time to a new value and continue the
    simulation.
* `last_step = nothing`: Set the last step to a new value and continue the simulation.
* `wall_time = nothing`: Set the allowed wall time to a new value and continue the
    simulation.
* `reset_time = false`: Reset the `elapsed_time` counter and continue the simulation.
* `empty_report = false`: Empty the report before continuing the simulation.
* `replica_strategy = nothing`: Change the replica strategy. Requires the number of replicas
    to match the number of replicas in the simulation `sm`. Implies `empty_report = true`.
* `post_step_strategy = nothing`: Change the post-step strategy. Implies
    `empty_report = true`.
* `reporting_strategy = nothing`: Change the reporting strategy. Implies
    `empty_report = true`.
* `metadata = nothing`: Add metadata to the report.

See also [`QuantumDynamicsProblem`](@ref), [`init`](@ref), [`solve`](@ref),
[`step!`](@ref), [`QDSimulation`](@ref).
"""
function CommonSolve.solve!(sm::QDSimulation;
    maximum_time = nothing,
    last_step = nothing,
    wall_time = nothing,
    reset_time = false,
    replica_strategy=nothing,
    post_step_strategy=nothing,
    reporting_strategy=nothing,
    empty_report=false,
    metadata=nothing,
    display_name=nothing,
)
    reset_flags = reset_time
    if !isnothing(maximum_time)
        state = sm.state
        sm.state = @set state.simulation_plan.maximum_time = maximum_time
        reset_flags = true
    end
    if !isnothing(last_step)
        state = sm.state
        sm.state = @set state.simulation_plan.last_step = last_step
        metadata!(sm.report, "laststep", last_step)
        reset_flags = true
    end
    if !isnothing(wall_time)
        state = sm.state
        sm.state = @set state.simulation_plan.wall_time = wall_time
        reset_flags = true
    end
    if !isnothing(replica_strategy)
        if num_replicas(sm) ≠ num_replicas(replica_strategy)
            throw(ArgumentError("Number of replicas in the strategy must match the number of replicas in the simulation."))
        end
        state = sm.state
        sm.state = @set state.replica_strategy = replica_strategy
        reset_flags = true
        empty_report = true
    end
    if !isnothing(post_step_strategy)
        if post_step_strategy isa PostStepStrategy
            post_step_strategy = (post_step_strategy,)
        end
        state = sm.state
        sm.state = @set state.post_step_strategy = post_step_strategy
        reset_flags = true
        empty_report = true
    end
    if !isnothing(reporting_strategy)
        state = sm.state
        sm.state = @set state.reporting_strategy = reporting_strategy
        reset_flags = true
    end

    @unpack report = sm
    if empty_report
        empty!(report)
        report_default_metadata!(report, sm.state)
    end
    isnothing(metadata) || metadata!(report, metadata)
    isnothing(display_name) || metadata!(report, "display_name", display_name)

    @unpack simulation_plan, step, reporting_strategy, time_step_parameters = sm.state

    last_step = simulation_plan.last_step
    initial_step = step[]
    maximum_time = simulation_plan.maximum_time

    if step[] >= last_step
        @warn "Simulation has already reached the last step."
        return sm
    end

    if real(time_step_parameters.time) >= maximum_time
        @warn "Simulation has already reached the maximum time."
        return sm
    end

    if reset_flags
        sm.aborted = false
        sm.success = false
        sm.message = ""
    end
    if reset_time
        sm.elapsed_time = 0.0
    end

    if sm.aborted || sm.success
        @warn "Simulation is already aborted or finished."
        return sm
    end
    un_finalize!(report)

    starting_time = time() + sm.elapsed_time
    time_est = maximum_time - real(time_step_parameters.time)
    last_step_est = min(last_step, step[] + time_est/real(time_step_parameters.time_step))
    update_steps = max((last_step_est - initial_step) ÷ 200, 100)
    name = Rimu.metadata(sm.report, "display_name")

    @withprogress name = while !sm.aborted && !sm.success
        if time() - starting_time > simulation_plan.wall_time
            sm.aborted = true
            sm.message = "Wall time limit reached."
            @warn "Wall time limit reached. Aborting simulation."
        else
            step!(sm)
        end
        if step[] % update_steps == 0
            @logprogress (step[] - initial_step) / (last_step_est - initial_step)
        end

    end
    sm.elapsed_time = time() - starting_time
    report_simulation_status_metadata!(report, sm)
    finalize_report!(reporting_strategy, report)
    return sm
end

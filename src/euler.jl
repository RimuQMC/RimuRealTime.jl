"""
    Euler() <: EvolutionStrategy
[`EvolutionStrategy`](@ref) for evolution using the first-order Euler method. In each step
the state is updated according to ``v_{n+1} = (1 - i H dt)v_n``.
"""
struct Euler <: EvolutionStrategy end

"""
    EulerSingleState(v, wm, id, hamiltonian, time_step) <: QDSingleState
Struct holding state vector and other vectors required for [`Euler`](@ref) time evolution.
See [`QDReplicaState`](@ref).
"""
mutable struct EulerSingleState{V,W,U} <: QDSingleState
    state_vector::V
    previous_vector::V
    working_mem::W
    evolution_operator::U
    id::String
    current_scale::Float64
end
function EulerSingleState(v, wm, id, hamiltonian, time_step)
    state_vector = deepcopy(v)
    previous_vector = zerovector(v)
    working_mem = wm isa PDWorkingMemory ? wm : working_memory(v)
    evolution_operator = FirstOrderTimeEvolution(hamiltonian, time_step)
    current_scale = 1.0
    return EulerSingleState(
        state_vector,
        previous_vector,
        working_mem,
        evolution_operator,
        id,
        current_scale
    )
end

function advance!(report, state::QDReplicaState, s_state::EulerSingleState)

    @unpack state_vector, previous_vector, evolution_operator, working_mem, id,
        current_scale = s_state
    @unpack time_step_parameters, shift, hamiltonian, reporting_strategy, algorithm = state
    @unpack time_step = time_step_parameters
    @unpack time_step_strategy, scaling_strategy = algorithm
    step = state.step[]

    step_stat_names, step_stat_values, working_mem, previous_vector = apply_operator!(
        working_mem, previous_vector, state_vector, evolution_operator
    )
    add!(previous_vector, state_vector, im*shift*time_step)
    state_vector, previous_vector = previous_vector, state_vector

    scale_names, scale_stats, current_scale = scale_state_vector!(
        scaling_strategy, state_vector, current_scale
    )

    comp_name = CompressionStrategy(state_vector) isa NoCompression ? () : (:len_before_compression,)
    comp_stat = compress!(state_vector)
    names = (step_stat_names..., comp_name..., scale_names...)
    stats = (step_stat_values..., comp_stat..., scale_stats...)

    if !(time_step_strategy isa ConstantTimeStep)
        evolution_operator = FirstOrderTimeEvolution(hamiltonian, time_step)
    end

    @pack! s_state = state_vector, previous_vector, evolution_operator, working_mem, id,
        current_scale

    if step % reporting_interval(reporting_strategy) == 0
        walkers, len = walkernumber_and_length(state_vector)

        report!(reporting_strategy, step, report, (; len), id)
        report!(reporting_strategy, step, report, (; walkers), id)

        report!(reporting_strategy, step, report, names, stats, id)

        post_step_stats = post_step_action(state.post_step_strategy, s_state, step)
        report!(reporting_strategy, step, report, post_step_stats, id)

        if len == 0
            @error "Population in state $(s_state.id) is dead. Aborting."
            return false
        end
    end

    return true
end

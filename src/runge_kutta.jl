"""
    RungeKutta(damping=0) <: EvolutionStrategy
[`EvolutionStrategy`](@ref) for evolution using a second-order Runge-Kutta algorithm. In
each step the state is updated according to ``v_{n+1} = v_n + u_1 u_2 v_n - u_2 v_n``,
where ``u1 = 1 - i H dt`` and ``u2 = 1 - (1 + d) i H dt / 2``,
and ``d`` is the `damping` coefficient that modifies the second-order term. Second-order
damping can counteract the effects of large spectral components in the Hamiltonian that may
lead to an unphysical growth of the 2-norm of the state vector.
"""
Base.@kwdef struct RungeKutta <: EvolutionStrategy
    damping::Float64 = 0
end

"""
    RKSingleState(v, wm, id, hamiltonian, time_step, damping) <: QDSingleState
Struct holding state vector and other vectors required for [`RungeKutta`](@ref) time
evolution. See [`QDReplicaState`](@ref).
"""
mutable struct RKSingleState{V,W,U} <: QDSingleState
    state_vector::V
    storage_vector_1::V # stores half_evolution_operator * state_vector
    storage_vector_2::V # stores evolution_operator * storage_vector_1
    working_mem::W
    evolution_operator::U
    half_evolution_operator::U
    id::String
    damping::Float64
    current_scale::Float64
end
function RKSingleState(v, wm, id, hamiltonian, time_step, damping=0.0)
    state_vector = deepcopy(v)
    storage_vector_1 = zerovector(v)
    storage_vector_2 = zerovector(v)
    working_mem = wm isa PDWorkingMemory ? wm : working_memory(v)
    evolution_operator = FirstOrderTimeEvolution(hamiltonian, time_step)
    half_evolution_operator = FirstOrderTimeEvolution(hamiltonian, time_step*(damping + 1)/2)
    current_scale = 1.0
    return RKSingleState(
        state_vector,
        storage_vector_1,
        storage_vector_2,
        working_mem,
        evolution_operator,
        half_evolution_operator,
        id,
        damping,
        current_scale
    )
end

function advance!(report, state::QDReplicaState, s_state::RKSingleState)

    @unpack state_vector, storage_vector_1, storage_vector_2, working_mem,
        evolution_operator, half_evolution_operator, id, damping, current_scale = s_state
    @unpack time_step_parameters, shift, hamiltonian, reporting_strategy, algorithm = state
    @unpack time_step = time_step_parameters
    @unpack  time_step_strategy, scaling_strategy = algorithm
    step = state.step[]

    a, b, working_mem, storage_vector_1 = apply_operator!(
        working_mem, storage_vector_1, state_vector, half_evolution_operator
    )
    add!(storage_vector_1, state_vector, im*shift*time_step/2)
    a, b, working_mem, storage_vector_2 = apply_operator!(
        working_mem, storage_vector_2, storage_vector_1, evolution_operator
    )
    add!(add!(state_vector, storage_vector_2), storage_vector_1, im*shift*time_step - 1)

    scale_names, scale_stats, current_scale = scale_state_vector!(
        scaling_strategy, state_vector, current_scale
    )
    
    comp_name = CompressionStrategy(state_vector) isa NoCompression ? () : (:len_before_compression,)
    comp_stat = compress!(state_vector)
    names = (comp_name..., scale_names...)
    stats = (comp_stat..., scale_stats...)

    if !(time_step_strategy isa ConstantTimeStep)
        evolution_operator = FirstOrderTimeEvolution(hamiltonian, time_step)
        half_evolution_operator = FirstOrderTimeEvolution(hamiltonian, time_step*(damping+1)/2)
    end

    @pack! s_state = state_vector, storage_vector_1, storage_vector_2, working_mem,
        evolution_operator, half_evolution_operator, id, current_scale

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

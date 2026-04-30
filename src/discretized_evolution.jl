"""
    DiscretizedEvolution(; kwargs...) <: QDAlgorithm

Algorithm used to solve a [`QuantumDynamicsProblem`](@ref) by evolving a state in time
using an approximation to the exponential time evolution operator over many steps. Whether
the evolution is deterministic or stochastic is controlled by the keyword arguments `style`
or `start_at` passed to [`QuantumDynamicsProblem`](@ref).

# Keyword arguments:
- `time_step_strategy = ConstantTimeStep()`: How to update the time step.
- `evolution_strategy = PEC()`: Which [`EvolutionStrategy`](@ref) to use to evolve the
  state.
- `scaling_strategy = NoScaling()`: Which [`ScalingStrategy`](@ref) to use to control the
  walker number.
"""
Base.@kwdef struct DiscretizedEvolution{
    TS<:TimeStepStrategy, ES<:EvolutionStrategy, SS<:ScalingStrategy
} <: QDAlgorithm
    time_step_strategy::TS = ConstantTimeStep()
    evolution_strategy::ES = PEC()
    scaling_strategy::SS = NoScaling()
end
function Base.show(io::IO, a::DiscretizedEvolution)
    print(io, "DiscretizedEvolution($(a.time_step_strategy), $(a.evolution_strategy), $(a.scaling_strategy))")
end

"""
    advance!(report, state::QDReplicaState, s_state::QDSingleState)
Advance the state `s_state` by one step, and write data to the `report`.
"""
function advance!(report, state::QDReplicaState, s_state::PECSingleState)

    @unpack state_vector, copy_vector, H_vector, H_vector_new, storage_vector, working_mem,
        id, damping, current_scale = s_state
    @unpack time_step_parameters, shift, hamiltonian, reporting_strategy, algorithm = state
    @unpack time_step = time_step_parameters
    @unpack scaling_strategy = algorithm
    step = state.step[]

    copy_vector = add!(zerovector!(copy_vector), state_vector)
    add!(copy_vector, H_vector, - im*time_step)
    step_stat_names, step_stat_values, working_mem, H_vector_new = apply_operator!(
        working_mem, storage_vector, copy_vector, hamiltonian
    )
    add!(H_vector_new, copy_vector, -shift)
    add!(state_vector, add!(H_vector, H_vector_new, 1+damping, 1-damping), -0.5*im*time_step)
    H_vector, storage_vector = H_vector_new, H_vector
    
    if scaling_strategy isa DynamicScaling
        walkers_prev = norm(state_vector,1)
        scale_names = (:walkers_before_scaling, :scale,)
        scale!(state_vector, scaling_strategy.target_walkers/walkers_prev)
        current_scale *= scaling_strategy.target_walkers/walkers_prev
        scale_stats = (walkers_prev, current_scale,)
    else
        scale_names = ()
        scale_stats = ()
    end
    comp_name = CompressionStrategy(state_vector) isa NoCompression ? () : (:len_before_compression,)
    comp_stat = compress!(state_vector)
    names = (step_stat_names..., comp_name..., scale_names...)
    stats = (step_stat_values..., comp_stat..., scale_stats...)
    @pack! s_state = state_vector, copy_vector, H_vector, H_vector_new, storage_vector,
        working_mem, current_scale

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

    if scaling_strategy isa DynamicScaling
        walkers_prev = norm(state_vector,1)
        scale_names = (:walkers_before_scaling, :scale,)
        scale!(state_vector, scaling_strategy.target_walkers/walkers_prev)
        current_scale *= scaling_strategy.target_walkers/walkers_prev
        scale_stats = (walkers_prev, current_scale,)
    else
        scale_names = ()
        scale_stats = ()
    end
    comp_name = CompressionStrategy(state_vector) isa NoCompression ? () : (:len_before_compression,)
    comp_stat = compress!(state_vector)
    names = (step_stat_names..., comp_name..., scale_names...)
    stats = (step_stat_values..., comp_stat..., scale_stats...)

    if !(time_step_strategy isa ConstantTimeStep)
        evolution_operator = FirstOrderTimeEvolution(hamiltonian, time_step)
    end

    @pack! s_state = state_vector, previous_vector, evolution_operator, working_mem, id, current_scale

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

function advance!(report, state::QDReplicaState, s_state::ProductSingleState)

    @unpack state_vector, previous_vector, evolution_operator, working_mem, id, order,
        current_scale = s_state
    @unpack time_step_parameters, shift, hamiltonian, reporting_strategy, algorithm = state
    @unpack time_step = time_step_parameters
    @unpack time_step_strategy, scaling_strategy = algorithm
    step = state.step[]

    step_stat_names, step_stat_values, working_mem, previous_vector = apply_operator!(
        working_mem, previous_vector, state_vector, evolution_operator
    )
    state_vector, previous_vector = previous_vector, state_vector

    if scaling_strategy isa DynamicScaling
        walkers_prev = norm(state_vector,1)
        scale_names = (:walkers_before_scaling, :scale,)
        scale!(state_vector, scaling_strategy.target_walkers/walkers_prev)
        current_scale *= scaling_strategy.target_walkers/walkers_prev
        scale_stats = (walkers_prev, current_scale,)
    else
        scale_names = ()
        scale_stats = ()
    end
    comp_name = CompressionStrategy(state_vector) isa NoCompression ? () : (:len_before_compression,)
    comp_stat = compress!(state_vector)
    names = (step_stat_names..., comp_name..., scale_names...)
    stats = (step_stat_values..., comp_stat..., scale_stats...)

    if !(time_step_strategy isa ConstantTimeStep)
        evolution_operator = NthOrderTimeEvolution(hamiltonian, time_step, order)
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

    if scaling_strategy isa DynamicScaling
        walkers_prev = norm(state_vector,1)
        scale_names = (:walkers_before_scaling, :scale,)
        scale!(state_vector, scaling_strategy.target_walkers/walkers_prev)
        current_scale *= scaling_strategy.target_walkers/walkers_prev
        scale_stats = (walkers_prev, current_scale,)
    else
        scale_names = ()
        scale_stats = ()
    end
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

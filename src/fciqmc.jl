"""
    CFCIQMC(; kwargs...) <: QDAlgorithm
Algorithm for FCIQMC in real or complex time.

#Keyword arguments:
- `time_step_strategy = ConstantTimeStep()`: How to update the time step.
- `evolution_strategy = PEC()`: Which [`EvolutionStrategy`](@ref) to use to evolve the
  state.
"""
Base.@kwdef struct CFCIQMC{TS<:TimeStepStrategy, ES<:EvolutionStrategy} <: QDAlgorithm
    time_step_strategy::TS = ConstantTimeStep()
    evolution_strategy::ES = PEC()
end
function Base.show(io::IO, a::CFCIQMC)
    print(io, "CFCIQMC($(a.time_step_strategy), $(a.evolution_strategy))")
end

"""
    advance!(report, state::QDReplicaState, s_state::QDSingleState)
Advance the state `s_state` by one step, and write data to the `report`.
"""
function advance!(report, state::QDReplicaState, s_state::PECSingleState)

    @unpack v, w, Hw, Hw_new, x, wm, id = s_state
    @unpack time_step_parameters, shift, hamiltonian, reporting_strategy = state
    @unpack time_step = time_step_parameters
    step = state.step[]

    w = add!(zerovector!(w), v)
    add!(w, Hw, - im*time_step)
    step_stat_names, step_stat_values, wm, Hw_new = apply_operator!(wm, x, w, hamiltonian)
    add!(Hw_new, w, -shift)
    add!(v, add!(Hw, Hw_new), -0.5*im*time_step)
    Hw, x = Hw_new, Hw
    comp_name = CompressionStrategy(v) isa NoCompression ? () : (:len_before_compression,)
    comp_stat = compress!(v)
    names = (step_stat_names..., comp_name...)
    stats = (step_stat_values..., comp_stat...)
    @pack! s_state = v, w, Hw, Hw_new, x, wm

    if step % reporting_interval(reporting_strategy) == 0
        walkers, len = walkernumber_and_length(v)

        report!(reporting_strategy, step, report, (; len), id)
        report!(reporting_strategy, step, report, (; walkers), id)

        report!(reporting_strategy, step, report, names, stats, id)
        
        post_step_stats = Rimu.post_step_action(state.post_step_strategy, s_state, step)
        report!(reporting_strategy, step, report, post_step_stats, id)

        if len == 0
            @error "Population in state $(s_state.id) is dead. Aborting."
            return false
        end
    end

    return true
end

function advance!(report, state::QDReplicaState, s_state::EulerSingleState)

    @unpack v, pv, u, wm, id = s_state
    @unpack time_step_parameters, shift, hamiltonian, time_step_strategy,
        reporting_strategy = state
    @unpack time_step = time_step_parameters
    step = state.step[]

    step_stat_names, step_stat_values, wm, pv = apply_operator!(wm, pv, v, u)
    add!(pv, v, im*shift*time_step)
    v, pv = pv, v
    comp_name = CompressionStrategy(v) isa NoCompression ? () : (:len_before_compression,)
    comp_stat = compress!(v)
    names = (step_stat_names..., comp_name...)
    stats = (step_stat_values..., comp_stat...)

    if !(time_step_strategy isa ConstantTimeStep)
        u = FirstOrderTimeEvolution(hamiltonian, time_step)
    end

    @pack! s_state = v, pv, u, wm, id

    if step % reporting_interval(reporting_strategy) == 0
        walkers, len = walkernumber_and_length(v)

        report!(reporting_strategy, step, report, (; len), id)
        report!(reporting_strategy, step, report, (; walkers), id)

        report!(reporting_strategy, step, report, names, stats, id)

        post_step_stats = Rimu.post_step_action(state.post_step_strategy, s_state, step)
        report!(reporting_strategy, step, report, post_step_stats, id)

        if len == 0
            @error "Population in state $(s_state.id) is dead. Aborting."
            return false
        end
    end

    return true
end

function advance!(report, state::QDReplicaState, s_state::ProductSingleState)

    @unpack v, pv, u, wm, id, order = s_state
    @unpack time_step_parameters, shift, hamiltonian, time_step_strategy,
        reporting_strategy = state
    @unpack time_step = time_step_parameters
    step = state.step[]

    step_stat_names, step_stat_values, wm, pv = apply_operator!(wm, pv, v, u)
    v, pv = pv, v
    comp_name = CompressionStrategy(v) isa NoCompression ? () : (:len_before_compression,)
    comp_stat = compress!(v)
    names = (step_stat_names..., comp_name...)
    stats = (step_stat_values..., comp_stat...)

    if !(time_step_strategy isa ConstantTimeStep)
        u = NthOrderTimeEvolution(hamiltonian, time_step, order)
    end

    @pack! s_state = v, pv, u, wm, id

    if step % reporting_interval(reporting_strategy) == 0
        walkers, len = walkernumber_and_length(v)

        report!(reporting_strategy, step, report, (; len), id)
        report!(reporting_strategy, step, report, (; walkers), id)

        report!(reporting_strategy, step, report, names, stats, id)

        post_step_stats = Rimu.post_step_action(state.post_step_strategy, s_state, step)
        report!(reporting_strategy, step, report, post_step_stats, id)

        if len == 0
            @error "Population in state $(s_state.id) is dead. Aborting."
            return false
        end
    end

    return true
end

function advance!(report, state::QDReplicaState, s_state::RKSingleState)

    @unpack v,w,x,wm,u1,u2,id = s_state
    @unpack time_step_parameters, shift, hamiltonian, time_step_strategy,
        reporting_strategy = state
    @unpack time_step = time_step_parameters
    step = state.step[]

    a, b, wm, w = apply_operator!(wm, w, v, u2)
    add!(w, v, im*shift*time_step/2)
    a, b, wm, x = apply_operator!(wm, x, w, u1)
    add!(add!(v, x), w, im*shift*time_step - 1)
    comp_name = CompressionStrategy(v) isa NoCompression ? () : (:len_before_compression,)
    comp_stat = compress!(v)

    if !(time_step_strategy isa ConstantTimeStep)
        u1 = FirstOrderTimeEvolution(hamiltonian, time_step)
        u2 = FirstOrderTimeEvolution(hamiltonian, time_step/2)
    end

    @pack! s_state = v,w,x,wm,u1,u2,id

    if step % reporting_interval(reporting_strategy) == 0
        walkers, len = walkernumber_and_length(v)

        report!(reporting_strategy, step, report, (; len), id)
        report!(reporting_strategy, step, report, (; walkers), id)

        report!(reporting_strategy, step, report, comp_name, comp_stat, id)

        post_step_stats = Rimu.post_step_action(state.post_step_strategy, s_state, step)
        report!(reporting_strategy, step, report, post_step_stats, id)

        if len == 0
            @error "Population in state $(s_state.id) is dead. Aborting."
            return false
        end
    end

    return true
end

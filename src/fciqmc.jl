"""
    CFCIQMC(; kwargs...) <: QDAlgorithm
Algorithm for FCIQMC in real or complex time.

# Keyword arguments:
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
    a, b, wm, Hw_new = apply_operator!(wm, x, w, hamiltonian)
    add!(Hw_new, w, -shift)
    add!(v, add!(Hw, Hw_new), -0.5*im*time_step)
    Hw, x = Hw_new, Hw
    compress!(v)

    @pack! s_state = v, w, Hw, Hw_new, x, wm

    if step % reporting_interval(reporting_strategy) == 0
        walkers, len = walkernumber_and_length(v)
        two_norm = norm(v, 2)

        report!(reporting_strategy, step, report, (; len), id)
        report!(reporting_strategy, step, report, (; walkers), id)
        report!(reporting_strategy, step, report, (; two_norm), id)

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
    return
end

function advance!(report, state::QDReplicaState, s_state::RKSingleState)
    return
end

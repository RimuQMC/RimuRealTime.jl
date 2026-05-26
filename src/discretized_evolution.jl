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

"""
    EvolutionStrategy
Abstract type for time evolution strategies. Passed as a parameter to
[`QuantumDynamicsProblem`](@ref) or to [`DiscretizedEvolution`](@ref).

## Implemented strategies:

* [`PEC`](@ref)
* [`RungeKutta`](@ref)
* [`Euler`](@ref)
* [`Product`](@ref)
"""
abstract type EvolutionStrategy end

"""
    ScalingStrategy
Abstract type for scaling strategies used to control the walker number. Passed as a
parameter to [`QuantumDynamicsProblem`](@ref) or to [`DiscretizedEvolution`](@ref).
Strategies may be implemented by adding a method for [`scale_state_vector!`](@ref).

## Implemented strategies:

* [`NoScaling`](@ref)
* [`ConstantScaling`](@ref)
* [`DynamicScaling`](@ref)
"""
abstract type ScalingStrategy end

"""
    scale_state_vector!(scaling_strategy::ScalingStrategy, state_vector, current_scale) ->
        names, stats, current_scale
Modify the `state_vector` according to the `scaling_strategy` and return the statistics to
be reported, and the cumulative `current_scale`.
"""
function scale_state_vector!(::ScalingStrategy, _, current_scale)
    return (), (), current_scale
end

"""
    NoScaling() <: ScalingStrategy
Default [`ScalingStrategy`](@ref) that does not scale the vector.
"""
struct NoScaling <: ScalingStrategy end

"""
    ConstantScaling(scale) <: ScalingStrategy
Scale the vector by ``\\exp(-scale*dt)`` every step. The exponential is approximated by the
[`EvolutionStrategy`](@ref).
"""
struct ConstantScaling <: ScalingStrategy
    scale::Float64
end

"""
    DynamicScaling(target_walkers) <: ScalingStrategy
Scale the vector every step so that the 1-norm is equal to `target_walkers`. The cumulative
scale is stored in the DataFrame as `scale`.
"""
struct DynamicScaling <: ScalingStrategy
    target_walkers::Float64
end

function scale_state_vector!(scaling_strategy::DynamicScaling, state_vector, current_scale)
    walkers_prev = norm(state_vector, 1)
    scale_names = (:walkers_before_scaling, :scale,)
    scale!(state_vector, scaling_strategy.target_walkers/walkers_prev)
    current_scale *= scaling_strategy.target_walkers/walkers_prev
    scale_stats = (walkers_prev, current_scale,)
    return scale_names, scale_stats, current_scale
end

"""
    TimeStepParameters
Struct for storing the total time and parameters related to the time step.
"""
mutable struct TimeStepParameters{K<:Union{Float64, ComplexF64}}
    alpha::Float64
    prev_walkers::Float64
    time::K
    time_step::K
    abs_time_step::Float64
end

"""
    WalkerControl(update_strength) <: TimeStepStrategy
Update the phase angle of the time step to control the walker number. The time step is
``dt \\exp(-iα)``, where ``α`` is updated according to 

```math
α_{n+1} = α_{n} + D \\arctan\\left(\\frac{N_\\mathrm{w}^{n+1}}{N_\\mathrm{w}^n}\\right),
```
where ``D`` is the `update_strength`.

The time step amplitude ``dt`` and the starting angle ``α`` are determined using keyword
arguments `time_step` and `alpha` passed to [`QuantumDynamicsProblem`](@ref).
"""
Base.@kwdef struct WalkerControl <: TimeStepStrategy
    update_strength::Float64 = 0.1
end

function update_time_step!(s::WalkerControl, time_step_parameters, walkers)
    @unpack time_step, alpha, prev_walkers, time, abs_time_step= time_step_parameters
    alpha += s.update_strength*atan(walkers/prev_walkers)
    if alpha < 0.0
        alpha = 0.0
    elseif alpha > pi/2
        alpha = pi/2
    end
    prev_walkers = walkers
    time_step = iszero(alpha) ? abs_time_step : abs_time_step*exp(-im*alpha)
    time += time_step
    @pack! time_step_parameters = time_step, alpha, prev_walkers, time, abs_time_step
    return (; time_step, alpha, time)
end

function update_time_step!(::ConstantTimeStep, time_step_parameters, walkers)
    @unpack time_step, time, prev_walkers = time_step_parameters
    time += time_step
    prev_walkers = walkers
    @pack! time_step_parameters = time, prev_walkers
    return (; time)
end

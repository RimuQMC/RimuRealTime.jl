"""
    EvolutionStrategy
Abstract type for time evolution strategies. Passed as a parameter to
[`QuantumDynamicsProblem`](@ref) or to [`DiscretizedEvolution`](@ref).

## Implemented strategies:

* [`PEC`](@ref)
* [`Runge_Kutta`](@ref)
* [`Euler`](@ref)
* [`Product`](@ref)
"""
abstract type EvolutionStrategy end

"""
    PEC() <: EvolutionStrategy
[`EvolutionStrategy`](@ref) for evolution using a second-order predict-evaluate-correct
algorithm. This requires only one application of the Hamiltonian per time step. The state
is updated every time step according to ``v_{n+1} = v_n - i \\frac{dt}{2}(x_n + x_{n+1})``,
where ``x_{n+1} = H w_{n+1}``, with ``w_{n+1} = v_n - idt x_n``. The vector ``x`` is
initialized as ``x_0 = H v_0``.
"""
struct PEC <: EvolutionStrategy end

"""
    Runge_Kutta(damping=0) <: EvolutionStrategy
[`EvolutionStrategy`](@ref) for evolution using a second-order Runge-Kutta algorithm. In
each step the state is updated according to ``v_{n+1} = v_n + u_1 u_2 v_n - u_2 v_n``,
where ``u1 = 1 - i H dt`` and ``u2 = 1 - (1 + d) i H dt / 2``,
and ``d`` is the `damping` coefficient that modifies the second order term. Second
order damping can counteract the effects of large spectral components in the Hamiltonian that
may lead to an unphysical growth of the 2-norm of the state vector.
"""
Base.@kwdef struct Runge_Kutta <: EvolutionStrategy
    damping::Float64 = 0
end

"""
    Euler() <: EvolutionStrategy
[`EvolutionStrategy`](@ref) for evolution using the first-order Euler method. In each step
the state is updated according to ``v_{n+1} = (1 - i H dt)v_n``.
"""
struct Euler <: EvolutionStrategy end

"""
    Product(n) <: EvolutionStrategy
[`EvolutionStrategy`](@ref) for evolution using an `n`th order expansion of the exponential
time evolution operator ``\\exp(-i H dt)``, where powers of the Hamiltonian are applied
using Rimu.HamiltonianProduct. This strategy does not support adding an energy shift.
"""
struct Product <: EvolutionStrategy
    order::Int
end

"""
    ScalingStrategy
Abstract type for scaling strategies used to control the walker number. Passed as a
parameter to [`QuantumDynamicsProblem`](@ref) or to [`DiscretizedEvolution`](@ref).

## Implemented strategies:

* [`NoScaling`](@ref)
* [`ConstantScaling`](@ref)
* [`DynamicScaling`](@ref)
"""
abstract type ScalingStrategy end

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

"""
    TimeStepParameters
Struct for storing parameters needed for updating the time step with
[`WalkerControl`](@ref).
"""
mutable struct TimeStepParameters{K<:Union{Float64, ComplexF64}}
    alpha::Float64
    prev_walkers::Float64
    time::K
    time_step::K
    abs_time_step::Float64
    D::Float64
end

"""
    WalkerControl() <: TimeStepStrategy
Update the time step to control the walker number. The time step is ``\\exp(-iα)dt``,
where ``α`` is updated according to 

```math
α_{n+1} = α_{n} + D\\arctan\\left(\\frac{N_\\mathrm{w}^{n+1}}{N_\\mathrm{w}^n}\\right).
```
"""
struct WalkerControl <: TimeStepStrategy end

function update_time_step!(::WalkerControl, time_step_parameters, walkers)
    @unpack time_step, alpha, prev_walkers, D, time, abs_time_step= time_step_parameters
    alpha += D*atan(walkers/prev_walkers)
    if alpha < 0.0
        alpha = 0.0
    elseif alpha > pi/2
        alpha = pi/2
    end
    prev_walkers = walkers
    time_step = iszero(alpha) ? abs_time_step : abs_time_step*exp(-im*alpha)
    time += time_step
    @pack! time_step_parameters = time_step, alpha, prev_walkers, D, time, abs_time_step
    return (; time_step, alpha, time)
end

function update_time_step!(::ConstantTimeStep, time_step_parameters, walkers)
    @unpack time_step, time, prev_walkers = time_step_parameters
    time += time_step
    prev_walkers = walkers
    @pack! time_step_parameters = time, prev_walkers
    return (; time)
end

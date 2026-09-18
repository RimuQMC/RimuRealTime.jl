"""
    EvolutionStrategy
Abstract type for time evolution strategies. Passed as a parameter to
[`QuantumDynamicsProblem`](@ref) or to [`DiscretizedEvolution`](@ref).

## Implemented strategies:

* [`Leapfrog`](@ref)
* [`LeapfrogComplex`](@ref)
* [`PEC`](@ref)
* [`RungeKutta`](@ref)
* [`Euler`](@ref)
* [`Product`](@ref)
* [`ExactEvolution`](@ref)
"""
abstract type EvolutionStrategy end

Rimu.default_style(::EvolutionStrategy) = IsDynamicSemistochastic{ComplexF64}()
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

"""
    FullOverlaps(
        n_replicas=2;
        operator=nothing,
        vecnorm=true,
        name="overlap"
    ) <: ReplicaStrategy{n_replicas}

Compute replica overlaps and report each set as an `n_replicas × n_replicas` matrix. Rows
label the left vectors and columns label the right vectors, so diagonal norms and both
orders of every replica pair are retained. This differs from [`Rimu.AllOverlaps`](@extref),
which reports the individual off-diagonal overlaps separately.

`operator` may be one observable or a tuple or vector of observables. Set `vecnorm=false`
to omit vector-vector overlaps. The vector-overlap column is named `name`. Operator columns
use `Op1`, `Op2`, and so on.

For a detailed description also read [`Rimu.AllOverlaps`](@extref).
See also [`Rimu.ReplicaStrategy`](@extref).
"""
struct FullOverlaps{N,O,VECNORM} <: ReplicaStrategy{N}
    operators::O
    name::String
end

const TupleOrVector = Union{Tuple, Vector}

function FullOverlaps(
    n_replicas=2;
    operator=nothing,
    vecnorm=true,
    name="overlap"
)
    n_replicas isa Integer || throw(ArgumentError("n_replicas must be an integer"))
    if isnothing(operator)
        operators = ()
    elseif operator isa TupleOrVector
        eltype(operator) <: AbstractObservable || throw(ArgumentError(
            "operator must be an AbstractObservable or a Tuple or Vector of AbstractObservables"
        ))
        operators = operator
    else
        operators = (operator,)
    end

    !vecnorm && isempty(operators) && return NoStats(n_replicas)
    return FullOverlaps{n_replicas,typeof(operators),vecnorm}(operators, string(name))
end

function Rimu.replica_stats(
    rs::FullOverlaps{N,<:Any,VECNORM}, states::Tuple{Vararg{Any,N}}
) where {N,VECNORM}
    vecs = SVector{N}(states[i].v for i in 1:N)
    wms = SVector{N}(states[i].wm for i in 1:N)
    return full_overlaps(rs.operators, vecs, wms, Val(VECNORM); name=rs.name)
end

function Rimu.ProjectorMonteCarloProblem{N,S}(
    ::Rimu.PMCAlgorithm,
    ::AbstractHamiltonian,
    _,
    ::StochasticStyle,
    ::InitiatorRule,
    ::Bool,
    ::Rimu.SimulationPlan,
    ::FullOverlaps{N},
    _,
    ::ReportingStrategy,
    ::Tuple,
    ::Rimu.SpectralStrategy{S},
    ::Int,
    ::Rimu.LittleDict{String,String},
    ::Union{Nothing,UInt64},
    ::Int
) where {N,S}
    throw(ArgumentError("`FullOverlaps` are not supported for `ProjectorMonteCarloProblem`."))
end

"""
    full_overlaps(
        operators,
        vectors,
        working_memories,
        vecnorm=true;
        name="overlap"
    )

Return names and `Matrix` values containing all replica overlaps. `vecnorm` controls
vector-vector matrices.
"""
function full_overlaps(
    operators::TupleOrVector,
    vecs::SVector{N,<:AbstractDVec},
    wms,
    ::Val{VECNORM};
    name::String="overlap"
) where {N,VECNORM}
    T = promote_type((valtype(v) for v in vecs)..., eltype.(operators)...)
    names, values = String[], Matrix{T}[]
    diagonal = all(isdiag, operators)
    local_vecs = SVector{N}(
        diagonal ? vecs[i] : DictVectors.copy_to_local!(wms[i], vecs[i])
        for i in 1:N
        )
    if VECNORM
        overlaps = Matrix{T}(undef, N, N)
        for i in 1:N
            overlaps[i, i] = norm(vecs[i], 2)^2
            for j in (i + 1):N
                overlap = dot(vecs[i], vecs[j])
                overlaps[i, j] = overlap
                overlaps[j, i] = conj(overlap)
            end
        end
        push!(names, name)
        push!(values, overlaps)
    end
    for (m, op) in enumerate(operators)
        overlaps = Matrix{T}(undef, N, N)
        for i in 1:N
            overlaps[i, i] = dot_from_right(local_vecs[i], op, vecs[i])
            for j in (i + 1):N
                overlap = dot_from_right(local_vecs[i], op, vecs[j])
                overlaps[i, j] = overlap
                overlaps[j, i] = conj(overlap)
            end
        end
        push!(names, "Op$(m)")
        push!(values, overlaps)
    end
    return Tuple(names), Tuple(values)
end

function Rimu.undo_transforms(
    strat::FullOverlaps{N,O,VECNORM}, ham::AbstractHamiltonian
) where {N,O,VECNORM}
    operators = map(op -> Rimu.undo_transform(ham, op), strat.operators)
    identity = Rimu.undo_transform(ham, IdentityOperator())
    if identity ≢ IdentityOperator()
        operators = (operators..., identity)
    end
    return FullOverlaps{N,typeof(operators),VECNORM}(operators, strat.name)
end

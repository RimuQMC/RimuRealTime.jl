"""
    Leapfrog() <: EvolutionStrategy
[`EvolutionStrategy`](@ref) for evolution using a second-order Leapfrog algorithm.
The real and imaginary parts of the state vector are propagated on staggered time grids
according to (P. B. Visscher, 1991):
```math
R(t+dt)     = R(t)      + dt \\cdot (H - s) \\cdot I(t + dt/2)
I(t+3dt/2)  = I(t+dt/2) - dt \\cdot (H - s) \\cdot R(t + dt)
```
where ``s`` is the shift. Only [`ConstantTimeStep`](@ref) is supported.
"""
Base.@kwdef struct Leapfrog <: EvolutionStrategy end

"""
    LeapfrogSingleState(v, wm, id, hamiltonian, shift, time_step) <: QDSingleState
Struct holding state vector and auxiliary vectors required for [`Leapfrog`](@ref) time evolution.
"""

mutable struct LeapfrogSingleState{V,W} <: QDSingleState
    state_vector::V # the current, valid state vector
    state_real::V # the real part of the state vector
    state_imag_staggered::V # the imaginary part of the state vector in staggered time
    state_imag_staggered_previous::V # the imaginary part of the state vector in staggered time at the previous step
    h_real::V # hamiltonian * real part of the state vector
    h_imag::V # hamiltonian * imaginary part of the state
    working_mem::W
    id::String
    current_scale::Float64
end


const DVecOrVec = Union{AbstractDVec,AbstractVector}

"""
    Norm2LeapfrogProjector() <: Rimu.AbstractProjector
Results in computing the staggered two-norm for [`Leapfrog`](@ref) evolution.
```julia
-> norm2 = sqrt(max(0.0, real(dot(R,R)) + real(dot(Ip, Im))))
``` 
## Usage
```julia
   post_step_strategy = Projector(norm2 = Norm2LeapfrogProjector())
```
"""
mutable struct Norm2LeapfrogProjector() <: Rimu.AbstractProjector
    state_real::Any # R(t+dt)
    state_imag::Any # I(t+3dt/2)
    state_imag_previous::Any # I(t+dt/2)
    Norm2LeapfrogProjector() = new(nothing, nothing, nothing)
end

function Rimu.VectorInterface.inner(proj::Norm2LeapfrogProjector, y::DVecOrVec)
    isnothing(proj.state_real) && return norm(y,2)
    R, Ip, Im = proj.state_real, proj.state_imag, proj.state_imag_previous
    return sqrt(max(0.0, real(dot(R,R)) + real(dot(Ip, Im))))
end


# ----
function LeapfrogSingleState(v, wm, id, hamiltonian, shift, time_step)
    state_vector = deepcopy(v)
    state_real = zerovector(v)
    add!(state_real, v,)
    state_imag_staggered = zerovector(v)
    state_imag_staggered_previous = zerovector(v)
    h_real = zerovector(v)    
    h_imag = zerovector(v)
    working_mem = wm isa PDWorkingMemory ? wm : working_memory(v)
    names, values, working_mem, h_real = apply_operator!(working_mem, h_real, state_real, hamiltonian)
    add!(h_real, state_real, -shift)
    add!(state_imag_staggered, h_real, -time_step / 2)
    add!(state_imag_staggered_previous, h_real, time_step / 2)
    current_scale = 1.0
    return LeapfrogSingleState(
        state_vector,
        state_real,
        state_imag_staggered,
        state_imag_staggered_previous,
        h_real,
        h_imag,
        working_mem,
        id,
        current_scale
    )
end

function advance!(report, state::QDReplicaState, s_state::LeapfrogSingleState)

    @unpack state_vector, state_real, state_imag_staggered, state_imag_staggered_previous, 
        h_real, h_imag, working_mem, id, current_scale = s_state
    @unpack time_step_parameters, shift, hamiltonian, reporting_strategy, algorithm = state
    @unpack time_step = time_step_parameters
    @unpack time_step_strategy, scaling_strategy = algorithm
    step = state.step[]

    @assert time_step_strategy isa ConstantTimeStep "Only constant time step is currently implemented for Leapfrog."

    # I(t+dt/2) is captured before advancing the staggered imaginary part to I(t+3dt/2)
    zerovector!(state_imag_staggered_previous)
    add!(state_imag_staggered_previous, state_imag_staggered,1.0)

    # R(t+dt) = R(t) + dt * (H - s) * I(t+dt/2)
    step_stat_names, step_stat_values, working_mem, h_imag = apply_opreator!(NoCompression(),
        working_mem, h_imag, state_imag_staggered, hamiltonian
    )
    add!(h_imag, state_imag_staggered, -shift)
    add!(state_real, h_imag, time_step)

    # I(t+3dt/2) = I(t+dt/2) - dt * (H - s) * R(t+dt)
    step_stat_names, step_stat_values, working_mem, h_real = apply_opreator!(NoCompression(), 
        working_mem, h_real, state_real, hamiltonian
    )
    add!(h_real, state_real, -shift)
    add!(state_imag_staggered, h_real, -time_step)

    # reconstruct the full state vector at integer time (t+dt)
    # I(t+dt) = 1/2 (I(t+dt/2) + I(t+3dt/2))
    zerovector!(state_vector)
    add!(state_vector, state_real, 1.0)
    add!(state_vector, state_imag_staggered, 0.5im) # +1/2 I(t+3dt/2)
    add!(state_vector, state_imag_staggered_previous, 0.5im) # +1/2 I(t+dt/2)

    if scaling_strategy isa DynamicScaling
        walkers_prev = norm(state_vector,1)
        scale_names = (:walkers_before_scaling, :scale,)
        scale!(state_vector, scaling_strategy.target_walkers/walkers_prev)
        scale!(state_real, scaling_strategy.target_walkers/walkers_prev)
        scale!(state_imag_staggered, scaling_strategy.target_walkers/walkers_prev)
        scale!(state_imag_staggered_previous, scaling_strategy.target_walkers/walkers_prev) # both staggered parts share the same scale
        current_scale *= scaling_strategy.target_walkers/walkers_prev
        scale_stats = (walkers_prev, current_scale,)
    else
        scale_names , scale_stats, current_scale = scale_state_vector!(
            scaling_strategy, state_vector, current_scale
        )
    end

    # Compression
    comp_name = CompressionStrategy(state_vector) isa NoCompression ? () : (:len_before_compression,)
    comp_stat = compress!(state_vector)
    names = (step_stat_names..., comp_name..., scale_names...)
    stats = (step_stat_values..., comp_stat..., scale_stats...)

    @pack! s_state = state_vector, state_real, state_imag_staggered, state_imag_staggered_previous,
        h_real, h_imag, working_mem, id, current_scale
    
    if step % reporting_interval(reporting_strategy) == 0
        walkers, len = walkernumber_and_length(state_vector)

        report!(reporting_strategy, step, report, (; len), id)
        report!(reporting_strategy, step, report, (; walkers), id)
        report!(reporting_strategy, step, report, names, stats, id)

        # inject the raw staggered fields into any Norm2LeapfrogProjector found in post step
        for strat in state.post_step_strategy
            if strat isa Rimu.Projector && strat.projector isa Norm2LeapfrogProjector
                strat.projector.state_real = state_real # R(t+dt)
                strat.projector.state_imag = state_imag_staggered # I(t+3dt/2)
                strat.projector.state_imag_prev = state_imag_staggered_previous # I(t+dt/2)
            end
        end

        post_step_stats = post_step_action(state.post_step_strategy, s_state, step)
        report!(reporting_strategy, step, report, post_step_stats, id)

        if len == 0
            @error "Population in state $(s_state.id) is dead. Aborting."
            return false
        end

    end

    return true
end

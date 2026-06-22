"""
    Leapfrog() <: EvolutionStrategy

[`EvolutionStrategy`](@ref) for evolution using a second-order Leapfrog algorithm.
The real and imaginary parts of the state vector are propagated on staggered time grids
according to (P. B. Visscher, 1991):
``R_{n+1} = R_n + \\Delta t\\,(H - s)\\,I_{n+1/2}`` and
``I_{n+1/2} = I_{n-1/2} - \\Delta t\\,(H - s)\\,R_n``,
where ``s`` is the shift.

For a general complex initial state ``\\Psi_0 = R_0 + i I_0``, the staggered imaginary
parts are initialised as
``I_{+1/2} = I_0 - \\frac{\\Delta t}{2}(H-s)R_0`` and
``I_{-1/2} = I_0 + \\frac{\\Delta t}{2}(H-s)R_0``.
For a real initial state (``I_0 = 0``) this reduces to
``I_{1/2} = -\\frac{\\Delta t}{2}(H-s)R_0``, with ``I_{-1/2}`` set by time-reversal symmetry.
"""
Base.@kwdef struct Leapfrog <: EvolutionStrategy end

"""
    LeapfrogSingleState(v, wm, id, hamiltonian, shift, time_step) <: QDSingleState

Struct holding the state vectors and scratch arrays required for [`Leapfrog`](@ref) time
evolution. The input `v` must be a complex-valued `AbstractDVec`; its real and imaginary
parts are extracted into separate real-valued vectors, enabling real-arithmetic operations
on all staggered fields.

The staggered imaginary parts are initialised from the general complex initial state
``\\Psi_0 = R_0 + i I_0`` as
``I_{\\pm 1/2} = I_0 \\mp \\frac{\\Delta t}{2}(H-s)R_0``.
The bracketing pair ``(I_{n+1/2},\\, I_{n-1/2})`` is retained at each step so that
[`Norm2LeapfrogProjector`](@ref) can evaluate the conserved Visscher norm without
additional storage or state injection.

See [`QDReplicaState`](@ref).
"""
mutable struct LeapfrogSingleState{CV, V, W} <: QDSingleState
    state_vector::CV                  # complex reconstructed state Ψ(t) = R(t) + i·I(t)
    state_real::V                     # real part R(t), on the integer time grid
    state_imag_staggered::V           # imaginary part I(t+½dt), on the staggered grid
    state_imag_staggered_previous::V  # imaginary part I(t-½dt), retained from the previous step
    h_real::V                         # scratch vector: (H-s)·R
    h_imag::V                         # scratch vector: (H-s)·I
    working_mem::W
    id::String
    current_scale::Float64
end

const DVecOrVec = Union{AbstractDVec,AbstractVector}

"""
    Norm2LeapfrogProjector() <: Rimu.AbstractProjector

Sentinel type that triggers computation of the Visscher (1991) conserved staggered two-norm
via a specialised dispatch of `post_step_action` on [`LeapfrogSingleState`](@ref).
The conserved norm is defined as
``\\|\\Psi\\|_{\\rm Visscher} = \\sqrt{R_n \\cdot R_n + I_{n+1/2} \\cdot I_{n-1/2}}``,
where ``R_n``, ``I_{n+1/2}``, and ``I_{n-1/2}`` are read directly from the fields of
[`LeapfrogSingleState`](@ref) at each reporting step.
Usage:
```julia
post_step_strategy = Projector(norm2 = Norm2LeapfrogProjector())
```
"""
struct Norm2LeapfrogProjector <: Rimu.AbstractProjector end

function Rimu.post_step_action(p::Rimu.Projector{Norm2LeapfrogProjector}, s_state::LeapfrogSingleState, step)
    R  = s_state.state_real
    Ip = s_state.state_imag_staggered
    Im = s_state.state_imag_staggered_previous
    val = sqrt(max(0.0, real(dot(R, R)) + real(dot(Ip, Im))))
    return (p.name => val,)
end

"""
    real_part(v::AbstractDVec{K, Complex{T}}) -> DVec{K, T}

Extract the real part of a complex `AbstractDVec` into a new real-valued `DVec`, dropping zero entries.
"""
function real_part(v::AbstractDVec{K, Complex{T}}) where {K, T<:Real}
    r = DVec{K, T}()
    for (k, val) in pairs(v)
        x = real(val)
        iszero(x) || (r[k] = x)
    end
    return r
end

"""
    imag_part(v::AbstractDVec{K, Complex{T}}) -> DVec{K, T}

Extract the imaginary part of a complex `AbstractDVec` into a new real-valued `DVec`, dropping zero entries.
"""
function imag_part(v::AbstractDVec{K, Complex{T}}) where {K, T<:Real}
    r = DVec{K, T}()
    for (k, val) in pairs(v)
        x = imag(val)
        iszero(x) || (r[k] = x)
    end
    return r
end

"""
    complex_dvec(v::DVec{K, T}) -> DVec{K, Complex{T}}

Promote a real-valued `DVec` to its complex counterpart.
"""
function complex_dvec(v::DVec{K, T}) where {K, T<:Real}
    c = DVec{K, Complex{T}}()
    for (k, val) in pairs(v)
        c[k] = Complex(val)
    end
    return c
end

function LeapfrogSingleState(v::AbstractDVec{K, Complex{T}}, wm, id, hamiltonian, shift, time_step) where {K, T<:Real}
    state_real = real_part(v)      # R₀ = Re(Ψ₀)

    # Compute (H-s)·R₀ for use in the staggered initialisation
    h_r = zerovector(state_real)
    working_mem_r = working_memory(state_real)
    names, values, working_mem_r, h_r = apply_operator!(working_mem_r, h_r, state_real, hamiltonian)
    add!(h_r, state_real, -shift)  # h_r = (H-s)·R₀

    # General staggered initialisation: I_{±½} = I₀ ∓ ½dt·(H-s)·R₀
    i0 = imag_part(v)              # I₀ = Im(Ψ₀)  (zero vector for real initial states)
    state_imag_staggered          = zerovector(state_real)
    state_imag_staggered_previous = zerovector(state_real)
    add!(state_imag_staggered,          i0,  1.0)
    add!(state_imag_staggered,          h_r, -time_step / 2)  # I_{+½} = I₀ - ½dt·(H-s)·R₀
    add!(state_imag_staggered_previous, i0,  1.0)
    add!(state_imag_staggered_previous, h_r, +time_step / 2)  # I_{-½} = I₀ + ½dt·(H-s)·R₀

    h_real = zerovector(state_real)
    h_imag = zerovector(state_real)

    # Reconstruct Ψ₀ = R₀ + i·(I_{+½} + I_{-½})/2 = R₀ + i·I₀
    state_vector = complex_dvec(state_real)
    add!(state_vector, state_imag_staggered,          0.5im)
    add!(state_vector, state_imag_staggered_previous, 0.5im)

    working_mem = working_mem_r
    return LeapfrogSingleState(
        state_vector, state_real,
        state_imag_staggered, state_imag_staggered_previous,
        h_real, h_imag, working_mem, id, 1.0
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

    # Archive I(t+½dt) as the "previous" staggered value before it is overwritten below
    zerovector!(state_imag_staggered_previous)
    add!(state_imag_staggered_previous, state_imag_staggered, 1.0)

    # Advance the real part: R(t+dt) = R(t) + dt·(H-s)·I(t+½dt)
    step_stat_names, step_stat_values, working_mem, h_imag = apply_operator!(
        working_mem, h_imag, state_imag_staggered, hamiltonian
    )
    add!(h_imag, state_imag_staggered, -shift)  # h_imag = (H-s)·I(t+½dt)
    add!(state_real, h_imag, time_step)         # R(t+dt) = R(t) + dt·h_imag

    # Advance the imaginary part: I(t+3dt/2) = I(t+½dt) - dt·(H-s)·R(t+dt)
    step_stat_names, step_stat_values, working_mem, h_real = apply_operator!(
        working_mem, h_real, state_real, hamiltonian
    )
    add!(h_real, state_real, -shift)               # h_real = (H-s)·R(t+dt)
    add!(state_imag_staggered, h_real, -time_step) # I(t+3dt/2) = I(t+½dt) - dt·h_real

    # Reconstruct the full complex state at integer time t+dt:
    # Ψ(t+dt) = R(t+dt) + i·I(t+dt),  where I(t+dt) = ½[I(t+½dt) + I(t+3dt/2)]
    zerovector!(state_vector)
    add!(state_vector, state_real,                    1.0)    # R(t+dt)
    add!(state_vector, state_imag_staggered,          0.5im)  # + ½i·I(t+3dt/2)
    add!(state_vector, state_imag_staggered_previous, 0.5im)  # + ½i·I(t+½dt)

    if scaling_strategy isa DynamicScaling
        walkers_prev = norm(state_vector, 1)
        scale_names = (:walkers_before_scaling, :scale,)
        scale!(state_vector,                  scaling_strategy.target_walkers / walkers_prev)
        scale!(state_real,                    scaling_strategy.target_walkers / walkers_prev)
        scale!(state_imag_staggered,          scaling_strategy.target_walkers / walkers_prev)
        scale!(state_imag_staggered_previous, scaling_strategy.target_walkers / walkers_prev)
        # all four fields are rescaled together to keep the Visscher norm consistent
        current_scale *= scaling_strategy.target_walkers / walkers_prev
        scale_stats = (walkers_prev, current_scale,)
    else
        scale_names, scale_stats, current_scale = scale_state_vector!(
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

        post_step_stats = post_step_action(state.post_step_strategy, s_state, step)
        report!(reporting_strategy, step, report, post_step_stats, id)

        if len == 0
            @error "Population in state $(s_state.id) is dead. Aborting."
            return false
        end
    end

    return true
end
"""
    Leapfrog() <: EvolutionStrategy

[`EvolutionStrategy`](@ref) for evolution using a second-order Leapfrog algorithm.
Pass `Leapfrog()` to [`QuantumDynamicsProblem`](@ref) with the keyword 
`evolution_strategy` to enable this algorithm.
The real and imaginary parts of the state vector are propagated on staggered time grids
according to [P. B. Visscher (1991)](https://doi.org/10.1063/1.168415)::
```math
R_{n+1} = R_n + Δt(H - S)I_{n+1/2}\\\\
I_{n+1/2} = I_{n-1/2} - Δt(H - S)R_n
```
where ``S`` is the shift. Note that [`Norm2LeapfrogProjector`](@ref) is available as a
specialised [`Rimu.PostStepStrategy`](@extref) to compute a conserved 2-norm for 
`Leapfrog` time evolution.

For a general complex initial state ``Ψ_0 = R_0 + i I_0``, the staggered imaginary
parts are initialised as:
```math
I_{+1/2} = I_0 - \\frac{Δt}{2}(H-S)R_0\\\\
I_{-1/2} = I_0 + \\frac{Δt}{2}(H-S)R_0
```
Only [`Rimu.ConstantTimeStep`](@extref) is supported.

See also [`Norm2LeapfrogProjector`](@ref), [`LeapfrogSingleState`](@ref).
"""
struct Leapfrog <: EvolutionStrategy end

"""
    LeapfrogSingleState(v, wm, id, hamiltonian, shift, time_step) <: QDSingleState

Struct holding the state vectors and scratch arrays required for [`Leapfrog`](@ref) time
evolution. The input `v` must be a complex-valued `AbstractDVec`; its real and imaginary
parts are extracted into separate real-valued vectors, enabling real-arithmetic operations
on all staggered fields.

The staggered imaginary parts are initialised from the general complex initial state
``Ψ_0 = R_0 + i I_0`` as:
```math
I_{\\pm 1/2} = I_0 \\mp \\frac{Δt}{2}(H-S)R_0
```
The bracketing pair ``(I_{n+1/2},\\, I_{n-1/2})`` is retained at each step.

See [`Leapfrog`](@ref), [`QDReplicaState`](@ref), [`QuantumDynamicsProblem`](@ref).
"""
mutable struct LeapfrogSingleState{CV, V, W} <: QDSingleState
    state_vector::CV                  # the current, valid complex reconstructed state Psi(t) = R(t) + i.I(t)
    state_real::V                     # real part R(t), on the integer time grid
    state_imag_staggered::V           # imaginary part I(t+1/2dt), on the staggered grid
    state_imag_staggered_previous::V  # imaginary part I(t-1/2dt), retained from the previous step
    h_real::V                         # scratch vector: (H-S).R
    h_imag::V                         # scratch vector: (H-S).I
    working_mem::W
    id::String
    current_scale::Float64
end

function LeapfrogSingleState(v::AbstractDVec{K, Complex{T}}, wm, id, hamiltonian, shift, time_step) where {K, T<:Real}
    state_real = dvec_real(v)      # R_0 = Re(Psi_0)

    # Compute (H-S).R_0 for the use in the staggered initialisation
    h_r = zerovector(state_real)
    working_mem_r = wm isa PDWorkingMemory ? wm : working_memory(state_real)
    names, values, working_mem_r, h_r = apply_operator!(working_mem_r, h_r, state_real, hamiltonian)
    add!(h_r, state_real, -shift)  # h_r = (H-S).R_0

    # General staggered initialisation I_{±1/2} = I_0 ∓ 1/2dt.(H-S).R_0
    i0 = dvec_imag(v) # I_0 = Im(Psi_0)  (zero vector for real initial states)
    state_imag_staggered = zerovector(state_real)
    state_imag_staggered_previous = zerovector(state_real)
    add!(state_imag_staggered, i0, 1.0)
    add!(state_imag_staggered, h_r, -time_step / 2)  # I_{+1/2} = I_0 - 1/2dt.(H-S).R_0
    add!(state_imag_staggered_previous, i0, 1.0)
    add!(state_imag_staggered_previous, h_r, +time_step / 2)  # I_{-1/2} = I_0 + 1/2dt.(H-S).R_0

    h_real = zerovector(state_real)
    h_imag = zerovector(state_real)

    # Reconstruct Psi_0 = R_0 + i.(I_{+1/2} + I_{-1/2})/2 = R_0 + i.I_0
    state_vector = dvec_complex(v)
    add!(state_vector, state_imag_staggered, 0.5im)
    add!(state_vector, state_imag_staggered_previous, 0.5im)
    current_scale = 1.0

    return LeapfrogSingleState(
        state_vector, state_real,
        state_imag_staggered, state_imag_staggered_previous,
        h_real, h_imag, working_mem_r, id, current_scale
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

    # Archive I(t+1/2dt) as the "previous" staggered value before it is overwritten below
    copy!(state_imag_staggered_previous, state_imag_staggered)

    # Advance the real part R(t+dt) = R(t) + dt.(H-S).I(t+1/2dt)
    step_stat_names, step_stat_values, working_mem, h_imag = apply_operator!(NoCompression(),
        working_mem, h_imag, state_imag_staggered, hamiltonian
    )
    add!(h_imag, state_imag_staggered, -shift)  # h_imag = (H-S).I(t+1/2dt)
    add!(state_real, h_imag, time_step)         # R(t+dt) = R(t) + dt.h_imag

    # Advance the imaginary part: I(t+3dt/2) = I(t+1/2dt) - dt.(H-S).R(t+dt)
    step_stat_names, step_stat_values, working_mem, h_real = apply_operator!(NoCompression(),
        working_mem, h_real, state_real, hamiltonian
    )
    add!(h_real, state_real, -shift)               # h_real = (H-S).R(t+dt)
    add!(state_imag_staggered, h_real, -time_step) # I(t+3dt/2) = I(t+1/2dt) - dt.h_real

    # Reconstruct the full complex state at integer time t+dt:
    # Psi(t+dt) = R(t+dt) + i.I(t+dt),  where I(t+dt) = 1/2[I(t+1/2dt) + I(t+3dt/2)]
    zerovector!(state_vector)
    add!(state_vector, state_real, 1.0)    # R(t+dt)
    add!(state_vector, state_imag_staggered, 0.5im)  # + (i/2)*I(t+3dt/2)
    add!(state_vector, state_imag_staggered_previous, 0.5im)  # + (i/2)*I(t+1/2dt)

    if scaling_strategy isa DynamicScaling
        walkers_prev = norm(state_vector, 1)
        scale_names = (:walkers_before_scaling, :scale,)
        scale!(state_vector, scaling_strategy.target_walkers / walkers_prev)
        scale!(state_real, scaling_strategy.target_walkers / walkers_prev)
        scale!(state_imag_staggered, scaling_strategy.target_walkers / walkers_prev)
        scale!(state_imag_staggered_previous, scaling_strategy.target_walkers / walkers_prev)
        current_scale *= scaling_strategy.target_walkers / walkers_prev
        scale_stats = (walkers_prev, current_scale,)
    else
        scale_names = ()
        scale_stats = ()
    end

    # Compression
    comp_name = CompressionStrategy(state_vector) isa NoCompression ? () : (:len_before_compression,)
    comp_stat = compress!(state_vector)
    compress!(state_real)
    compress!(state_imag_staggered)
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

"""
    Norm2LeapfrogProjector() <: Rimu.AbstractProjector

Sentinel type for computing the Visscher (1991) conserved staggered two-norm
when used in `post_step_action` with the [`Leapfrog`](@ref) evolution strategy.
The conserved norm is
```math
|Ψ|_{\\rm Visscher} = \\sqrt{R_n ⋅ R_n + I_{n+1/2} ⋅ I_{n-1/2}},
```
where ``R_n`` is the real component at integer time step ``n`` , and 
``I_{n \\pm 1/2}`` are the imaginary components at the 
adjacent half-integer steps of the staggered grid.

Usage:
```julia
post_step_strategy = Projector(norm2 = Norm2LeapfrogProjector())
```
See [`Rimu.PostStepStrategy`](@extref), [`Rimu.Projector`](@extref),  [`Rimu.DictVectors.AbstractProjector`](@extref).
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
    dvec_real(v::AbstractDVec{K, Complex{T}}) -> AbstractDVec{K, T}

Extract the real part of a complex `AbstractDVec` into a new real-valued vector of the
same concrete type, dropping zero entries.
"""
function dvec_real(v::AbstractDVec{K, Complex{T}}) where {K, T<:Real}
    r = empty(v, T)
    for (k, val) in pairs(v)
        x = real(val)
        iszero(x) || (r[k] = x)
    end
    return r
end


"""
    dvec_imag(v::AbstractDVec{K, Complex{T}}) -> AbstractDVec{K, T}

Extract the imaginary part of a complex `AbstractDVec` into a new real-valued vector of
the same concrete type, dropping zero entries.
"""
function dvec_imag(v::AbstractDVec{K, Complex{T}}) where {K, T<:Real}
    r = empty(v, T)
    for (k, val) in pairs(v)
        x = imag(val)
        iszero(x) || (r[k] = x)
    end
    return r
end

"""
    dvec_complex(v::AbstractDVec{K, T}) -> AbstractDVec{K, Complex{T}}
    dvec_complex(v::AbstractDVec{K, Complex{T}}) -> AbstractDVec{K, Complex{T}}

Promote a real-valued `AbstractDVec` to its complex counterpart of the same concrete type.
If `v` is already complex-valued, returns a copy.
"""
function dvec_complex(v::AbstractDVec{K, T}) where {K, T<:Real}
    c = empty(v, Complex{T})
    for (k, val) in pairs(v)
        c[k] = Complex(val)
    end
    return c
end
function dvec_complex(v::AbstractDVec{K, T}) where {K, T<:Complex}
    c = empty(v, T)
    for (k, val) in pairs(v)
        c[k] = val
    end
    return c
end


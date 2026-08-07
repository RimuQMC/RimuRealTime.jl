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
struct LeapfrogSingleState{CV, V, W} <: QDSingleState
    state_vector::CV # the current, valid complex reconstructed state Psi(t) = R(t) + i.I(t)
    state_real::V # real part R(t), on the integer time grid
    state_imag_staggered::V # imaginary part I(t+1/2dt), on the staggered grid
    state_imag_staggered_previous::V # imaginary part I(t-1/2dt), retained from the previous step
    h_real::V # scratch vector: (H-S).R                     
    h_imag::V # scratch vector: (H-S).I
    working_mem::W
    id::String
    current_scale::Ref{Float64}
end

function LeapfrogSingleState(v::AbstractDVec{K, Complex{T}}, wm, id, hamiltonian, shift, time_step) where {K, T<:Real}
    state_real = copy(v) # R_0 = Re(Psi_0)

    # Compute (H-S).R_0 for the use in the staggered initialisation
    h_r = zerovector(state_real)
    working_mem_r = wm isa PDWorkingMemory ? wm : working_memory(state_real)
    _, _, working_mem_r, h_r = apply_operator!(working_mem_r, h_r, state_real, hamiltonian)
    add!(h_r, state_real, -shift) # h_r = (H-S).R_0

    # General staggered initialisation I_{±1/2} = I_0 ∓ 1/2dt.(H-S).R_0
    i0 = zerovector(v) # I_0 = Im(Psi_0)  (zero vector for real initial states)
    state_imag_staggered = zerovector(state_real)
    state_imag_staggered_previous = zerovector(state_real)
    add!(state_imag_staggered, i0, 1.0)
    add!(state_imag_staggered, h_r, -time_step / 2)  # I_{+1/2} = I_0 - 1/2dt.(H-S).R_0
    add!(state_imag_staggered_previous, i0, 1.0)
    add!(state_imag_staggered_previous, h_r, +time_step / 2) # I_{-1/2} = I_0 + 1/2dt.(H-S).R_0

    h_real = zerovector(state_real)
    h_imag = zerovector(state_real)

    state_vector = v
    current_scale = 1.0

    return LeapfrogSingleState(
        state_vector, state_real,
        state_imag_staggered, state_imag_staggered_previous,
        h_real, h_imag, working_mem_r, id, Ref(current_scale)
    )
end


function advance!(report, state::QDReplicaState, s_state::LeapfrogSingleState, algorithm::DiscretizedEvolution)
    
    @unpack state_vector, state_real, state_imag_staggered, state_imag_staggered_previous,
        h_real, h_imag, working_mem, id, current_scale = s_state
    @unpack time_step_parameters, shift, hamiltonian, reporting_strategy = state
    @unpack time_step = time_step_parameters
    @unpack scaling_strategy = algorithm
    time_step_strategy = algorithm.time_step_strategy
    step = state.step[]

    @assert time_step_strategy isa ConstantTimeStep "Only constant time step is currently implemented for Leapfrog."

    # Archive I(t+1/2dt) as the "previous" staggered value before it is overwritten below
    # state_imag_staggered_previous, state_imag_staggered = state_imag_staggered,
    copy!(state_imag_staggered_previous, state_imag_staggered)

    # Advance the real part R(t+dt) = R(t) + dt.(H-S).I(t+1/2dt)
    step_stat_names, step_stat_values, working_mem, h_imag = apply_operator!(NoCompression(),
        working_mem, h_imag, state_imag_staggered, hamiltonian
    )
    add!(h_imag, state_imag_staggered, -shift) # h_imag = (H-S).I(t+1/2dt)
    add!(state_real, h_imag, time_step) # R(t+dt) = R(t) + dt.h_imag

    # Advance the imaginary part: I(t+3dt/2) = I(t+1/2dt) - dt.(H-S).R(t+dt)
    step_stat_names, step_stat_values, working_mem, h_real = apply_operator!(NoCompression(),
        working_mem, h_real, state_real, hamiltonian
    )
    add!(h_real, state_real, -shift) # h_real = (H-S).R(t+dt)
    add!(state_imag_staggered, h_real, -time_step)  # I(t+3dt/2) = I(t+1/2dt) - dt.h_real

    # Reconstruct the full complex state at integer time t+dt:
    # Psi(t+dt) = R(t+dt) + i.I(t+dt),  where I(t+dt) = 1/2[I(t+1/2dt) + I(t+3dt/2)]
    zerovector!(state_vector)
    add!(state_vector, state_real, 1.0) # R(t+dt)
    add!(state_vector, state_imag_staggered, 0.5im) # + (i/2)*I(t+3dt/2)
    add!(state_vector, state_imag_staggered_previous, 0.5im) # + (i/2)*I(t+1/2dt)
    if scaling_strategy isa DynamicScaling
        walkers_prev = norm(state_vector, 1)
        scale_names = (:walkers_before_scaling, :scale,)
        scale!(state_vector, scaling_strategy.target_walkers / walkers_prev)
        scale!(state_real, scaling_strategy.target_walkers / walkers_prev)
        scale!(state_imag_staggered, scaling_strategy.target_walkers / walkers_prev)
        scale!(state_imag_staggered_previous, scaling_strategy.target_walkers / walkers_prev)
        current_scale[] *= scaling_strategy.target_walkers / walkers_prev
        scale_stats = (walkers_prev, current_scale[],)
    else
        scale_names = ()
        scale_stats = ()
    end

    
    comp_name = CompressionStrategy(state_vector) isa NoCompression ? () : (:len_before_compression,)
    comp_stat = compress!(state_vector)
    compress!(state_real)
    compress!(state_imag_staggered)
    compress!(state_imag_staggered_previous)
    names = (step_stat_names..., comp_name..., scale_names...)
    stats = (step_stat_values..., comp_stat..., scale_stats...)

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

function create_single_state(es::Leapfrog, algorithm, v, wm, id, hamiltonian, shift, time_step)
    return LeapfrogSingleState(v, wm, id, hamiltonian, shift, time_step)
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

function Rimu.post_step_action(p::Rimu.Projector{Norm2LeapfrogProjector}, s_state::LeapfrogSingleState, _step)
    R  = s_state.state_real
    Ip = s_state.state_imag_staggered
    Im = s_state.state_imag_staggered_previous
    val = sqrt(max(0.0, real(dot(R, R)) + real(dot(Ip, Im))))
    return (p.name => val,)
end


"""
    LeapfrogComplex() <: EvolutionStrategy

[`EvolutionStrategy`](@ref) for evolution with a complex-valued staggered-state
Leapfrog scheme. Pass `LeapfrogComplex()` to [`QuantumDynamicsProblem`](@ref)
with the keyword `evolution_strategy` to enable this algorithm. The initial
staggered states are obtained by Euler estimates at the preceding half-step
time points,

```math
C_{-1/2} = C_0 + \\frac{i \\Delta t}{2}(H-S) C_0,\\\\
C_{-3/2} = C_0 + \\frac{3 i \\Delta t}{2}(H-S) C_0.
```

At each step, the staggered state and integer-grid state are advanced according to

```math
C_{n+1/2} = C_{n-1/2} - i \\Delta t (H-S) C_n,\\\\
C_{n+1} = C_n - i \\Delta t (H-S) C_{n+1/2}.
```

Only [`Rimu.ConstantTimeStep`](@extref) is supported.

See also [`Leapfrog`](@ref), [`LeapfrogComplexSingleState`](@ref), and
[`Norm2LeapfrogComplexProjector`](@ref).
"""
struct LeapfrogComplex <: EvolutionStrategy end

"""
    LeapfrogComplexSingleState(v, wm, id, hamiltonian, shift, time_step) <: QDSingleState

Struct holding the complex state, its staggered states, and scratch storage
required by [`LeapfrogComplex`](@ref) time evolution.

The input `v` must be a complex-valued `AbstractDVec`. `state_vector` stores the
state on the integer time grid. `state_staggered` and
`state_staggered_previous` store the states at the preceding half-step and
three-half-step time points, respectively, and are initialized with Euler
estimates from the initial state. The staggered pair is retained at each step.

See also [`LeapfrogComplex`](@ref), [`QDReplicaState`](@ref), and
[`QuantumDynamicsProblem`](@ref).
"""
struct LeapfrogComplexSingleState{CV, V, W} <: QDSingleState
    state_vector::CV # the current, valid complex state
    state_vector_previous::CV # the complex state from the previous step
    state_staggered::V # the staggered complex state
    state_staggered_previous::V # the staggered complex state from the previous step
    h_vector::V # scratch vector: (H-S).Psi
    h_staggered::V # scratch vector: (H-S).Psi_staggered
    working_mem::W
    id::String
    current_scale::Ref{Float64}
end
 
function LeapfrogComplexSingleState(v::AbstractDVec{K, Complex{T}}, wm ,id, hamiltonian, shift, time_step) where {K, T<:Real}
    state_vector = copy(v) # Current Complex Vector
    state_vector_previous = zerovector(state_vector) # Previous Complex Vector

    h_vector = zerovector(state_vector)
    working_mem = wm isa PDWorkingMemory ? wm : working_memory(state_vector)
    _, _, working_mem, h_vector = apply_operator!(working_mem, h_vector, state_vector, hamiltonian)
    add!(h_vector, state_vector, -shift)

    state_staggered = copy(state_vector)
    add!(state_staggered, h_vector, +im * time_step / 2) # C_{-1/2} = [1 + i dt / 2 (H-S)] C_0 (Euler step)

    state_staggered_previous = copy(state_vector)
    add!(state_staggered_previous, h_vector, + 3im * time_step / 2) # C_{-3/2} = [1 + 3 i dt / 2 (H-S)] C_0 (Euler step)

    h_staggered = zerovector(state_vector)
    current_scale = 1.0
    
    return LeapfrogComplexSingleState(
        state_vector, state_vector_previous, state_staggered, state_staggered_previous,
        h_vector, h_staggered, working_mem, id, Ref(current_scale)
)
end
 
function advance!(report, state::QDReplicaState, s_state::LeapfrogComplexSingleState, algorithm::DiscretizedEvolution)

    @unpack state_vector, state_vector_previous, state_staggered, state_staggered_previous, h_vector,
        h_staggered, working_mem, id, current_scale = s_state
    @unpack time_step_parameters, shift, hamiltonian, reporting_strategy = state
    @unpack time_step = time_step_parameters
    @unpack scaling_strategy = algorithm
    time_step_strategy = algorithm.time_step_strategy
    step = state.step[]

    @assert time_step_strategy isa ConstantTimeStep "Only constant time step is currently implemented for Leapfrog"
    
    copy!(state_staggered_previous, state_staggered) # archive C(t-dt/2) as the "previous" staggered value

    step_stat_names, step_stat_values, working_mem, h_vector = apply_operator!(NoCompression(),
        working_mem, h_vector, state_vector, hamiltonian
    )
    add!(h_vector, state_vector, -shift)
    add!(state_staggered, h_vector, -im * time_step) # C(t+dt/2) = C(t-dt/2) - i dt.(H-S).C(t) # new staggered
    step_stat_names, step_stat_values, working_mem, h_staggered = apply_operator!(NoCompression(),
        working_mem, h_staggered, state_staggered, hamiltonian
    )
    add!(h_staggered, state_staggered, -shift)
    copy!(state_vector_previous, state_vector) # archive C(t) as the "previous" integer-grid value
    add!(state_vector, h_staggered, -im * time_step) # C(t+dt) = C(t) - i dt.(H-S).C(t+dt/2) # new integer
    if scaling_strategy isa DynamicScaling
        walkers_prev = norm(state_vector, 1)
        scale_names = (:walkers_before_scaling, :scale,)
        scale!(state_vector, scaling_strategy.target_walkers / walkers_prev)
        scale!(state_staggered, scaling_strategy.target_walkers / walkers_prev)
        scale!(state_staggered_previous, scaling_strategy.target_walkers / walkers_prev)
        scale!(state_vector_previous, scaling_strategy.target_walkers / walkers_prev)
        current_scale[] *= scaling_strategy.target_walkers / walkers_prev
        scale_stats = (walkers_prev, current_scale[],)
    else
        scale_names = ()
        scale_stats = ()
    end
    
    
    comp_name = CompressionStrategy(state_vector) isa NoCompression ? () : (:len_before_compression,)
    comp_stat = compress!(state_vector)
    compress!(state_staggered)
    names = (step_stat_names..., comp_name..., scale_names...)
    stats = (step_stat_values..., comp_stat..., scale_stats...)

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
    Norm2LeapfrogComplexProjector() <: Rimu.AbstractProjector

Sentinel type for reporting norm diagnostics for
[`LeapfrogComplex`](@ref) through `post_step_action`.

The associated post-step action reports four diagnostics. The first two are
the ordinary 2-norms of the current integer-grid state ``C_n`` and the current
staggered state ``C_{n+1/2}``. The other two are real and imaginary component
combinations evaluated using the retained staggered pair
``(C_{n+1/2}, C_{n-1/2})``. Writing ``C_k = R_k + i I_k``, these diagnostics are:

```math
N_1 = \\sqrt{C_n^* \\cdot C_n},\\\\
N_2 = \\sqrt{C_{n+1/2}^* \\cdot C_{n+1/2}},\\\\
N_3 = \\sqrt{R_n \\cdot R_n + I_{n-1/2} \\cdot I_{n+1/2}},\\\\
N_4 = \\sqrt{R_{n-1/2} \\cdot R_{n+1/2} + I_n \\cdot I_n}.
```

The values are returned under the keys `:norm2_1`, `:norm2_2`, `:norm2_3`,
and `:norm2_4`. Usage:

```julia
post_step_strategy = Projector(norm2 = Norm2LeapfrogComplexProjector())
```

This projector is specific to the complex staggered-state implementation and
should not be confused with [`Norm2LeapfrogProjector`](@ref), which applies to
[`Leapfrog`](@ref).

See also [`Rimu.PostStepStrategy`](@extref), [`Rimu.Projector`](@extref), and
[`Rimu.DictVectors.AbstractProjector`](@extref).
"""
struct Norm2LeapfrogComplexProjector <: Rimu.AbstractProjector end

"""
    component_dot_products(u, v)

Compute the real and imaginary component dot products of complex-valued vectors
`u` and `v` without constructing temporary real or imaginary valued vectors.

Returns `(real_component_dot, imaginary_component_dot)`, where:

```math
D_R(u, v) =Re(u) \\cdot Re(v),\\\\
D_I(u, v) = Im(u) \\cdot Im(v).
```

Here, `D_R` and `D_I` denote the real and imaginary component dot products.

See also [`Norm2LeapfrogComplexProjector`](@ref).
"""
function component_dot_products(u, v)
    real_component_dot = 0.0
    imaginary_component_dot = 0.0
    for (key, u_value) in pairs(u)
        v_value = v[key]
        real_component_dot = muladd(real(u_value), real(v_value), real_component_dot)
        imaginary_component_dot = muladd(imag(u_value), imag(v_value), imaginary_component_dot)
    end
    return real_component_dot, imaginary_component_dot
end

function Rimu.post_step_action(p::Rimu.Projector{Norm2LeapfrogComplexProjector}, s_state::LeapfrogComplexSingleState,_step)
    @unpack state_vector, state_vector_previous, state_staggered, state_staggered_previous = s_state

    # C_n = R_n + i I_n and C_{n+1/2} = R_{n+1/2} + i I_{n+1/2}.

    # N_1 = sqrt(C_n* ⋅ C_n).
    # N_2 = sqrt(C_{n+1/2}* ⋅ C_{n+1/2}).
    current_state_norm = norm(state_vector, 2)
    staggered_state_norm = norm(state_staggered, 2)

    # C_{n-1/2} is retained in state_staggered_previous.
    current_real_self, current_imag_self = component_dot_products(
        state_vector_previous, state_vector_previous
    )
    previous_real_current_real, previous_imag_current_imag = component_dot_products(
        state_staggered_previous, state_staggered
    )

    # N_3 = sqrt(R_n·R_n + I_{n-1/2}·I_{n+1/2}).
    staggered_norm_left = sqrt(max(
        0.0, current_real_self + previous_imag_current_imag
    ))

    # N_4 = sqrt(R_{n-1/2}·R_{n+1/2} + I_n·I_n).
    staggered_norm_right = sqrt(max(
        0.0, previous_real_current_real + current_imag_self
    ))
    
    return (
        :norm2_1 => current_state_norm,
        :norm2_2 => staggered_state_norm,
        :norm2_3 => staggered_norm_left,
        :norm2_4 => staggered_norm_right
    )
end

function create_single_state(es::LeapfrogComplex, algorithm, v, wm, id, hamiltonian, shift, time_step)
    return LeapfrogComplexSingleState(v, wm, id, hamiltonian, shift, time_step)
end

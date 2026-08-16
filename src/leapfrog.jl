"""
    Leapfrog() <: EvolutionStrategy

[`EvolutionStrategy`](@ref) for evolution using a second-order Leapfrog algorithm.
Pass `Leapfrog()` to [`QuantumDynamicsProblem`](@ref) with the keyword
`evolution_strategy` to enable this algorithm.
The real and imaginary parts of the state vector are propagated on staggered time grids
according to [P. B. Visscher (1991)](https://doi.org/10.1063/1.168415):
```math
\\begin{aligned}
𝐑_{n+1} &= 𝐑_n + dt(𝐇 - S)𝐈_{n+½}\\\\
𝐈_{n+1½} &= 𝐈_{n+½} - dt(𝐇 - S)𝐑_{n+1}
\\end{aligned}
```
where ``S`` is the shift. Note that [`Norm2LeapfrogProjector`](@ref) is available as a
specialised [`Rimu.PostStepStrategy`](@extref) to compute a conserved 2-norm for
`Leapfrog` time evolution.

For a general complex initial state ``Ψ_0 = 𝐑_0 + i𝐈_0``, the staggered imaginary
parts are initialised as:
```math
\\begin{aligned}
𝐈_{+½} &= 𝐈_0 - \\frac{dt}{2}(𝐇-S)𝐑_0\\\\
𝐈_{-½} &= 𝐈_0 + \\frac{dt}{2}(𝐇-S)𝐑_0
\\end{aligned}
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
``Ψ_0 = 𝐑_0 + i𝐈_0`` as:
```math
𝐈_{\\pm ½} = 𝐈_0 \\mp \\frac{dt}{2}(𝐇-S)𝐑_0
```
The bracketing pair ``(𝐈_{n+½},\\, 𝐈_{n-½})`` is retained at each step.

See [`Leapfrog`](@ref), [`QDReplicaState`](@ref), [`QuantumDynamicsProblem`](@ref).
"""
struct LeapfrogSingleState{CV, V, W} <: QDSingleState
    state_vector::CV # the current, valid complex reconstructed state Psi(t) = R(t) + i.I(t)
    state_real::V # real part R(t), on the integer time grid
    state_imag_staggered::V # imaginary part I(t+1/2dt), on the staggered grid
    state_imag_staggered_previous::V # imaginary part I(t-1/2dt), retained from the previous step
    h_real::V # scratch vector: dt.(H-S).R
    h_imag::V # scratch vector: dt.(H-S).I
    working_mem::W
    id::String
    current_scale::Ref{Float64}
end

function LeapfrogSingleState(v::AbstractDVec{K, Complex{T}}, wm, id, hamiltonian, shift, time_step) where {K, T<:Real}
    state_real = copy(v) # R_0 = Re(Psi_0)

    # Compute (H-S).R_0 for the use in the staggered initialisation
    h_r = zerovector(state_real)
    working_mem_r = wm isa PDWorkingMemory ? wm : working_memory(state_real)
    _, _, working_mem_r, h_r = apply_operator!(
        working_mem_r, h_r, state_real, hamiltonian - shift *I
    ) # h_r = (H-S).R_0

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
        working_mem, h_imag, state_imag_staggered, time_step * (hamiltonian - shift*I)
    ) # h_imag = dt.(H-S).I(t+1/2dt)
    add!(state_real, h_imag) # R(t+dt) = R(t) + h_imag

    # Advance the imaginary part: I(t+3dt/2) = I(t+1/2dt) - dt.(H-S).R(t+dt)
    step_stat_names, step_stat_values, working_mem, h_real = apply_operator!(NoCompression(),
        working_mem, h_real, state_real, -time_step * (hamiltonian - shift*I)
    ) # h_real = -dt.(H-S).R(t+dt)
    add!(state_imag_staggered, h_real) # I(t+3dt/2) = I(t+1/2dt) + h_real

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
|Ψ|_{\\rm Visscher} = \\sqrt{𝐑_n \\cdot 𝐑_n + 𝐈_{n+½} \\cdot 𝐈_{n-½}},
```
where ``𝐑_n`` is the real component at integer time step ``n``, and
``𝐈_{n \\pm ½}`` are the imaginary components at the
adjacent half-integer steps of the staggered grid.

Usage:
```julia
post_step_strategy = Projector(norm2 = Norm2LeapfrogProjector())
```
See [`Rimu.PostStepStrategy`](@extref), [`Rimu.Projector`](@extref),  [`Rimu.DictVectors.AbstractProjector`](@extref).
"""
struct Norm2LeapfrogProjector <: Rimu.AbstractProjector end

function Rimu.post_step_action(
    p::Rimu.Projector{Norm2LeapfrogProjector},
    s_state::LeapfrogSingleState,
    _
)
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
with the keyword `evolution_strategy` to enable this algorithm.

At each step, the staggered state and integer-time state are advanced according to
```math
\\begin{aligned}
𝐂_{n+½} &= 𝐂_{n-½} - i dt (𝐇-S) 𝐂ₙ\\\\
𝐂ₙ₊₁ &= 𝐂ₙ - i dt (𝐇-S) 𝐂_{n+½}
\\end{aligned}
```

At each step of the calculation, three state vectors at different times are available to
facilitate the calculation of the Visscher norm (see
[`Norm2LeapfrogComplexProjector`](@ref)). Given a starting vector ``𝐕``, the
initialisation of the missing time points is performed using Euler steps:
```math
\\begin{aligned}
𝐂₀ &= 𝐕\\\\
𝐂_{-½} &= 𝐂₀ + \\frac{i dt}{2}(𝐇-S) 𝐂₀,\\\\
𝐂₋₁ &= 𝐂₀ + i dt(𝐇-S) 𝐂₀.
\\end{aligned}
```
The only supported `time_step_strategy` is [`Rimu.ConstantTimeStep`](@extref).

See also [`Leapfrog`](@ref), [`LeapfrogComplexSingleState`](@ref), and
[`Norm2LeapfrogComplexProjector`](@ref).
"""
struct LeapfrogComplex <: EvolutionStrategy end

"""
    LeapfrogComplexSingleState(v, wm, id, hamiltonian, shift, time_step) <: QDSingleState

Struct holding the complex state, its staggered state, and scratch storage
required by [`LeapfrogComplex`](@ref) time evolution.

The input `v` must be a complex-valued `AbstractDVec`. `state_vector` stores the
state on the integer time grid, and `state_vector_previous` retains the state
from the preceding integer time step. `state_staggered` stores the state at
the preceding half-step time point. `state_staggered` and `state_vector_previous`
are both initialized with Euler estimates from the initial state, and are
updated once per step.

See also [`LeapfrogComplex`](@ref), [`QDReplicaState`](@ref), and
[`QuantumDynamicsProblem`](@ref).
"""
struct LeapfrogComplexSingleState{CV, V, W} <: QDSingleState
    state_vector::CV # 𝐂(t), the current complex state
    state_vector_previous::CV # 𝐂(t - dt), the state from the previous step
    state_staggered::V # 𝐂(t - dt/2), the staggered complex state
    # Scratch vector for applying (𝐇-S) to state_vector or state_staggered.
    scratch_vector::V
    working_mem::W
    id::String
    current_scale::Ref{Float64}
end
 
function LeapfrogComplexSingleState(
    v::AbstractDVec{K, Complex{T}},
    wm,
    id,
    hamiltonian,
    shift,
    time_step
) where {K, T<:Real}
    state_vector = copy(v) # 𝐂(0), current complex vector

    # scratch_vector = (𝐇 - S) * state_vector.
    scratch_vector = zerovector(state_vector)
    working_mem = wm isa PDWorkingMemory ? wm : working_memory(state_vector)
    _, _, working_mem, scratch_vector = apply_operator!(
        NoCompression(), working_mem, scratch_vector, state_vector, hamiltonian - shift *I
    )

    # Initialize 𝐂₋½ = [1 + i dt / 2 (𝐇-S)] 𝐂₀ with a half-step Euler estimate.
    state_staggered = copy(state_vector)
    add!(state_staggered, scratch_vector, +im * time_step / 2)
    compress!(state_staggered)

    # Initialize 𝐂₋₁ = [1 + i dt (𝐇-S)] 𝐂₀ with a full-step Euler estimate.
    state_vector_previous = copy(state_vector)
    add!(state_vector_previous, scratch_vector, +im * time_step)
    compress!(state_vector_previous)

    current_scale = 1.0

    return LeapfrogComplexSingleState(
        state_vector, state_vector_previous, state_staggered,
        scratch_vector, working_mem, id, Ref(current_scale)
    )
end
 
function advance!(
    report,
    state::QDReplicaState,
    s_state::LeapfrogComplexSingleState,
    algorithm::DiscretizedEvolution
)
    @unpack state_vector, state_vector_previous, state_staggered, scratch_vector,
        working_mem, id, current_scale = s_state
    # state_vector == 𝐂(t)
    # state_vector_previous == 𝐂(t - dt)
    # state_staggered == 𝐂(t - dt/2)
    @unpack time_step_parameters, shift, hamiltonian, reporting_strategy = state
    @unpack time_step = time_step_parameters
    @unpack scaling_strategy = algorithm
    time_step_strategy = algorithm.time_step_strategy
    step = state.step[]

    @assert(
        time_step_strategy isa ConstantTimeStep,
        "Only constant time step is currently implemented for Leapfrog"
    )

    # Update the staggered vector: 𝐂(t+dt/2) = 𝐂(t-dt/2) - i dt.(𝐇-S).𝐂(t).
    step_stat_names, step_stat_values, working_mem, scratch_vector = apply_operator!(
        NoCompression(), working_mem, scratch_vector, state_vector, -im * time_step * (hamiltonian - shift * I)
    )
    add!(state_staggered, scratch_vector)

    # Archive 𝐂(t) as the previous integer-grid value.
    copy!(state_vector_previous, state_vector)

    # Update the integer state: 𝐂(t+dt) = 𝐂(t) - i dt.(𝐇-S).𝐂(t+dt/2).
    step_stat_names, step_stat_values, working_mem, scratch_vector = apply_operator!(
        NoCompression(), working_mem, scratch_vector, state_staggered, -im * time_step * (hamiltonian - shift * I)
    )
    add!(state_vector, scratch_vector)
    if scaling_strategy isa DynamicScaling
        walkers_prev = norm(state_vector, 1)
        scale_names = (:walkers_before_scaling, :scale,)
        scale!(state_vector, scaling_strategy.target_walkers / walkers_prev)
        scale!(state_staggered, scaling_strategy.target_walkers / walkers_prev)
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
the ordinary 2-norms of the current integer-grid state ``𝐂ₙ`` and the
previous integer-grid state ``𝐂ₙ₋₁``. The other two are real and imaginary
component combinations evaluated using ``𝐂ₙ``, ``𝐂ₙ₋₁``, and the retained
staggered state ``𝐂_{n-½}``. Writing ``𝐂ₖ = 𝐑ₖ + i 𝐈ₖ``, these
diagnostics are:

```math
\\begin{aligned}
N₁^2 &= 𝐂ₙ^† 𝐂ₙ\\\\
N₂^2 &= 𝐂ₙ₋₁^† 𝐂ₙ₋₁\\\\
N₃^2 &= 𝐑ₙ₋₁^† 𝐑ₙ + 𝐈_{n-½}^† 𝐈_{n-½}\\\\
N₄^2 &= 𝐑_{n-½}^† 𝐑_{n-½} + 𝐈ₙ₋₁^† 𝐈ₙ
\\end{aligned}
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

Returns the tuple ``(ℜ(𝐮)⋅ℜ(𝐯), ℑ(𝐮)⋅ℑ(𝐯))`` with the real and imaginary part 
dot products, respectively.

See also [`Norm2LeapfrogComplexProjector`](@ref).
"""
function component_dot_products(u, v)
    real_component_dot, imaginary_component_dot = sum(
        pairs(u); init = StaticArrays.SVector(0.0, 0.0)
    ) do (key, u_value)
        v_value = v[key]
        StaticArrays.SVector(real(u_value) * real(v_value), imag(u_value) * imag(v_value))
    end
    return real_component_dot, imaginary_component_dot
end

function Rimu.post_step_action(
    p::Rimu.Projector{Norm2LeapfrogComplexProjector},
    s_state::LeapfrogComplexSingleState,
    _
)
    @unpack state_vector, state_vector_previous, state_staggered = s_state

    # 𝐂ₙ = 𝐑ₙ + i𝐈ₙ and 𝐂ₙ₋₁ = 𝐑ₙ₋₁ + i𝐈ₙ₋₁.

    # N₁ = sqrt(𝐂ₙ† ⋅ 𝐂ₙ)
    # N₂ = sqrt(𝐂ₙ₋₁† ⋅ 𝐂ₙ₋₁)
    current_state_norm = norm(state_vector, 2)
    previous_state_norm = norm(state_vector_previous, 2)

    cross_real, cross_imag = component_dot_products(state_vector_previous, state_vector)
    staggered_real_self, staggered_imag_self = component_dot_products(
        state_staggered, state_staggered
    )

    # N₃ = sqrt(𝐑ₙ₋₁·𝐑ₙ + 𝐈ₙ₋½·𝐈ₙ₋½).
    staggered_norm_left = sqrt(max(
        0.0, cross_real + staggered_imag_self
    ))

    # N₄ = sqrt(𝐑ₙ₋½·𝐑ₙ₋½ + 𝐈ₙ₋₁·𝐈ₙ).
    staggered_norm_right = sqrt(max(
        0.0, staggered_real_self + cross_imag
    ))

    return (
        :norm2_1 => current_state_norm,
        :norm2_2 => previous_state_norm,
        :norm2_3 => staggered_norm_left,
        :norm2_4 => staggered_norm_right
    )
end

"""
    InternalCoherence([name=:internal_coherence]) <: PostStepStrategy
    InternalCoherence(; [name=:internal_coherence]) <: PostStepStrategy

[`Rimu.PostStepStrategy`](@extref) to compute the internal coherence (sign correlation)
between the two staggered states of the [`LeapfrogComplex`](@ref).

At each step of the calculation three state vectors are available: the current
integer-grid state ``𝐂ₙ``, the previous integer-grid state ``𝐂ₙ₋₁``, and the
retained staggered state ``𝐂_{n-½}``. Writing ``𝐂ₖ = 𝐑ₖ + i 𝐈ₖ``, two
reconstructed complex state vectors at the intermediate half-step time point
``n - ½`` (time ``t - dt/2``) are formed:

```math
\\begin{aligned}
\bar{𝐂}_{n-½} &= ½(𝐑ₙ₋₁ + 𝐑ₙ) + i 𝐈_{n-½}\\
\tilde{𝐂}_{n-½} &= 𝐑_{n-½} + ½ i (𝐈ₙ₋₁ + 𝐈ₙ)
\\end{aligned}
```

The internal coherence is evaluated using [`Rimu.Hamiltonians.SignCorrelator`](@extref).
The result is reported under the key `name`, which defaults to `:internal_coherence`.

Usage:
```julia
post_step_strategy = InternalCoherence()
```
or with a custom column name:
```julia
post_step_strategy = InternalCoherence(:my_coherence)
```

This strategy is specific to [`LeapfrogComplex`](@ref) time evolution.

See also [`LeapfrogComplex`](@ref), [`Rimu.Hamiltonians.SignCorrelator`](@extref), 
and [`Rimu.PostStepStrategy`](@extref).
"""
struct InternalCoherence <: PostStepStrategy
    name::Symbol
end

InternalCoherence(; name::Symbol=:internal_coherence) = InternalCoherence(name)

function Rimu.post_step_action(
    ic::InternalCoherence,
    s_state::LeapfrogComplexSingleState,
    _
)
    @unpack state_vector, state_vector_previous, state_staggered = s_state
    ks = union(keys(state_staggered), keys(state_vector), keys(state_vector_previous))
    c_bar = zerovector(state_staggered)   # C̄_{n-½}
    c_tilde = zerovector(state_staggered) # C̃_{n-½}

    for k in ks
        stag, curr, prev = state_staggered[k], state_vector[k], state_vector_previous[k]

        # C̄_{n-½}[k] = ½(𝐑ₙ₋₁[k] + 𝐑ₙ[k]) + i 𝐈_{n-½}[k]
        c_bar_k   = Complex(0.5 * (real(prev) + real(curr)), imag(stag))
        # C̃_{n-½}[k] = 𝐑_{n-½}[k] + ½ i (𝐈ₙ₋₁[k] + 𝐈ₙ[k])
        c_tilde_k = Complex(real(stag), 0.5 * (imag(prev) + imag(curr)))

        iszero(c_bar_k)   || (c_bar[k] = c_bar_k)
        iszero(c_tilde_k) || (c_tilde[k] = c_tilde_k)
    end
    # Internal coherence C̄_{n-½} ⋅ Ŝ ⋅ C̃_{n-½}
    return (ic.name => dot(c_bar, SignCorrelator(), c_tilde),)
end

function create_single_state(
    es::LeapfrogComplex,
    algorithm,
    v,
    wm,
    id,
    hamiltonian,
    shift,
    time_step
)
    return LeapfrogComplexSingleState(v, wm, id, hamiltonian, shift, time_step)
end

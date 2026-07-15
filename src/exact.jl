"""
    ExactEvolution(; kwargs...) <: EvolutionStrategy
[`EvolutionStrategy`](@ref) for evolution using a Krylov subspace exponentiation method.
Pass `ExactEvolution()` to [`QuantumDynamicsProblem`](@ref) with the keyword
`evolution_strategy` to enable this algorithm.
The state is updated every time step according to ``v_{n+1} = e^{-i H dt} v_n``,
where the matrix exponential is computed via a Krylov subspace approximation using
[`KrylovKit.exponentiate`](@extref). This method applies the matrix exponential once per time step.

# Keyword arguments

The following keyword argument and defaults are applied. See [`KrylovKit.exponentiate`](@extref) 
for a detailed description.

- `krylovdim = 30`
- `tol = 1e-12`
- `maxiter = 100`
- `eager = true`
- `verbosity = 0`

Requires a deterministic style: use `style = IsDeterministic{ComplexF64}()` when
constructing the [`QuantumDynamicsProblem`](@ref).

See also [`ExactSingleState`](@ref).
"""
Base.@kwdef struct ExactEvolution <: EvolutionStrategy
    krylovdim::Int = 30      # Dimension of the Krylov subspace
    tol::Float64 = 1e-12     # Tolerance for the Krylov approximation
    maxiter::Int = 100       # Maximum number of Krylov subspace iterations
    eager::Bool = true       # Whether to use the eager variant
    verbosity::Int = 0       # Verbosity level
end

Rimu.default_style(::ExactEvolution) = Rimu.StochasticStyles.IsDeterministic{ComplexF64}()

"""
    ExactSingleState(v, working_memory, id, algorithm) <: QDSingleState
Struct holding the state vector and scratch arrays required for [`ExactEvolution`](@ref)
time evolution. The state ``v_n`` is advanced each step via ``v_{n+1} = e^{-i H dt} v_n`` using a Krylov
subspace approximation configured by `algorithm`.

See [`ExactEvolution`](@ref), [`QDReplicaState`](@ref), [`QuantumDynamicsProblem`](@ref).
"""
mutable struct ExactSingleState{V,W} <: QDSingleState
    state_vector::V           # the current state vector v_n
    working_mem::W            # working memory for Hamiltonian application
    id::String                # identifier for the state
    algorithm::ExactEvolution # parameters for the Krylov exponentiation
end

function ExactSingleState(v, wm, id::String, algorithm::ExactEvolution=ExactEvolution())
    if !(StochasticStyle(v) isa Rimu.StochasticStyles.IsDeterministic)
        throw(ArgumentError(
            "ExactEvolution requires a deterministic stochastic style, but got " *
            "$(typeof(StochasticStyle(v))). Pass `style = IsDeterministic()` to " *
            "`QuantumDynamicsProblem`, or provide a `start_at` AbstractDVec that already " *
            "has a deterministic style."
        ))
    end
    state_vector = deepcopy(v)
    working_mem = wm isa PDWorkingMemory ? wm : working_memory(v)
    return ExactSingleState{typeof(state_vector),typeof(working_mem)}(
        state_vector,
        working_mem,
        id,
        algorithm,
    )
end

"""
    advance!(report, state::QDReplicaState, s_state::ExactSingleState)
Advance the state `s_state` by one step via ``v_{n+1} = e^{-i H dt} v_n`` using a
Krylov subspace approximation, and write data to the `report`.
"""
function advance!(report, state::QDReplicaState, s_state::ExactSingleState)

    @unpack state_vector, working_mem, id, algorithm = s_state
    @unpack krylovdim, tol, maxiter, eager, verbosity = algorithm
    @unpack time_step_parameters, hamiltonian, reporting_strategy = state
    @unpack time_step = time_step_parameters
    step = state.step[]

    # define the Hamiltonian action as a closure that preserves working memory
    wm = Ref(working_mem)
    function hamiltonian_action(x)
        y = zerovector(x)
        _, _, new_wm, y = apply_operator!(NoCompression(), wm[], y, x, hamiltonian)
        wm[] = new_wm
        return y
    end

    # exponentiate: v_{n+1} = exp(-i * H * dt) * v_n
    state_vector, info = exponentiate(
        hamiltonian_action, -im * time_step, state_vector;
        krylovdim, tol=tol, maxiter, ishermitian=ishermitian(hamiltonian), eager, verbosity,
    )
    working_mem = wm[]

    names = (:krylov_converged, :krylov_normres, :krylov_numops, :krylov_numiter)
    stats = (info.converged, info.normres, info.numops, info.numiter)

    @pack! s_state = state_vector, working_mem

    if step % reporting_interval(reporting_strategy) == 0
        walkers, len = walkernumber_and_length(state_vector)

        report!(reporting_strategy, step, report, (; len), id)
        report!(reporting_strategy, step, report, (; walkers), id)

        report!(reporting_strategy, step, report, names, stats, id)

        post_step_stats = post_step_action(state.post_step_strategy, s_state, step)
        report!(reporting_strategy, step, report, post_step_stats, id)

        if len == 0
            @error "State vector in $(s_state.id) has collapsed to zero. Aborting."
            return false
        end
    end

    return true
end
"""
    PEC(damping=0) <: EvolutionStrategy
[`EvolutionStrategy`](@ref) for evolution using a second-order predict-evaluate-correct
algorithm. This requires only one application of the Hamiltonian per time step. The state
is updated every time step according to
``v_{n+1} = v_n - i \\frac{dt}{2}((1 - d) x_n + (1 + d) x_{n+1})``, where
``x_{n+1} = H w_{n+1}``, with ``w_{n+1} = v_n - idt x_n``. The vector ``x`` is initialized
as ``x_0 = H v_0``. ``d`` is the `damping` coefficient that modifies the second-order term.
Second-order damping can counteract the effects of large spectral components in the
Hamiltonian that may lead to an unphysical growth of the 2-norm of the state vector.
"""
Base.@kwdef struct PEC <: EvolutionStrategy
    damping::Float64 = 0
end

"""
    PECSingleState(v, working_memory, id, hamiltonian, shift, damping) <: QDSingleState
Struct holding state vector and other vectors required for [`PEC`](@ref) time evolution.
See [`QDReplicaState`](@ref).
"""
mutable struct PECSingleState{V,W} <: QDSingleState
    state_vector::V # the current, valid state vector
    predictor::V # dummy, at each step predictor = state_vector - im * time_step * h_predictor_old
    h_predictor_old::V # hamiltonian * predictor from previous step
    h_predictor::V # dummy, at each step h_predictor = hamiltonian * predictor
    working_mem::W
    id::String
    damping::Float64
    current_scale::Float64
end
function PECSingleState(v, wm, id, hamiltonian, shift, damping=0.0)
    state_vector = deepcopy(v)
    predictor = zerovector(v)
    h_predictor = zerovector(v)
    working_mem = wm isa PDWorkingMemory ? wm : working_memory(v)
    names, values, working_mem, h_predictor_old = apply_operator!(working_mem, zerovector(v), v, hamiltonian)
    add!(h_predictor_old, v, -shift)
    current_scale = 1.0
    return PECSingleState(
        state_vector,
        predictor,
        h_predictor_old,
        h_predictor,
        working_mem,
        id,
        damping,
        current_scale
    )
end

"""
    advance!(report, state::QDReplicaState, s_state::QDSingleState)
Advance the state `s_state` by one step, and write data to the `report`.
"""
function advance!(report, state::QDReplicaState, s_state::PECSingleState)

    @unpack state_vector, predictor, h_predictor_old, h_predictor, working_mem, id,
        damping, current_scale = s_state
    @unpack time_step_parameters, shift, hamiltonian, reporting_strategy, algorithm = state
    @unpack time_step = time_step_parameters
    @unpack scaling_strategy = algorithm
    step = state.step[]

    predictor = add!(zerovector!(predictor), state_vector)
    # predictor step
    add!(predictor, h_predictor_old, - im*time_step) # w_{n+1} = v_n - i * dt * x_n
    compress!(predictor)

    # evaluate
    step_stat_names, step_stat_values, working_mem, h_predictor = apply_operator!(
        working_mem, h_predictor, predictor, hamiltonian
    )
    add!(h_predictor, predictor, -shift) # x_{n+1} = (H - shift) * w_{n+1}

    # corrector step
    add!(state_vector, add!(h_predictor_old, h_predictor, 1+damping, 1-damping), -im*time_step/2)
    # v_{n+1} = v_n - i*dt/2 * [(1-d) x_n + (1+d) x_{n+1}]

    h_predictor_old, h_predictor = h_predictor, h_predictor_old # swap names of x_{n+1} and x_n for next step
    
    if scaling_strategy isa DynamicScaling
        walkers_prev = norm(state_vector,1)
        scale_names = (:walkers_before_scaling, :scale,)
        scale!(state_vector, scaling_strategy.target_walkers/walkers_prev)
        scale!(h_predictor_old, scaling_strategy.target_walkers/walkers_prev)
        current_scale *= scaling_strategy.target_walkers/walkers_prev
        scale_stats = (walkers_prev, current_scale,)
    else
        scale_names = ()
        scale_stats = ()
    end
    
    comp_name = CompressionStrategy(state_vector) isa NoCompression ? () : (:len_before_compression,)
    comp_stat = compress!(state_vector)
    names = (step_stat_names..., comp_name..., scale_names...)
    stats = (step_stat_values..., comp_stat..., scale_stats...)

    @pack! s_state = state_vector, predictor, h_predictor_old, h_predictor, working_mem,
        current_scale

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

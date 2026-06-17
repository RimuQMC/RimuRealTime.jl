Base.@kwdef struct Leapfrog <: EvolutionStrategy end

mutable struct LeapfrogSingleState{V,W} <: QDSingleState
    state_vector::V # the current state vector
    state_real::V # the real part of the state vector
    state_imag_staggered::V # the imaginary part of the state vector in staggered time
    state_imag_staggered_previous::V # the imaginary part of the state vector in staggered time at the previous step
    h_real::V # hamiltonian * real part of the state vector
    h_imag::V # hamiltonian * imaginary part of the state
    working_mem::W
    id::String
    current_scale::Float64
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

# mutable struct Norm2LeapfrogProjector() <: Rimu.AbstractProjector
#     state_real::Any # R(t+dt)
#     state_imag::Any # I(t+3dt/2)
#     state_imag_previous::Any # I(t+dt/2)
#     Norm2LeapfrogProjector() = new(nothing, nothing, nothing)
# end

function advance!(report, state::QDReplicaState, s_state::LeapfrogSingleState)

    @unpack state_vector, state_real, state_imag_staggered, state_imag_staggered_previous, 
        h_real, h_imag, working_mem, id, current_scale = s_state
    @unpack time_step_parameters, shift, hamiltonian, reporting_strategy, algorithm = state
    @unpack time_step = time_step_parameters
    @unpack time_step_strategy, scaling_strategy = algorithm
    step = state.step[]

    @assert time_step_strategy isa ConstantTimeStep "Only constant time step is currently implemented for Leapfrog."

    zerovector!(state_imag_staggered_previous)
    add!(state_imag_staggered_previous, state_imag_staggered,1.0)
    step_stat_names, step_stat_values, working_mem, h_imag = apply_opreator!(
        working_mem, h_imag, state_imag_staggered, hamiltonian
    )
    add!(h_imag, state_imag_staggered, -shift)
    add!(state_real, h_imag, time_step)

end

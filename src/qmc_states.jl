"""
    QDSingleState
Abstract type for single states for use with different [`EvolutionStrategy`](@ref)s.
[`QDReplicaState`](@ref) holds the Hamiltonian and time step information.

## Concrete types:

* [`PECSingleState`](@ref)
* [`RKSingleState`](@ref)
* [`EulerSingleState`](@ref)
* [`ProductSingleState`](@ref)
"""
abstract type QDSingleState end

#required for AllOverlaps and some of Rimu's post step strategies to work
Rimu.num_spectral_states(::QDSingleState) = 1
Base.getindex(s::QDSingleState, _) = s
function Base.getproperty(s::QDSingleState, sym::Symbol)
    if sym === :v
        return s.state_vector
    elseif sym === :wm
        return s.working_mem
    else
        return getfield(s, sym)
    end
end

"""
    PECSingleState(v, working_memory, id, hamiltonian, shift, damping) <: QDSingleState
Struct holding state vector and other vectors required for [`PEC`](@ref) time evolution.
See [`QDReplicaState`](@ref).
"""
mutable struct PECSingleState{V,W} <: QDSingleState
    state_vector::V
    copy_vector::V # at each step copy_vector = state_vector - im*time_step*H_vector
    H_vector::V # H_vector_new from previous step
    H_vector_new::V # at each step H_vector_new = hamiltonian*copy_vector
    storage_vector::V
    working_mem::W
    id::String
    damping::Float64
    current_scale::Float64
end
function PECSingleState(v, wm, id, hamiltonian, shift, damping=0.0)
    state_vector = deepcopy(v)
    copy_vector = zerovector(v)
    H_vector_new = zerovector(v)
    storage_vector = zerovector(v)
    working_mem = wm isa PDWorkingMemory ? wm : working_memory(v)
    names, values, working_mem, H_vector = apply_operator!(working_mem, zerovector(v), v, hamiltonian)
    add!(H_vector, v, -shift)
    current_scale = 1.0
    return PECSingleState(
        state_vector,
        copy_vector,
        H_vector,
        H_vector_new,
        storage_vector,
        working_mem,
        id,
        damping,
        current_scale
    )
end

"""
    RKSingleState(v, wm, id, hamiltonian, time_step, damping) <: QDSingleState
Struct holding state vector and other vectors required for [`Runge_Kutta`](@ref) time
evolution. See [`QDReplicaState`](@ref).
"""
mutable struct RKSingleState{V,W,U} <: QDSingleState
    state_vector::V
    storage_vector_1::V # stores half_evolution_operator*state_vector
    storage_vector_2::V # stores evolution_operator*storage_vector_1
    working_mem::W
    evolution_operator::U
    half_evolution_operator::U
    id::String
    damping::Float64
    current_scale::Float64
end
function RKSingleState(v, wm, id, hamiltonian, time_step, damping=0.0)
    state_vector = deepcopy(v)
    storage_vector_1 = zerovector(v)
    storage_vector_2 = zerovector(v)
    working_mem = wm isa PDWorkingMemory ? wm : working_memory(v)
    evolution_operator = FirstOrderTimeEvolution(hamiltonian, time_step)
    half_evolution_operator = FirstOrderTimeEvolution(hamiltonian, time_step*(damping + 1)/2)
    current_scale = 1.0
    return RKSingleState(
        state_vector,
        storage_vector_1,
        storage_vector_2,
        working_mem,
        evolution_operator,
        half_evolution_operator,
        id,
        damping,
        current_scale
    )
end

"""
    EulerSingleState(v, wm, id, hamiltonian, time_step) <: QDSingleState
Struct holding state vector and other vectors required for [`Euler`](@ref) time evolution.
See [`QDReplicaState`](@ref).
"""
mutable struct EulerSingleState{V,W,U} <: QDSingleState
    state_vector::V
    previous_vector::V
    working_mem::W
    evolution_operator::U
    id::String
    current_scale::Float64
end
function EulerSingleState(v, wm, id, hamiltonian, time_step)
    state_vector = deepcopy(v)
    previous_vector = zerovector(v)
    working_mem = wm isa PDWorkingMemory ? wm : working_memory(v)
    evolution_operator = FirstOrderTimeEvolution(hamiltonian, time_step)
    current_scale = 1.0
    return EulerSingleState(
        state_vector,
        previous_vector,
        working_mem,
        evolution_operator,
        id,
        current_scale
    )
end

"""
    ProductSingleState(v, wm, id, hamiltonian, time_step, order) <: QDSingleState
Struct holding state vector and other vectors required for [`Product`](@ref) time
evolution. See [`QDReplicaState`](@ref).
"""
mutable struct ProductSingleState{V,W,U} <: QDSingleState
    state_vector::V
    previous_vector::V
    working_mem::W
    evolution_operator::U
    id::String
    order::Int64
    current_scale::Float64
end
function ProductSingleState(v, wm, id, hamiltonian, time_step, order)
    state_vector = deepcopy(v)
    previous_vector = zerovector(v)
    working_mem = wm isa PDWorkingMemory ? wm : working_memory(v)
    evolution_operator = NthOrderTimeEvolution(hamiltonian, time_step, order)
    current_scale = 1.0
    return ProductSingleState(
        state_vector,
        previous_vector,
        working_mem,
        evolution_operator,
        id,
        order,
        current_scale
    )
end

"""
    QDReplicaState <: AbstractVector{QDSingleState}
Holds information about multiple replicas of [`QDSingleState`](@ref)s.

## Fields
- `single_states`: Tuple of `QDSingleState`s.
- `time_step_parameters`: Time step and parameters for updating it.
- `shift`: Energy shift.
- `hamiltonian`: Hamiltonian.
- `algorithm`: Algorithm.
- `step::Ref{Int}`: Current step of the simulation
- `simulation_plan`: Simulation plan
- `reporting_strategy`: Reporting strategy
- `post_step_strategy`: Post-step strategy
- `replica_strategy`: Replica strategy
"""
struct QDReplicaState{
    N,
    NS<:NTuple{N, QDSingleState},
    RRS<:ReplicaStrategy,
    H,
    A,
    RS<:ReportingStrategy,
    PS<:NTuple{<:Any,PostStepStrategy}
} <: AbstractVector{QDSingleState}
    single_states::NS
    time_step_parameters::TimeStepParameters
    shift::Union{Float64,ComplexF64}
    hamiltonian::H
    algorithm::A
    step::Ref{Int}
    simulation_plan::QDSimulationPlan
    reporting_strategy::RS
    post_step_strategy::PS
    replica_strategy::RRS
end

Rimu.num_replicas(::QDReplicaState{N}) where {N} = N
Rimu.num_overlaps(::QDReplicaState{<:Any,<:Any,<:NoStats}) = 0
Rimu.num_overlaps(::QDReplicaState{N,<:Any,<:AllOverlaps{N,<:Any,<:Any,B}}) where {N,B} = B*N*(N-1)÷2

Base.size(r::QDReplicaState) = (num_replicas(r),)
Base.getindex(r::QDReplicaState, i::Int) = r.single_states[i]

function Rimu.report_default_metadata!(report::Report, state::QDReplicaState)
    report_metadata!(report, "pkgversion(RimuRealTime)", pkgversion(RimuRealTime))
    report_metadata!(report, "algorithm", state.algorithm)
    report_metadata!(report, "maximum_time", state.simulation_plan.maximum_time)
    report_metadata!(report, "num_replicas", num_replicas(state))
    report_metadata!(report, "num_overlaps", num_overlaps(state))
    report_metadata!(report, "hamiltonian", state.hamiltonian)
    report_metadata!(report, "reporting_strategy", state.reporting_strategy)
    report_metadata!(report, "time_step_strategy", state.algorithm.time_step_strategy)
    report_metadata!(report, "time_step", state.time_step_parameters.abs_time_step)
    report_metadata!(report, "step", state.step[])
    report_metadata!(report, "shift", state.shift)
    report_metadata!(report, "post_step_strategy", state.post_step_strategy)
    report_metadata!(report, "v_summary", summary(first(state).state_vector))
    report_metadata!(report, "v_type", typeof(first(state).state_vector))
    return report
end

function Rimu.print_stats(io::IO, step, state::QDReplicaState)
    print(io, "[ ", lpad(step, 11), " | ")
    time = lpad(round(state.time_step_parameters.time, digits=4), 10)
    walkers = lpad(round(state.time_step_parameters.prev_walkers, digits=4), 10)
    println(io, "time: ", time, " | walkers: ", walkers)
    flush(io)
end

"""
    QDSingleState
Abstract type for single states, with different [`EvolutionStrategy`](@ref)s.
[`QDReplicaState`](@ref) holds the Hamiltonian and time step information.

##Concrete types:

* [`PECSingleState`](@ref)
* [`RKSingleState`](@ref)
* [`EulerSingleState`](@ref)
* [`ProductSingleState`](@ref)
"""
abstract type QDSingleState end

#required for AllOverlaps to work
Rimu.num_spectral_states(::QDSingleState) = 1
Base.getindex(s::QDSingleState, _) = s

"""
    PECSingleState(v, working_memory, id, hamiltonian, shift) <: QDSingleState
Struct holding state vector and other vectors required for [`PEC`](@ref) time evolution.
See [`QDReplicaState`](@ref).
"""
mutable struct PECSingleState{V,W} <: QDSingleState
    v::V
    w::V
    Hw::V
    Hw_new::V
    x::V
    wm::W
    id::String
    current_scale::Float64
end

function PECSingleState(v, wm, id, hamiltonian, shift)
    vec = deepcopy(v)
    w = zerovector(v)
    Hw = apply_operator(hamiltonian, v) - shift*v
    Hw_new = zerovector(v)
    x = zerovector(v)
    wm = wm isa PDWorkingMemory ? wm : working_memory(v)
    current_scale = 1.0
    return PECSingleState(vec,w,Hw,Hw_new,x,wm,id,current_scale)
end

"""
    RKSingleState(v, wm, id, hamiltonian, time_step, damping) <: QDSingleState
Struct holding state vector and other vectors required for [`Runge_Kutta`](@ref) time
evolution. See [`QDReplicaState`](@ref).
"""
mutable struct RKSingleState{V,W,U} <: QDSingleState
    v::V
    w::V
    x::V
    wm::W
    u1::U
    u2::U
    id::String
    damping::Float64
    current_scale::Float64
end

function RKSingleState(v, wm, id, hamiltonian, time_step, damping=0.0)
    vec = deepcopy(v)
    w = zerovector(v)
    x = zerovector(v)
    wm = wm isa PDWorkingMemory ? wm : working_memory(v)
    u1 = FirstOrderTimeEvolution(hamiltonian, time_step)
    u2 = FirstOrderTimeEvolution(hamiltonian, time_step*(damping + 1)/2)
    current_scale = 1.0
    return RKSingleState(vec, w, x, wm, u1, u2, id, damping, current_scale)
end

"""
    EulerSingleState(v, wm, id, hamiltonian, time_step) <: QDSingleState
Struct holding state vector and other vectors required for [`Euler`](@ref) time evolution.
See [`QDReplicaState`](@ref).
"""
mutable struct EulerSingleState{V,W,U} <: QDSingleState
    v::V
    pv::V
    wm::W
    u::U
    id::String
    current_scale::Float64
end

function EulerSingleState(v, wm, id, hamiltonian, time_step)
    vec = deepcopy(v)
    pv = zerovector(v)
    wm = wm isa PDWorkingMemory ? wm : working_memory(v)
    u = FirstOrderTimeEvolution(hamiltonian, time_step)
    current_scale = 1.0
    return EulerSingleState(vec, pv, wm, u, id, current_scale)
end

"""
    ProductSingleState(v, wm, id, hamiltonian, time_step, order) <: QDSingleState
Struct holding state vector and other vectors required for [`Product`](@ref) time
evolution. See [`QDReplicaState`](@ref).
"""
mutable struct ProductSingleState{V,W,U} <: QDSingleState
    v::V
    pv::V
    wm::W
    u::U
    id::String
    order::Int64
    current_scale::Float64
end

function ProductSingleState(v, wm, id, hamiltonian, time_step, order)
    vec = deepcopy(v)
    pv = zerovector(v)
    wm = wm isa PDWorkingMemory ? wm : working_memory(v)
    u = NthOrderTimeEvolution(hamiltonian, time_step, order)
    current_scale = 1.0
    return ProductSingleState(vec, pv, wm, u, id, order, current_scale)
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
    TS<:TimeStepStrategy,
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
    time_step_strategy::TS
    reporting_strategy::RS
    post_step_strategy::PS
    replica_strategy::RRS
end

num_replicas(::QDReplicaState{N}) where {N} = N
num_overlaps(::QDReplicaState{<:Any,<:Any,<:NoStats}) = 0
num_overlaps(::QDReplicaState{N,<:Any,<:AllOverlaps{N,<:Any,<:Any,B}}) where {N,B} = B*N*(N-1)÷2

Base.size(r::QDReplicaState) = (num_replicas(r),)
Base.getindex(r::QDReplicaState, i::Int) = r.single_states[i]

function report_default_metadata!(report::Report, state::QDReplicaState)
    report_metadata!(report, "RimuRealTime.PACKAGE_VERSION", RimuRealTime.PACKAGE_VERSION)
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
    report_metadata!(report, "v_summary", summary(first(state).v))
    report_metadata!(report, "v_type", typeof(first(state).v))
    return report
end

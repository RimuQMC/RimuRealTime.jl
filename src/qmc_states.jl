"""
    QDSingleState
Abstract type for single states.
"""
abstract type QDSingleState end

"""
    PECSingleState(v, id) <: QDSingleState
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
end

"""
    RKSingleState(v, id) <: QDSingleState
Struct holding state vector and other vectors required for [`Runge_Kutta`](@ref) time
evolution. See [`QDReplicaState`](@ref).
"""
mutable struct RKSingleState <: QDSingleState end

"""
    EulerSingleState(v, id) <: QDSingleState
Struct holding state vector and other vectors required for [`Euler`](@ref) time evolution.
See [`QDReplicaState`](@ref).
"""
mutable struct EulerSingleState{V,W} <: QDSingleState
    v::V
    pv::V
    wm::W
    id::String
end

#required for AllOverlaps to work
Rimu.num_spectral_states(::QDSingleState) = 1
Base.getindex(s::QDSingleState, _) = s

"""
    QDReplicaState <: AbstractVector{QDSingleState}
Holds information about multiple replicas of [`QDSingleState`](@ref)s.

## Fields
- `single_states`: Tuple of `QDSingleState`s.
- `time_step_parameters`: Time step and parameters for updating it.
- `shift`: Energy shift.
- `hamiltonian`: Hamiltonian.
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
    shift::Float64
    hamiltonian::H
    algorithm::A
    step::Ref{Int}
    simulation_plan::QDSimulationPlan
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
    report_metadata!(report, "laststep", state.simulation_plan.last_step)
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

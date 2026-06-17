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
    metadata!(report, "pkgversion(RimuRealTime)", pkgversion(RimuRealTime))
    metadata!(report, "algorithm", state.algorithm)
    metadata!(report, "maximum_time", state.simulation_plan.maximum_time)
    metadata!(report, "num_replicas", num_replicas(state))
    metadata!(report, "num_overlaps", num_overlaps(state))
    metadata!(report, "hamiltonian", state.hamiltonian)
    metadata!(report, "reporting_strategy", state.reporting_strategy)
    metadata!(report, "time_step_strategy", state.algorithm.time_step_strategy)
    metadata!(report, "time_step", state.time_step_parameters.abs_time_step)
    metadata!(report, "step", state.step[])
    metadata!(report, "shift", state.shift)
    metadata!(report, "post_step_strategy", state.post_step_strategy)
    metadata!(report, "v_summary", summary(first(state).state_vector))
    metadata!(report, "v_type", typeof(first(state).state_vector))
    return report
end

function Rimu.print_stats(io::IO, step, state::QDReplicaState)
    print(io, "[ ", lpad(step, 11), " | ")
    time = lpad(round(state.time_step_parameters.time, digits=4), 10)
    walkers = lpad(round(state.time_step_parameters.prev_walkers, digits=4), 10)
    println(io, "time: ", time, " | walkers: ", walkers)
    flush(io)
end

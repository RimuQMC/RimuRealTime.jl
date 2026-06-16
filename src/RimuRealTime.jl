module RimuRealTime

using CommonSolve: CommonSolve, init, step!, solve, solve!
using DataFrames: DataFrames, DataFrame
using OrderedCollections: OrderedCollections, LittleDict
using Parameters: Parameters, @pack!, @unpack
using ProgressLogging: ProgressLogging, @logprogress, @withprogress
using Random: RandomDevice
using Rimu: Rimu, AbstractDVec, AbstractFockAddress, AbstractHamiltonian,
    AbstractObservable, AbstractOperator, AbstractOperatorColumn, AdjointKnown,
    AdjointUnknown, AllOverlaps, CompressionStrategy, ConstantTimeStep, DVec, Hamiltonians,
    HamiltonianSum, IdentityOperator, InitiatorRule, IsDiagonal, IsDynamicSemistochastic,
    IsHermitian, NoCompression, NonInitiator, NoStats, PDWorkingMemory, PostStepStrategy,
    ReplicaStrategy, Report, ReportDFAndInfo, ReportingStrategy, StochasticStyle,
    TimeStepStrategy, add!, allows_address_type, apply_operator!, compress!,
    default_starting_vector, diagonal_element, dimension, dot, ensure_correct_lengths,
    finalize_report!, metadata, has_iterable_offdiagonals, has_random_offdiagonal,
    LOStructure, mpi_seed!, norm, num_offdiagonals, num_overlaps, num_replicas,
    offdiagonals, operator_column, parent_operator, post_step_action, random_offdiagonal,
    refine_reporting_strategy, replica_stats, report!, report_after_step!,
    report_default_metadata!, metadata!, reporting_interval, scale!,
    starting_address, un_finalize!, walkernumber_and_length, working_memory, zerovector,
    zerovector!
using Rimu.Hamiltonians: ModifiedHamiltonian
using Setfield: Setfield, @set

const PACKAGE_NAME = "RimuRealTime"

@doc """
    RimuRealTime
`RimuRealTime` is a package for simulating many-body quantum systems in real time.

Welcome to `RimuRealTime` version $(pkgversion(RimuRealTime)) !
"""
RimuRealTime

include("time_evolution_operators.jl")
include("exponential_sampler.jl")
include("clock.jl")
include("strategies_and_params.jl")
include("quantum_dynamics_problem.jl")
include("qmc_states.jl")
include("pec.jl")
include("runge_kutta.jl")
include("euler.jl")
include("product.jl")
include("discretized_evolution.jl")
include("qd_simulation.jl")

export FirstOrderTimeEvolution, NthOrderTimeEvolution, ExponentialSampler
export Clock, ClockAddress, ClockOperator, ClockObservable, clock_projector
export time_index, fock_address, num_steps, time_evolution_operator, starting_state
export DiscretizedEvolution, WalkerControl, QuantumDynamicsProblem, QDSimulationPlan
export EvolutionStrategy, PEC, RungeKutta, Euler, Product, num_replicas, num_overlaps
export ScalingStrategy, NoScaling, ConstantScaling, DynamicScaling
export init, step!, solve, solve!

end

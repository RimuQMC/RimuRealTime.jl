module RimuRealTime

using Arrow: Arrow
using OrderedCollections: OrderedCollections, LittleDict
using Parameters: Parameters, @pack!, @unpack, @with_kw
using ProgressLogging: ProgressLogging, @logprogress, @withprogress
using Random: Random, RandomDevice, seed!
using Rimu
import Rimu: num_replicas
using Rimu.Hamiltonians: ModifiedHamiltonian
using Setfield: Setfield, @set
import TOML

const PACKAGE_NAME = "RimuRealTime"
const PACKAGE_VERSION = VersionNumber(TOML.parsefile(pkgdir(RimuRealTime, "Project.toml"))["version"])

@doc """
    RimuRealTime
`RimuRealTime` is a package for simulating many-body quantum systems in real time.

Welcome to `RimuRealTime` version $PACKAGE_VERSION !
"""
RimuRealTime

"""
    apply_operator(U::AbstractOperator, v::AbstractDVec) -> AbstractDVec

Non-mutating version of Rimu.apply_operator!, computing the product `Uv` and returning
the resulting vector.
"""
function apply_operator(U::AbstractOperator, v::AbstractDVec)
    n, v, wm, new = apply_operator!(working_memory(v), zerovector(v), v, U)
    return new
end

include("TimeEvolutionOperators.jl")
include("Exponential.jl")
include("Clock.jl")
include("strategies_and_params.jl")
include("quantum_dynamics_problem.jl")
include("qmc_states.jl")
include("fciqmc.jl")
include("qd_simulation.jl")

export apply_operator
export FirstOrderTimeEvolution, NthOrderTimeEvolution, ExponentialSampler
export Clock, ClockAddress, ClockOperator, ClockObservable, ClockProjector, time_index
export address, num_steps, time_evolution_operator, starting_state, time_step
export CFCIQMC, WalkerControl, QuantumDynamicsProblem, PEC, Runge_Kutta, Euler, Product
export ReportingStrategy, ReportDFAndInfo, ReportToFile, num_replicas, QDSimulationPlan
export NoScaling, ConstantScaling, DynamicScaling

end

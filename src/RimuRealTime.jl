module RimuRealTime

using Rimu
using Rimu.Hamiltonians: ModifiedHamiltonian
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
include("fSim.jl")

export apply_operator
export FirstOrderTimeEvolution, NthOrderTimeEvolution, ExponentialSampler
export Clock, ClockAddress, ClockOperator, ClockObservable, ClockProjector, time_index
export address, num_steps, time_evolution_operator, starting_state, time_step
export fSim


end

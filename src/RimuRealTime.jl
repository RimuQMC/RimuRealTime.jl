module RimuRealTime

using Rimu
import TOML

const PACKAGE_NAME = "RimuRealTime"
const PACKAGE_VERSION = VersionNumber(TOML.parsefile(pkgdir(RimuRealTime, "Project.toml"))["version"])

@doc """
    RimuRealTime
`RimuRealTime` is a package for simulating many-body quantum systems in real time.

Welcome to `RimuRealTime` version $PACKAGE_VERSION !
"""
RimuRealTime

function apply_operator(U::AbstractOperator, v::AbstractDVec)
    step_stat_names, step_stat_values, wm, new = apply_operator!(working_memory(v), zerovector(v), v, U)
    return new
end

include("FirstOrderTimeEvolution.jl")
include("Exponential.jl")
include("Clock.jl")

export FirstOrderTimeEvolution, Clock, ClockAddress, FirstOrderClock, ExponentialSampler, apply_operator

end

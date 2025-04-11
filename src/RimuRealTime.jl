module RimuRealTime

using Rimu
import TOML

const PACKAGE_NAME = "RimuRealTime"
const PACKAGE_VERSION = VersionNumber(TOML.parsefile(pkgdir(RimuRealTime, "Project.toml"))["version"])

@doc """
    RimuRealTime
`RimuRealTime` is a package for simulating many-body quantum systems in real time.

Welcome to `RimuRealTime` version $PACKAGE_VERSION!
"""
RimuRealTime

# Write your package code here.

end

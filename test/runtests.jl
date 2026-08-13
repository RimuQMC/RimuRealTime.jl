using Rimu
using RimuRealTime
using RimuRealTime: LeapfrogSingleState, LeapfrogComplexSingleState, PECSingleState, RKSingleState, EulerSingleState, ProductSingleState, ExactSingleState
using SafeTestsets
using Test
using ExplicitImports: check_no_implicit_imports

@safetestset "ExplicitImports" begin
    using RimuRealTime
    using ExplicitImports
    @test check_no_implicit_imports(
        RimuRealTime; skip=(RimuRealTime, Base, Core)
    ) === nothing
end

@safetestset "TimeEvolutionOperators Test" begin
    include("TimeEvolutionOperatorsTest.jl")
end

@safetestset "Clock Test" begin
    include("ClockTest.jl")
end

@safetestset "ExponentialSampler Test" begin
    include("ExponentialSamplerTest.jl")
end

@safetestset "QDStates Test" begin
    include("QDStatesTest.jl")
end

@safetestset "QuantumDynamics Test" begin
    include("QuantumDynamicsTest.jl")
end

@safetestset "NormProjectors Test" begin
    include("NormProjectorTest.jl")
end

@safetestset "MultiReplicaAlgorithm Test" begin
    include("MultiReplicaTest.jl")
end

@safetestset "QDSimulation Test" begin
    include("QDSimulationTest.jl")
end

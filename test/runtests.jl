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

@safetestset "TimeEvolutionOperators" begin
    include("TimeEvolutionOperatorsTest.jl")
end

@safetestset "Clock" begin
    include("ClockTest.jl")
end

@safetestset "ExponentialSampler" begin
    include("ExponentialSamplerTest.jl")
end

@safetestset "QDStates" begin
    include("QDStatesTest.jl")
end

@safetestset "QuantumDynamics" begin
    include("QuantumDynamicsTest.jl")
end

@safetestset "NormProjectors" begin
    include("NormProjectorTest.jl")
end

@safetestset "MultiReplicaAlgorithm" begin
    include("MultiReplicaTest.jl")
end

@safetestset "QDSimulation Test" begin
    include("QDSimulationTest.jl")
end

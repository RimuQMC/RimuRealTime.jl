using Rimu
using RimuRealTime
using RimuRealTime: LeapfrogSingleState, LeapfrogComplexSingleState, PECSingleState, RKSingleState, EulerSingleState, ProductSingleState, ExactSingleState
using Test

@testset "QDStates" begin
    address = FermiFS(1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
    hamiltonian = ExtendedHubbardReal1D(address; v=-2)
    shift = solve(ExactDiagonalizationProblem(hamiltonian)).values[1]
    problem = QuantumDynamicsProblem(hamiltonian)
    shift = 0.0

    v = DVec(address => 1.0)
    v_complex = DVec(address => 1.0 + 0.0im)
    v_leapfrog_complex = DVec(address => 1.0 + 0.5im; style=IsDeterministic{ComplexF64}())

    Leapfrog_state = LeapfrogSingleState(v_complex, working_memory(v_complex), "", hamiltonian, shift, 0.01)
    @test Leapfrog_state.state_vector == v_complex
    @test Leapfrog_state.state_vector === v_complex
    @test Leapfrog_state.state_real == v
    @test Leapfrog_state.state_real !== v
    @test Leapfrog_state.state_imag_staggered == -0.01 / 2 * (hamiltonian * v - shift * v)
    @test Leapfrog_state.state_imag_staggered_previous == 0.01 / 2 * (hamiltonian * v - shift * v)

    Leapfrog_complex_state = LeapfrogComplexSingleState(v_leapfrog_complex, working_memory(v_leapfrog_complex), "", hamiltonian, shift, 0.01)
    @test Leapfrog_complex_state.state_vector == v_leapfrog_complex
    @test Leapfrog_complex_state.state_vector !== v_leapfrog_complex
    h_v_leapfrog_complex = hamiltonian * v_leapfrog_complex - shift * v_leapfrog_complex
    @test Leapfrog_complex_state.state_vector_previous ≈ v_leapfrog_complex + im * 0.01 * h_v_leapfrog_complex
    @test Leapfrog_complex_state.state_staggered ≈ v_leapfrog_complex + im * 0.01 / 2 * h_v_leapfrog_complex

    v_stochastic = DVec(address => 1.0; style=IsDynamicSemistochastic())
    PEC_state = PECSingleState(v, working_memory(v), "", hamiltonian, shift)
    @test PEC_state.state_vector == v
    @test PEC_state.state_vector !== v
    @test PEC_state.h_predictor_old == hamiltonian * v - shift * v

    RK_state = RKSingleState(v, working_memory(v), "", hamiltonian, 0.01)
    @test RK_state.state_vector == v
    @test RK_state.state_vector !== v

    Euler_state = EulerSingleState(v, working_memory(v), "")
    @test Euler_state.state_vector == v
    @test Euler_state.state_vector !== v

    product_state = ProductSingleState(v, working_memory(v), "", hamiltonian, 0.01, 2)
    @test product_state.state_vector == v
    @test product_state.state_vector !== v

    Exact_state = ExactSingleState(v, working_memory(v), "")
    @test Exact_state.state_vector == v
    @test Exact_state.algorithm == ExactEvolution()

    Exact_state2 = ExactSingleState(v, working_memory(v), "", ExactEvolution(krylovdim=10, tol=1e-8))
    @test Exact_state2.algorithm.krylovdim == 10
    @test Exact_state2.algorithm.tol == 1e-8

    @test_throws ArgumentError ExactSingleState(v_stochastic, working_memory(v_stochastic), "")

    @test num_spectral_states(Leapfrog_state) == 1

    state = init(problem).state
    @test num_overlaps(state) == 0

    io = IOBuffer()
    Rimu.print_stats(io, 7, state)
    @test String(take!(io)) == "[ " * lpad(7, 11) * " | time: " *
                               lpad(round(state.time_step_parameters.time, digits=4), 10) * " | walkers: " *
                               lpad(round(state.time_step_parameters.prev_walkers, digits=4), 10) * "\n"
end

using Rimu
using RimuRealTime
using RimuRealTime: LeapfrogSingleState, LeapfrogComplexSingleState
using Test

@testset "Norm2LeapfrogProjector" begin
    address = FermiFS(1,1,1,1,1,0,0,0,0,0)
    hamiltonian = ExtendedHubbardReal1D(address; v=-2)
    shift = solve(ExactDiagonalizationProblem(hamiltonian)).values[1]
    v = DVec(address => 1.0 + 0.0im)
    lf = RimuRealTime.LeapfrogSingleState(v, working_memory(v), "", hamiltonian, shift, 0.01)

    p = Rimu.Projector(projector=Norm2LeapfrogProjector())

    expected = sqrt(max(0.0, real(dot(lf.state_real, lf.state_real)) +
                             real(dot(lf.state_imag_staggered, lf.state_imag_staggered_previous))))

    @test Rimu.post_step_action(p, lf, 1)[1].second ≈ expected

    copy!(lf.state_real, v)
    copy!(lf.state_imag_staggered, v)
    copy!(lf.state_imag_staggered_previous, -10.0 * v)

    result = Rimu.post_step_action(p, lf, 2)[1].second
    @test result >= 0.0
    @test !isnan(result)
end

@testset "LeapfrogStaggeredNormConservation" begin
    address = FermiFS(1,1,1,1,0,0,0,0)
    hamiltonian = ExtendedHubbardReal1D(address; v=-2)
    shift = solve(ExactDiagonalizationProblem(hamiltonian)).values[1]
    post_step_strategy = Projector(norm2 = Norm2LeapfrogProjector())

    problem = QuantumDynamicsProblem(
        hamiltonian;
        shift,
        time_step=0.01,
        last_step=100,
        start_at=DVec(address => 1.0+0.0im; style=IsDeterministic{ComplexF64}()),
        evolution_strategy=Leapfrog(),
        post_step_strategy
    )
    sim = solve(problem)
    df = DataFrame(sim)
    norms = real.(df.norm2)
    @test all(n -> abs(n - norms[1]) / norms[1] < 1e-10, norms)
end

@testset "LeapfrogComplex norm projector and component products" begin
    address = FermiFS(1,1,1,1,0,0,0,0)
    hamiltonian = ExtendedHubbardReal1D(address; v=-2)
    shift = solve(ExactDiagonalizationProblem(hamiltonian)).values[1]
    time_step = 0.01
    v = DVec(address => 1.0 + 0.5im; style=IsDeterministic{ComplexF64}())
    h_v = hamiltonian * v - shift * v
    u = DVec(address => 1.0 + 2.0im)
    w = DVec(address => 3.0 - 4.0im)

    @test RimuRealTime.component_dot_products(u, w) == (3.0, -8.0)

    state = LeapfrogComplexSingleState(
        v, working_memory(v), "", hamiltonian, shift, time_step
            )
    copy!(state.state_vector_previous, v)
    copy!(state.state_staggered_previous, v)

    copy!(state.state_staggered, (2.0 - im) * v)

    projector = Rimu.Projector(projector=Norm2LeapfrogComplexProjector())
    values = Rimu.post_step_action(projector, state, 1)

    current_real, current_imag = RimuRealTime.component_dot_products(
        state.state_vector_previous, state.state_vector_previous
    )
    staggered_real, staggered_imag = RimuRealTime.component_dot_products(
        state.state_staggered_previous, state.state_staggered
    )
    @test values[1].first == :norm2_1
    @test values[1].second ≈ norm(state.state_vector, 2)
    @test values[2].second ≈ norm(state.state_staggered, 2)
    @test values[3].second ≈ sqrt(max(0.0, current_real + staggered_imag))
    @test values[4].second ≈ sqrt(max(0.0, staggered_real + current_imag))
end
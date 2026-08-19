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
    copy!(state.state_staggered, (2.0 - im) * v)

    projector = Rimu.Projector(projector=Norm2LeapfrogComplexProjector())
    values = Rimu.post_step_action(projector, state, 1)

    cross_real, cross_imag = RimuRealTime.component_dot_products(
        state.state_vector_previous, state.state_vector
    )
    staggered_real, staggered_imag = RimuRealTime.component_dot_products(
        state.state_staggered, state.state_staggered
    )
    @test values[1].first == :norm2_1
    @test values[1].second ≈ norm(state.state_vector, 2)
    @test values[2].second ≈ norm(state.state_vector_previous, 2)
    @test values[3].second ≈ sqrt(max(0.0, cross_real + staggered_imag))
    @test values[4].second ≈ sqrt(max(0.0, staggered_real + cross_imag))
end

@testset "InternalCoherence" begin
    address = FermiFS(1,1,1,1,0,0,0,0)
    hamiltonian = ExtendedHubbardReal1D(address; v=-2)
    shift = solve(ExactDiagonalizationProblem(hamiltonian)).values[1]
    time_step = 0.01
    v = DVec(address => 1.0 + 0.5im; style=IsDeterministic{ComplexF64}())

    state = LeapfrogComplexSingleState(
        v, working_memory(v), "", hamiltonian, shift, time_step
    )
    copy!(state.state_vector_previous, v)
    copy!(state.state_staggered, (2.0 - im) * v)

    c_bar_expected = DVec(address => 1.0 + 0.0im)
    c_tilde_expected = DVec(address => 2.5 + 0.5im)
    expected_coherence = dot(c_bar_expected, SignCorrelator(), c_tilde_expected)

    ic = InternalCoherence()
    res = Rimu.post_step_action(ic, state, 1)
    @test res[1].first == :internal_coherence
    @test res[1].second ≈ expected_coherence

    ic_custom = InternalCoherence(:custom_coherence)
    res_custom = Rimu.post_step_action(ic_custom, state, 1)
    @test res_custom[1].first == :custom_coherence
    @test res_custom[1].second ≈ expected_coherence

    a1 = FermiFS(1,1,1,1,0,0,0,0)
    a2 = FermiFS(1,1,1,0,1,0,0,0)
    a3 = FermiFS(1,1,0,1,1,0,0,0)

    sv      = DVec(a1 => 1.0 + 0.5im, a2 => -0.3 + 0.2im; style=IsDeterministic{ComplexF64}())
    sv_prev = DVec(a1 => 0.4 - 0.1im, a2 => 0.6 + 0.1im, a3 => 0.2 + 0.3im; style=IsDeterministic{ComplexF64}())
    sv_stag = DVec(a2 => 1.2 - 0.4im, a3 => -0.5 + 0.7im; style=IsDeterministic{ComplexF64}())

    partial_state = LeapfrogComplexSingleState(
        sv, sv_prev, sv_stag, zerovector(sv), working_memory(sv), "", Ref(1.0)
    )

    c_bar_partial   = DVec(a1 => 0.7 + 0.0im,  a2 => 0.15 - 0.4im)
    c_tilde_partial = DVec(a1 => 0.0 + 0.2im,  a2 => 1.2 + 0.15im)
    expected_partial = dot(c_bar_partial, SignCorrelator(), c_tilde_partial)

    res_partial = Rimu.post_step_action(InternalCoherence(), partial_state, 1)
    @test res_partial[1].second ≈ expected_partial

    problem = QuantumDynamicsProblem(
        hamiltonian;
        shift,
        time_step=0.01,
        last_step=10,
        start_at=v,
        evolution_strategy=LeapfrogComplex(),
        post_step_strategy=InternalCoherence()
    )
    sim = solve(problem)
    df = DataFrame(sim)
    @test :internal_coherence in propertynames(df)
    @test eltype(df.internal_coherence) == ComplexF64
    @test length(df.internal_coherence) == 10
end


using Rimu
using RimuRealTime
using Test

@testset "GPAnsatz" begin
    addr = BoseFS(2, 1)
    H = HubbardReal1D(addr)
    ansatz = GPAnsatz(H)

    # Constructor with Hamiltonian and methods
    @test starting_address(ansatz) == addr
    @test build_basis(ansatz) == build_basis(H)
    @test sprint(show, ansatz) == "GPAnsatz{Float64, modes=2}($H)"

    # Constructor with Address and custom valtype
    ansatz_addr = GPAnsatz(addr; valtype=ComplexF64)
    @test starting_address(ansatz_addr) == addr
    @test valtype(ansatz_addr) == ComplexF64

    # Argument error for multi-component address
    @test_throws ArgumentError GPAnsatz(FermiFS2C((1, 0), (0, 1)))

    # Amplitude evaluation
    c = [0.6, 0.8]
    val = ansatz(addr, c)
    @test val ≈ sqrt(factorial(3) / (factorial(2) * factorial(1))) * (0.6^2) * 0.8

    # Zero parameter branch
    @test iszero(ansatz(BoseFS(2, 1), [0.6, 0.0]))

    # Mismatched particle and mode count branches
    @test iszero(ansatz(BoseFS(1, 1), c))
    @test iszero(ansatz(BoseFS(2, 1, 0), [0.6, 0.8, 0.0]))
end

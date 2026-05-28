"""
    FirstOrderTimeEvolution(H::AbstractHamiltonian, dt) <: AbstractOperator{ComplexF64}

Time evolution operator that approximates time evolution under a Hamiltonian `H` as
``\\exp(-iHdt) \\approx 1 - iHdt``. Apply to an `AbstractDVec` using `Rimu.apply_operator!`
to evolve the state.
"""
struct FirstOrderTimeEvolution{H<:AbstractHamiltonian} <: ModifiedHamiltonian{ComplexF64}
    hamiltonian::H
    dt::Union{Float64, ComplexF64}
end

Rimu.parent_operator(u::FirstOrderTimeEvolution) = u.hamiltonian
Rimu.Hamiltonians.modify_diagonal(u::FirstOrderTimeEvolution, _, value) = 1 - im*u.dt*value
function Rimu.Hamiltonians.modify_offdiagonal(u::FirstOrderTimeEvolution, _, addr, value)
    addr => -im*u.dt*value
end

function Rimu.LOStructure(::Type{<:FirstOrderTimeEvolution{H}}) where {H}
    if LOStructure(H) == IsDiagonal()
        return IsDiagonal()
    elseif LOStructure(H) == AdjointUnknown()
        return AdjointUnknown()
    else
        return AdjointKnown()
    end
end

function Rimu.adjoint(u::FirstOrderTimeEvolution)
    return FirstOrderTimeEvolution(u.hamiltonian', -conj(u.dt))
end

"""
    NthOrderTimeEvolution(H::AbstractHamiltonian, dt, N) <: AbstractOperator{ComplexF64}

Time evolution operator that approximates time evolution under a Hamiltonian `H` as
the `N`th order Taylor expansion of``\\exp(-iHdt)``. Apply to an `AbstractDVec` using
`Rimu.apply_operator!` to evolve the state. If `N == -1`, returns an
[`ExponentialSampler`](@ref), so the vector must not use exact spawning.
"""
function NthOrderTimeEvolution(
    H::AbstractHamiltonian,
    dt::Union{Float64, ComplexF64},
    N::Int
)
    if N == 0
        return IdentityOperator()
    elseif N == -1
        return ExponentialSampler(H, -im*dt)
    else
        op = FirstOrderTimeEvolution(H, dt)
        prod = H
        factor = 1
        count = 1
        while count < N
            count += 1
            prod = H*prod
            factor /= count
            op = HamiltonianSum(op, (factor*(-im*dt)^count)*prod; weight = (count-1)/count)
        end
        return op
    end
end

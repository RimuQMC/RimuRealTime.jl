"""
    FirstOrderTimeEvolution(H::AbstractHamiltonian, dt) <: AbstractOperator{ComplexF64}

Time evolution operator that approximates time evolution under a Hamiltonian `H` as
``\\exp(-iHdt) \\approx 1 - iHdt``. Apply to an `AbstractDVec` using `apply_operator` to
evolve the state.
"""
struct FirstOrderTimeEvolution{H<:AbstractHamiltonian} <: AbstractHamiltonian{ComplexF64}
    hamiltonian::H
    dt::Number
end

Rimu.starting_address(u::FirstOrderTimeEvolution) = starting_address(u.hamiltonian)

parent_operator(u::FirstOrderTimeEvolution) = u.hamiltonian

Rimu.allows_address_type(u::FirstOrderTimeEvolution, ::Type{A}) where {A} = allows_address_type(u.hamiltonian, A)

Rimu.dimension(u::FirstOrderTimeEvolution, add) = dimension(u.hamiltonian, add)

function Rimu.LOStructure(::Type{<:FirstOrderTimeEvolution{H}}) where {H}
    if Rimu.LOStructure(H) == IsDiagonal()
        return IsDiagonal()
    elseif Rimu.LOStructure(H) == AdjointUnknown()
        return AdjointUnknown()
    else
        return AdjointKnown()
    end
end

Rimu.adjoint(u::FirstOrderTimeEvolution) = FirstOrderTimeEvolution(parent_operator(u)', -conj(u.dt))

struct FirstOrderTimeEvolutionColumn{A,U<:FirstOrderTimeEvolution,C<:AbstractOperatorColumn} <: AbstractOperatorColumn{A,ComplexF64,U}
    op::U
    address::A
    ham_column::C
    dt::Number
end
Rimu.operator_column(u::FirstOrderTimeEvolution, add) = FirstOrderTimeEvolutionColumn(u, add, operator_column(u.hamiltonian, add), u.dt)

Rimu.parent_operator(c::FirstOrderTimeEvolutionColumn) = c.op
Rimu.starting_address(c::FirstOrderTimeEvolutionColumn) = c.address
Rimu.diagonal_element(c::FirstOrderTimeEvolutionColumn) = 1 - im*c.dt*diagonal_element(c.ham_column)
Rimu.num_offdiagonals(c::FirstOrderTimeEvolutionColumn) = num_offdiagonals(c.ham_column)

function Rimu.random_offdiagonal(c::FirstOrderTimeEvolutionColumn)
    new_add, prob, val = random_offdiagonal(c.ham_column)
    return new_add, prob, -im*val*c.dt
end

struct FOTEOffdiagonals{O}
    ods::O
    dt::Number
end
Rimu.offdiagonals(c::FirstOrderTimeEvolutionColumn) = FOTEOffdiagonals(offdiagonals(c.ham_column), c.dt)

function Base.iterate(o::FOTEOffdiagonals)
    first = iterate(o.ods)
    if isnothing(first)
        return nothing
    end
    (add, val), state = first
    new_val = -im*val*o.dt
    return add => new_val, state
end

function Base.iterate(o::FOTEOffdiagonals, state)
    new = iterate(o.ods, state)
    if isnothing(new)
        return nothing
    end
    (add, val), state = new
    new_val = -im*val*o.dt
    return add => new_val, state
end

"""
    NthOrderTimeEvolution(H::AbstractHamiltonian, dt, N) <: AbstractOperator{ComplexF64}

Time evolution operator that approximates time evolution under a Hamiltonian `H` as
the `N`th order Taylor expansion of``\\exp(-iHdt)``. Apply to an `AbstractDVec` using
`apply_operator` to evolve the state. If `N == -1`, returns an
[`ExponentialSampler`](@ref), so the vector must not use exact spawning.
"""
function NthOrderTimeEvolution(H::AbstractHamiltonian, dt::Number, N::Int)
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
            op = HamiltonianSum(op, prod; a=1, b=factor*(-im*dt)^count, weight=count-1)# equal weighting for each term
        end
        return op
    end
end

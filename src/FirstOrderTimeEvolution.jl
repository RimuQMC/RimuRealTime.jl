"""
    FirstOrderTimeEvolution(H::AbstractHamiltonian, dt) <: AbstractOperator{ComplexF64}

Time evolution operator that approximates time evolution under a Hamiltonian `H` as
``\exp(-iHdt) \approx 1 - iHdt``. Apply to an `AbstractDVec` using `apply_operator` to
evolve the state.
"""
struct FirstOrderTimeEvolution{H<:AbstractHamiltonian} <: AbstractOperator{ComplexF64}
    hamiltonian::H
    dt::Number
end

parent_operator(u::FirstOrderTimeEvolution) = u.hamiltonian

Rimu.allows_address_type(u::FirstOrderTimeEvolution, add) = allows_address_type(u.hamiltonian, add)

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

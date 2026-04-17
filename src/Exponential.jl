"""
    ExponentialSampler(H<:AbstractHamiltonian, coeff::Number) <: AbstractHamiltonian

Hamiltonian that samples `exp(coeff*H)`. This Hamiltonian does not have iterable
offdiagonals, so exact spawning is not possible.
"""
struct ExponentialSampler{T,H<:AbstractHamiltonian} <: AbstractHamiltonian{T}
    hamiltonian::H
    coeff::T
end
function ExponentialSampler(h::AbstractHamiltonian{T}, coeff::Number) where {T}
    S = promote_type(float(T),typeof(coeff))
    return ExponentialSampler{S,typeof(h)}(h,S(coeff))
end

Rimu.starting_address(e::ExponentialSampler) = starting_address(e.hamiltonian)
Rimu.parent_operator(e::ExponentialSampler) = e.hamiltonian

function Rimu.LOStructure(::Type{<:ExponentialSampler{T,H}}) where {T, H}
    if LOStructure(H) == IsDiagonal()
        return IsDiagonal()
    elseif LOStructure(H) == AdjointUnknown()
        return AdjointUnknown()
    elseif LOStructure(H) == IsHermitian() && T <: Real
        return IsHermitian()
    else
        return AdjointKnown()
    end
end

function Base.adjoint(e::ExponentialSampler{T}) where {T}
    return ExponentialSampler(e.hamiltonian', T(conj(e.coeff)+0.0im))
end

Rimu.has_iterable_offdiagonals(::Type{<:ExponentialSampler}) = false

struct ExponentialOperatorColumn{
    A,
    T,
    O<:ExponentialSampler{T},
    C<:AbstractOperatorColumn
} <: AbstractOperatorColumn{A,T,O}
    op::O
    address::A
    ham_column::C
    diag::T
    coeff::T
end

function Rimu.operator_column(e::ExponentialSampler{T}, add) where {T}
    col = operator_column(e.hamiltonian, add)
    return ExponentialOperatorColumn(e, add, col, T(diagonal_element(col)), e.coeff)
end

Rimu.parent_operator(c::ExponentialOperatorColumn) = c.op
Rimu.starting_address(c::ExponentialOperatorColumn) = c.address
Rimu.num_offdiagonals(::ExponentialOperatorColumn) = Inf
Rimu.diagonal_element(::ExponentialOperatorColumn{<:Any, T}) where {T} = T(1)

function Rimu.random_offdiagonal(c::ExponentialOperatorColumn)
    num = num_offdiagonals(c.ham_column)
    if rand() < 1/(num + 1)
        add = c.address
        prob = 1/(num + 1)
        val = c.diag
    else
        add, prob, val = random_offdiagonal(c.ham_column)
        prob *= num/(num + 1)
    end
    val *= c.coeff
    prob *= 0.5
    count = 2

    while rand() < 0.5
        col = operator_column(c.op.hamiltonian, add)
        num = num_offdiagonals(col)
        if rand() < 1/(num + 1)
            new_prob = 1/(num + 1)
            new_val = diagonal_element(col)
        else
            add, new_prob, new_val = random_offdiagonal(col)
            new_prob *= num/(num + 1)
        end
        val *= new_val*c.coeff/count
        count += 1
        prob *= new_prob*0.5
    end
    return add, prob, val
end

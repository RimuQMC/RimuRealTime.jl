"""
    ExponentialSampler(H<:AbstractHamiltonian, coeff::Number) <: AbstractHamiltonian

Hamiltonian that samples ``\exp(coeff*H)``. This Hamiltonian does not have iterable
offdiagonals, so exact spawning is not possible.
"""
struct ExponentialSampler{T,H<:AbstractHamiltonian} <: AbstractHamiltonian{T}
    hamiltonian::H
    coeff::Number
end
ExponentialSampler(H::AbstractHamiltonian{T}, coeff::N) where {T,N} = ExponentialSampler{promote_type(float(T),N),typeof(H)}(H,coeff)

parent_operator(e::ExponentialSampler) = e.hamiltonian

struct ExponentialOperatorColumn{A,T,O<:ExponentialSampler{T},C<:AbstractOperatorColumn} <: AbstractOperatorColumn{A,T,O}
    op::O
    address::A
    ham_column::C
    diag::T
    coeff::Number
end

function Rimu.operator_column(e::ExponentialSampler{T}, add) where {T}
    col = operator_column(e.hamiltonian, add)
    return ExponentialOperatorColumn(e, add, col, T(diagonal_element(col)), e.coeff)
end

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

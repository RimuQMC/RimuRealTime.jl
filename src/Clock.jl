"""
    Clock <: AbstractHamiltonian{ComplexF64}

Abstract type for clock Hamiltonians. In addition to the `AbstractHamiltonian` interface
from Rimu.jl, the following methods should be implemented:

 * `parent_operator(clock)` - the underlying Hamiltonian
 * `num_steps(clock)` - the number of time steps
 * `time_step(clock)` - the size of the time step
 * `starting_state(clock)` - an `AbstractDVec` in the underlying Fock space, representing
 the state of the system at ``t=0``.
"""
abstract type Clock <: AbstractHamiltonian{ComplexF64} end

LOStructure(::Clock) = IsHermitian()

"""
    ClockAddress(address, t) <: AbstractFockAddress

Address type for use with [`Clock`](@ref)s. Stores an address in the underlying Fock space,
and a time step index `t`. These are accessed with `address(::ClockAddress)` and
`time_index(::ClockAddress)`.
"""
struct ClockAddress{N,M,A<:AbstractFockAddress{N,M}} <: AbstractFockAddress{N,M}
    address::A
    t::Int
end

function Base.show(io::IO, a::ClockAddress)
    print(io, "ClockAddress(", address(a), ", ", time_index(a), ")")
end

address(a::ClockAddress) = a.address
time_index(a::ClockAddress) = a.t

Rimu.allows_address_type(c::Clock, a::ClockAddress) = allows_address_type(parent_operator(c), address(a))

Rimu.starting_address(c::Clock) = ClockAddress(starting_address(parent_operator(c)), 0)

Rimu.dimension(c::Clock, a) = num_steps(c)*dimension(parent_operator(c), address(a))

"""
    FirstOrderClock(H, dt, length; start_at) <: Clock

[`Clock`](@ref) Hamiltonian using a first order approximation to the time evolution operator
``\exp(-iHdt)``. `length` specifies the number of time steps. The optional argument
`start_at` specifies the state of the system at ``t=0``, otherwise this defaults to
`DVec(starting_address(H) => 1.0)`.
"""
struct FirstOrderClock{H<:AbstractHamiltonian} <: Clock
    hamiltonian::H
    dt::Float64
    length::Int
    start_at::AbstractDVec
end
FirstOrderClock(h, dt, length; start_at=DVec(starting_address(h) => 1.0)) = FirstOrderClock(h, dt, length, start_at)

num_steps(c::FirstOrderClock) = c.length
time_step(c::FirstOrderClock) = c.dt
parent_operator(c::FirstOrderClock) = c.hamiltonian
starting_state(c::FirstOrderClock) = c.start_at

struct FirstOrderClockColumn{A<:ClockAddress,O<:FirstOrderClock,C<:AbstractOperatorColumn} <: AbstractOperatorColumn{A,ComplexF64,O}
    clock::O
    address::A
    ham_column::C
end
function Rimu.operator_column(c::FirstOrderClock, a::ClockAddress)
    return FirstOrderClockColumn(c, a, operator_column(parent_operator(c), address(a)))
end

Rimu.starting_address(c::FirstOrderClockColumn) = c.address
parent_operator(c::FirstOrderClockColumn) = c.clock

function Rimu.num_offdiagonals(c::FirstOrderClockColumn)
    if time_index(starting_address(c)) == 0 || time_index(starting_address(c)) == num_steps(parent_operator(c))
        return num_offdiagonals(c.ham_column) + 1
    else
        return 2*(num_offdiagonals(c.ham_column) + 1)
    end
end

function Rimu.diagonal_element(c::FirstOrderClockColumn)
    if time_index(starting_address(c)) == 0
        return 1.5 - abs2(starting_state(parent_operator(c))[address(starting_address(c))])
    elseif time_index(starting_address(c)) == num_steps(parent_operator(c))
        return 0.5
    else
        return 1
    end
end

struct FirstOrderClockOffdiagonals{A<:ClockAddress,T,OD}
    clock::Clock
    address::A
    diag::T
    ods::OD
    dt::Float64
end

function Rimu.offdiagonals(c::FirstOrderClockColumn)
    return FirstOrderClockOffdiagonals(parent_operator(c), starting_address(c), diagonal_element(c.ham_column),offdiagonals(c.ham_column), time_step(parent_operator(c)))
end

Base.IteratorSize(o::FirstOrderClockOffdiagonals) = IteratorSize(o.ods)

function Base.iterate(o::FirstOrderClockOffdiagonals)
    if time_index(o.address) == 0
        return ClockAddress(address(o.address), 1) => -0.5*(1 - im*time_step(o.clock)*o.diag), nothing
    elseif time_index(o.address) == num_steps(o.clock)
        return ClockAddress(address(o.address), time_index(o.address) - 1) => -0.5*(1 + im*time_step(o.clock)*o.diag), nothing
    else
        return ClockAddress(address(o.address), time_index(o.address) - 1) => -0.5*(1 + im*time_step(o.clock)*o.diag), (nothing, true)
    end
end

function Base.iterate(o::FirstOrderClockOffdiagonals, state)
    if time_index(o.address) == 0
        if isnothing(state)
            new = iterate(o.ods)
        else
            new = iterate(o.ods, state)
        end
        if isnothing(new)
            return nothing
        end
        (add, val), state = new
        new_val = -im*val*o.dt
        return ClockAddress(add, 1) => -0.5*new_val, state
    elseif time_index(o.address) == num_steps(o.clock)
        if isnothing(state)
            new = iterate(o.ods)
        else
            new = iterate(o.ods, state)
        end
        if isnothing(new)
            return nothing
        end
        (add, val), state = new
        new_val = im*val*o.dt
        return ClockAddress(add, time_index(o.address) - 1) => -0.5*new_val, state
    else
        if isnothing(state[1])
            new = iterate(o.ods)
        else
            new = iterate(o.ods, state[1])
        end
        if state[2]
            if isnothing(new)
                return ClockAddress(address(o.address), time_index(o.address) + 1) => -0.5*(1 - im*time_step(o.clock)*o.diag), (nothing, false)
            end
            (add, val), state1 = new
            state = (state1, true)
            new_val = im*val*o.dt
            return ClockAddress(add, time_index(o.address) - 1) => -0.5*new_val, state
        else
            if isnothing(new)
                return nothing
            end
            (add, val), state1 = new
            state = (state1, false)
            new_val = -im*val*o.dt
            return ClockAddress(add, time_index(o.address) + 1) => -0.5*new_val, state
        end
    end
end

function Rimu.random_offdiagonal(c::FirstOrderClockColumn)
    num = num_offdiagonals(c.ham_column)
    if rand() < 1/(num + 1)
        new_add = address(starting_address(c))
        val = 1 - im*time_step(parent_operator(c))*diagonal_element(c.ham_column)
        prob = 1/(num + 1)
        if time_index(starting_address(c)) == 0    
            new_add = ClockAddress(new_add, 1)
            return new_add, prob, -0.5*val
        elseif time_index(starting_address(c)) == num_steps(parent_operator(c))
            new_add = ClockAddress(new_add, time_index(starting_address(c)) - 1)
            return new_add, prob, -0.5*conj(val)
        else
            if rand() < 0.5
                new_add = ClockAddress(new_add, time_index(starting_address(c)) - 1)
                return new_add, 0.5*prob, -0.5*conj(val)
            else
                new_add = ClockAddress(new_add, time_index(starting_address(c)) + 1)
                return new_add, 0.5*prob, -0.5*val
            end
        end
    else
        new_add, prob, val = random_offdiagonal(c.ham_column)
        prob *= num/(num + 1)
        val *= -im*time_step(parent_operator(c))
        if time_index(starting_address(c)) == 0    
            new_add = ClockAddress(new_add, 1)
            return new_add, prob, -0.5*val
        elseif time_index(starting_address(c)) == num_steps(parent_operator(c))
            new_add = ClockAddress(new_add, time_index(starting_address(c)) - 1)
            return new_add, prob, -0.5*conj(val)
        else
            if rand() < 0.5
                new_add = ClockAddress(new_add, time_index(starting_address(c))- 1)
                return new_add, 0.5*prob, -0.5*conj(val)
            else
                new_add = ClockAddress(new_add, time_index(starting_address(c)) + 1)
                return new_add, 0.5*prob, -0.5*val
            end
        end
    end
end

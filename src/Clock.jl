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

function Base.isless(a::ClockAddress, b::ClockAddress)
    if time_index(a) == time_index(b)
        return address(a) < address(b)
    else
        return time_index(a) < time_index(b)
    end
end

"""
    Clock(u, length; start_at, penalty=1.0) <: AbstractHamiltonian{ComplexF64}

Clock Hamiltonian using time evolution operator `u` with `length` time steps. The optional
argument `start_at` specifies the state of the system at ``t=0``, otherwise this defaults
to `DVec(starting_address(u) => 1.0)`. `penalty` is the size of the multiplier on the
``t=0`` diagonal element that forces the initial state to remain as `start_at`.

Information about the clock is accessed with the following:

 * `time_evolution_operator(clock)` - the time evolution operator
 * `num_steps(clock)` - the number of time steps
 * `starting_state(clock)` - an `AbstractDVec` in the underlying Fock space, representing
 the state of the system at ``t=0``.
"""
struct Clock{U1,U2} <: AbstractHamiltonian{ComplexF64}
    time_evolution_op::U1
    adjoint_time_evolution_op::U2
    length::Int
    start_at::AbstractDVec
    penalty::Float64
end

function Clock(u, length; start_at=DVec(starting_address(u) => 1.0), penalty=1.0) 
    return Clock(u, u', length, start_at, penalty)
end

Rimu.allows_address_type(c::Clock, ::Type{<:ClockAddress{<:Any,<:Any,A}}) where {A} = allows_address_type(time_evolution_operator(c), A)
Rimu.starting_address(c::Clock) = ClockAddress(starting_address(time_evolution_operator(c)), 0)
Rimu.dimension(c::Clock, a) = (num_steps(c) + 1)*dimension(time_evolution_operator(c), address(a))

Rimu.LOStructure(::Clock) = IsHermitian()
Rimu.has_iterable_offdiagonals(::Type{<:Clock{U1,U2}}) where {U1,U2} = has_iterable_offdiagonals(U1) && has_iterable_offdiagonals(U2)
Rimu.has_random_offdiagonal(::Type{<:Clock{U1,U2}}) where {U1,U2} = has_random_offdiagonal(U1) && has_random_offdiagonal(U2)

time_evolution_operator(c::Clock) = c.time_evolution_op
num_steps(c::Clock) = c.length
starting_state(c::Clock) = c.start_at

struct ClockColumn{A<:ClockAddress,O<:Clock,C1<:AbstractOperatorColumn,C2<:AbstractOperatorColumn} <: AbstractOperatorColumn{A,ComplexF64,O}
    clock::O
    address::A
    u_column::C1
    ud_column::C2
end
function Rimu.operator_column(c::Clock, a::ClockAddress)
    return ClockColumn(c, a, operator_column(time_evolution_operator(c), address(a)), operator_column(c.adjoint_time_evolution_op, address(a)))
end

Rimu.starting_address(c::ClockColumn) = c.address
Rimu.parent_operator(c::ClockColumn) = c.clock


function Rimu.num_offdiagonals(c::ClockColumn)
    if time_index(starting_address(c)) == 0
        return num_offdiagonals(c.u_column) + 1
    elseif time_index(starting_address(c)) == num_steps(Rimu.parent_operator(c))
        return num_offdiagonals(c.ud_column) + 1
    else
        return num_offdiagonals(c.u_column) + num_offdiagonals(c.ud_column) + 2
    end
end

function Rimu.diagonal_element(c::ClockColumn)
    if time_index(starting_address(c)) == 0
        return 0.5 + c.clock.penalty*(1 - abs2(starting_state(Rimu.parent_operator(c))[address(starting_address(c))]))
    elseif time_index(starting_address(c)) == num_steps(Rimu.parent_operator(c))
        return 0.5
    else
        return 1
    end
end

function Rimu.random_offdiagonal(c::ClockColumn)
    num = min(100, num_offdiagonals(c.u_column))# if u is an ExponentialSampler, num_offdiagonals is infinite
    if rand() < 1/(num + 1)# diagonal of u or u†
        new_add = address(starting_address(c))
        prob = 1/(num + 1)
        if time_index(starting_address(c)) == 0
            val = diagonal_element(c.u_column)
            new_add = ClockAddress(new_add, 1)
            return new_add, prob, -0.5*val
        elseif time_index(starting_address(c)) == num_steps(Rimu.parent_operator(c))
            val = diagonal_element(c.ud_column)
            new_add = ClockAddress(new_add, time_index(starting_address(c)) - 1)
            return new_add, prob, -0.5*val
        else
            if rand() < 0.5
                new_add = ClockAddress(new_add, time_index(starting_address(c)) - 1)
                val = diagonal_element(c.ud_column)
                return new_add, 0.5*prob, -0.5*val
            else
                new_add = ClockAddress(new_add, time_index(starting_address(c)) + 1)
                val = diagonal_element(c.u_column)
                return new_add, 0.5*prob, -0.5*val
            end
        end
    else# offdiagonal of u or u†
        if time_index(starting_address(c)) == 0
            new_add, prob, val = random_offdiagonal(c.u_column)
            new_add = ClockAddress(new_add, 1)
            prob *= num/(num + 1)
            return new_add, prob, -0.5*val
        elseif time_index(starting_address(c)) == num_steps(Rimu.parent_operator(c))
            new_add, prob, val = random_offdiagonal(c.ud_column)
            new_add = ClockAddress(new_add, time_index(starting_address(c)) - 1)
            prob *= num/(num + 1)
            return new_add, prob, -0.5*val
        else
            if rand() < 0.5
                new_add, prob, val = random_offdiagonal(c.ud_column)
                new_add = ClockAddress(new_add, time_index(starting_address(c))- 1)
                prob *= num/(num + 1)
                return new_add, 0.5*prob, -0.5*val
            else
                new_add, prob, val = random_offdiagonal(c.u_column)
                new_add = ClockAddress(new_add, time_index(starting_address(c)) + 1)
                prob *= num/(num + 1)
                return new_add, 0.5*prob, -0.5*val
            end
        end
    end
end

struct ClockOffdiagonals{A<:ClockAddress,T,OD1,OD2}
    clock::Clock
    address::A
    diag_u::T
    diag_ud::T
    ods_u::OD1
    ods_ud::OD2
end

function Rimu.offdiagonals(c::ClockColumn)
    return ClockOffdiagonals(
        Rimu.parent_operator(c), starting_address(c), diagonal_element(c.u_column),
        diagonal_element(c.ud_column), offdiagonals(c.u_column), offdiagonals(c.ud_column))
end

Base.IteratorSize(::ClockOffdiagonals) = Base.SizeUnknown()
Base.eltype(::ClockOffdiagonals{A}) where {A} = Pair{A,ComplexF64}

struct ClockIterState{S1}
    s::Union{S1,Nothing}
    decreasing::Bool
end

function Base.iterate(o::ClockOffdiagonals)
    if time_index(o.address) == 0
        return ClockAddress(address(o.address), 1) => -0.5*o.diag_u, ClockIterState{Nothing}(nothing, false)
    elseif time_index(o.address) == num_steps(o.clock)
        return ClockAddress(address(o.address), time_index(o.address) - 1) => -0.5*o.diag_ud, ClockIterState{Nothing}(nothing, true)
    else
        return ClockAddress(address(o.address), time_index(o.address) - 1) => -0.5*o.diag_ud, ClockIterState{Nothing}(nothing, true)
    end
end

function Base.iterate(o::ClockOffdiagonals, state::ClockIterState{S1}) where {S1}
    if time_index(o.address) == 0
        if isnothing(state.s)
            new = iterate(o.ods_u)
            if isnothing(new)
                return nothing
            end
            (add, val), state1 = new
            return ClockAddress(add, 1) => -0.5*val, ClockIterState{typeof(state1)}(state1, false)
        else
            new = iterate(o.ods_u, state.s)
            if isnothing(new)
                return nothing
            end
            (add, val), state1 = new
            return ClockAddress(add, 1) => -0.5*val, ClockIterState{typeof(state1)}(state1, false)
        end
    elseif time_index(o.address) == num_steps(o.clock)
        if isnothing(state.s)
            new = iterate(o.ods_ud)
            if isnothing(new)
                return nothing
            end
            (add, val), state1 = new
            return ClockAddress(add, time_index(o.address) - 1) => -0.5*val, ClockIterState{typeof(state1)}(state1, true)
        else
            new = iterate(o.ods_ud, state.s)
            if isnothing(new)
                return nothing
            end
            (add, val), state1 = new
            return ClockAddress(add, time_index(o.address) - 1) => -0.5*val, ClockIterState{typeof(state1)}(state1, true)
        end
    else
        if state.decreasing
            if isnothing(state.s)
                new = iterate(o.ods_ud)
                if isnothing(new)
                    return ClockAddress(address(o.address), time_index(o.address) + 1) => -0.5*o.diag_u, ClockIterState{Nothing}(nothing, false)
                end
                (add, val), state1 = new
                return ClockAddress(add, time_index(o.address) - 1) => -0.5*val, ClockIterState{typeof(state1)}(state1, true)
            else
                new = iterate(o.ods_ud, state.s)
                if isnothing(new)
                    return ClockAddress(address(o.address), time_index(o.address) + 1) => -0.5*o.diag_u, ClockIterState{Nothing}(nothing, false)
                end
                (add, val), state1 = new
                return ClockAddress(add, time_index(o.address) - 1) => -0.5*val, ClockIterState{typeof(state1)}(state1, true)
            end
        else
            if isnothing(state.s)
                new = iterate(o.ods_u)
                if isnothing(new)
                    return nothing
                end
                (add, val), state1 = new
                return ClockAddress(add, time_index(o.address) + 1) => -0.5*val, ClockIterState{typeof(state1)}(state1, false)
            else
                new = iterate(o.ods_u, state.s)
                if isnothing(new)
                    return nothing
                end
                (add, val), state1 = new
                return ClockAddress(add, time_index(o.address) + 1) => -0.5*val, ClockIterState{typeof(state1)}(state1, false)
            end
            
        end
    end
end

"""
    ClockOperator(op::AbstractOperator, t::Int) <: AbstractOperator

Operator wrapper for use in replica strategy `AllOverlaps` with [`Clock`](@ref)
Hamiltonians. Observable expectation values are calculated using a `ClockOperator` for the
numerator and a [`ClockProjector`](@ref) with the same time index for the denominator.
"""
struct ClockOperator{T, O<:AbstractOperator{T}} <: AbstractOperator{T}
    op::O
    t::Int
end

Rimu.allows_address_type(o::ClockOperator, ::Type{ClockAddress{<:Any,<:Any,A}}) where {A} = allows_address_type(o.op, A)

time_index(o::ClockOperator) = o.t

struct ClockOperatorColumn{A<:ClockAddress,T,O<:ClockOperator{T},C} <: AbstractOperatorColumn{A,T,O}
    op::O
    address::A
    col::C
end
Rimu.operator_column(o::ClockOperator, a) = ClockOperatorColumn(o, a, operator_column(o.op, address(a)))

Rimu.parent_operator(c::ClockOperatorColumn) = c.op
Rimu.starting_address(c::ClockOperatorColumn) = c.address

function Rimu.diagonal_element(c::ClockOperatorColumn)
    if time_index(starting_address(c)) != time_index(Rimu.parent_operator(c))
        return 0
    else
        return diagonal_element(c.col)
    end
end

struct ClockOperatorOffdiagonals{O}
    ods::O
    t_op::Int
    t_add::Int
end

function Rimu.offdiagonals(c::ClockOperatorColumn)
    return ClockOperatorOffdiagonals(offdiagonals(c.col), time_index(Rimu.parent_operator(c)), time_index(c.address))
end

function Base.iterate(o::ClockOperatorOffdiagonals)
    first = iterate(o.ods)
    if isnothing(first) || o.t_op != o.t_add
        return nothing
    end
    (add, val), state = first
    return ClockAddress(add, o.t_op) => val, state
end

function Base.iterate(o::ClockOperatorOffdiagonals, state)
    next = iterate(o.ods, state)
    if isnothing(next)
        return nothing
    end
    (add, val), state = next
    return ClockAddress(add, o.t_op) => val, state
end

"""
    ClockProjector <: ClockOperator

Operator to calculate vector-vector overlaps at time step `t`.
"""
function ClockProjector(t)
    return ClockOperator(IdentityOperator(), t)
end

"""
    ClockObservable(op::AbstractObservable, t::Int) <: AbstractObservable

Observable for use in replica strategy AllOverlaps with Clock Hamiltonians.
"""
struct ClockObservable{T, O} <: AbstractObservable{T}
    op::O
    t::Int
end

ClockObservable(o::AbstractOperator, t) = ClockOperator(o, t)

function Rimu.Interfaces.dot_from_right(x, obs::ClockObservable, y)
    xt = DVec(address(add) => val for (add, val) in pairs(x) if time_index(add) == obs.t)
    yt = DVec(address(add) => val for (add, val) in pairs(y) if time_index(add) == obs.t)
    return dot(xt, obs.op, yt)
end

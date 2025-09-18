struct fSim{A<:FermiFS} <: AbstractHamiltonian{ComplexF64}
    θ::Float64
    ϕ::Float64
    even::Bool
    address::A
end

"""
    fSim(address::FermiFS, θ, ϕ) <: AbstractHamiltonian{ComplexF64}

Time evolution operator for application of fSim(θ, ϕ) gates to all neighboring pairs of
qubits in a chain, where the number of modes in `address` is the number of qubits. The
gates are first applied to odd pairs ((1,2),(3,4),...) and then to even pairs ((2,3),...).
In the small angle limit, this maps to an XXZ spin chain, with anisotropy parameter
``\\Delta = \\sin(\\phi/2)/\\sin(\\theta)``.
"""
fSim(address::FermiFS, θ, ϕ) = fSim(θ, ϕ, true, address)*fSim(θ, ϕ, false, address)

Rimu.starting_address(u::fSim) = u.address

Rimu.LOStructure(::fSim) = AdjointKnown()

Rimu.adjoint(u::fSim) = fSim(-u.θ, -u.ϕ, u.even, u.address)

struct fSimColumn{M,O<:fSim,A<:FermiFS{<:Any,M},S} <: AbstractOperatorColumn{A,ComplexF64,O}
    op::O
    address::A
    θ::Float64
    ϕ::Float64
    onr::S
    even::Bool
end
function Rimu.operator_column(op::fSim, address)
    return fSimColumn(op, address, op.θ, op.ϕ, onr(address), op.even)
end

Rimu.starting_address(c::fSimColumn) = c.address
Rimu.parent_operator(c::fSimColumn) = c.op
Rimu.num_offdiagonals(c::fSimColumn{M}) where {M} = 2^((M-c.even)÷2)-1

function Rimu.diagonal_element(c::fSimColumn{M}) where {M}
    result = 1.0
    for i in 1+c.even:2:M-1
        if c.onr[i] == 1
            if c.onr[i+1] == 1
                result *= exp(-im*c.ϕ)
            else
                result *= cos(c.θ)
            end
        else
            if c.onr[i+1] == 1
                result *= cos(c.θ)
            end
        end
    end
    return result
end

function Rimu.random_offdiagonal(c::fSimColumn{M}) where {M}
    newadd = starting_address(c)
    val = 1.0+0.0im
    diag = true
    flips = rand(1:2^((M-c.even)÷2)-1)
    for pair in 1:(M-c.even)÷2
        site1 = 2*pair - 1 + c.even
        site2 = 2*pair + c.even
        if c.onr[site1] == 1 && c.onr[site2] == 1
            val *= exp(-im*c.ϕ)
        elseif flips & 2^(pair-1) == 2^(pair-1)
            if c.onr[site1] == 1 && c.onr[site2] == 0
                newadd, v = excitation(newadd, (find_mode(c.address, site2),), (find_mode(c.address, site1),))
                val *= im*sin(c.θ)
                diag = false
            elseif c.onr[site2] == 1 && c.onr[site1] == 0
                newadd, v = excitation(newadd, (find_mode(c.address, site1),), (find_mode(c.address, site2),))
                val *= im*sin(c.θ)
                diag = false
            end
        end
    end
    if diag
        return newadd, 1/(2^((M-c.even)÷2)-1), 0.0im
    else
        return newadd, 1/(2^((M-c.even)÷2)-1), val
    end
end

struct fSimOD{M,A<:FermiFS{<:Any,M},S}
    address::A
    θ::Float64
    ϕ::Float64
    onr::S
    even::Bool
end
Rimu.offdiagonals(c::fSimColumn) = fSimOD(c.address, c.θ, c.ϕ, c.onr, c.even)

Base.IteratorSize(::fSimOD{M}) where {M} = Base.HasLength()
Base.eltype(::fSimOD{<:Any,A}) where {A} = Pair{A,ComplexF64}
Base.size(::fSimOD{M}) where {M} = Base.SizeUnknown()

function Base.iterate(o::fSimOD{M}, flips=0) where {M}
    flips += 1
    if flips >= 2^((M-o.even)÷2)
        return nothing
    end
    newadd = o.address
    val = 1.0+0.0im
    diag = true
    repeat = false
    for pair in 1:(M-o.even)÷2
        site1 = 2*pair - 1 + o.even
        site2 = 2*pair + o.even
        if o.onr[site1] == 1 && o.onr[site2] == 1
            val *= exp(-im*o.ϕ)
            if flips & 2^(pair-1) == 2^(pair-1)
                repeat = true
                break
            end
        elseif o.onr[site1] == 0 && o.onr[site2] == 0 && flips & 2^(pair-1) == 2^(pair-1)
            repeat = true
            break
        elseif o.onr[site1] == 1 && o.onr[site2] == 0
            if flips & 2^(pair-1) == 2^(pair-1)
                newadd, v = excitation(newadd, (find_mode(o.address, site2),), (find_mode(o.address, site1),))
                val *= im*sin(o.θ)
                diag = false
            else
                val *= cos(o.θ)
            end
        elseif o.onr[site2] == 1 && o.onr[site1] == 0
            if flips & 2^(pair-1) == 2^(pair-1)
                newadd, v = excitation(newadd, (find_mode(o.address, site1),), (find_mode(o.address, site2),))
                val *= im*sin(o.θ)
                diag = false
            else
                val *= cos(o.θ)
            end
        end
    end
    if diag || repeat
        return newadd => 0.0im, flips
    else
        return newadd => val, flips
    end
end

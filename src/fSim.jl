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
Rimu.num_offdiagonals(c::fSimColumn) = 2*num_occupied_modes(c.address)

function Rimu.diagonal_element(c::fSimColumn{M}) where {M}
    result = 1
    for i in 1+c.even:2:M-1
        if c.onr[i] == 1
            if c.onr[i+1] == 1
                result *= exp(im*c.ϕ)
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
    pair = rand(1:(M-c.even)÷2)
    site1 = 2*pair - 1 + c.even
    site2 = 2*pair + c.even
    if c.onr[site1] == 1 && c.onr[site2] == 0
        #newadd, v = hopnextneighbour(c.address, 2*site1 - 1, :hard_wall)
        newadd, v = excitation(c.address, (find_mode(c.address, site2),), (find_mode(c.address, site1),))
        return newadd, 1/((M-c.even)÷2), im*sin(c.θ)
    elseif c.onr[site2] == 1 && c.onr[site1] == 0
        #newadd, v = hopnextneighbour(c.address, 2*site2, :hard_wall)
        newadd, v = excitation(c.address, (find_mode(c.address, site1),), (find_mode(c.address, site2),))
        return newadd, 1/((M-c.even)÷2), im*sin(c.θ)
    else
        return c.address, 1/((M-c.even)÷2), 0.0
    end
end

struct fSimOD{M,A<:FermiFS{<:Any,M},S}
    address::A
    θ::Float64
    onr::S
    even::Bool
end
Rimu.offdiagonals(c::fSimColumn) = fSimOD(c.address, c.θ, c.onr, c.even)

Base.IteratorSize(::fSimOD{M}) where {M} = Base.HasLength()
Base.eltype(::fSimOD{<:Any,A}) where {A} = Pair{A,ComplexF64}
Base.size(o::fSimOD{M}) where {M} = ((M - o.even)÷2,)

function Base.iterate(o::fSimOD{M}) where {M}
    pair = 1
    if 2*pair > M - o.even
        return nothing
    end
    site1 = 2*pair - 1 + o.even
    site2 = 2*pair + o.even
    if o.onr[site1] == 1 && o.onr[site2] == 0
        newadd, v = excitation(o.address, (find_mode(o.address, site2),), (find_mode(o.address, site1),))
        return newadd => im*sin(o.θ), pair
    elseif o.onr[site2] == 1 && o.onr[site1] == 0
        newadd, v = excitation(o.address, (find_mode(o.address, site1),), (find_mode(o.address, site2),))
        return newadd => im*sin(o.θ), pair
    else
        return o.address => 0.0, pair
    end
end

function Base.iterate(o::fSimOD{M}, pair) where {M}
    pair += 1
    if 2*pair > M - o.even
        return nothing
    end
    site1 = 2*pair - 1 + o.even
    site2 = 2*pair + o.even
    if o.onr[site1] == 1 && o.onr[site2] == 0
        newadd, v = excitation(o.address, (find_mode(o.address, site2),), (find_mode(o.address, site1),))
        return newadd => im*sin(o.θ), pair
    elseif o.onr[site2] == 1 && o.onr[site1] == 0
        newadd, v = excitation(o.address, (find_mode(o.address, site1),), (find_mode(o.address, site2),))
        return newadd => im*sin(o.θ), pair
    else
        return o.address => 0.0, pair
    end
end

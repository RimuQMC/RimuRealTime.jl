"""
    GPAnsatz(hamiltonian_or_address; valtype = Float64) <: AbstractAnsatz

Ansatz representing a Gross-Pitaevskii product state of ``N`` bosons in ``M`` modes,
with single-particle orbital parameter vector ``𝐜 = (c_1, …, c_M)``:

```math
|Ψ_{\\mathrm{GP}}⟩ = \\frac{1}{\\sqrt{N!}} \\left( ∑_{m=1}^M cₘ aₘ^† \\right)^N |0⟩
```

Projected onto a Fock basis state ``|n_1, …, n_M⟩``, the amplitude is:

```math
⟨n₁, …, n_M | Ψ_{\\mathrm{GP}}⟩ = \\sqrt{\\frac{N!}{\\prod_{m=1}^M nₘ!}} \\prod_{m=1}^M cₘ^{nₘ}
```
"""
struct GPAnsatz{A,V,N,H} <: AbstractAnsatz{A,V,N}
    hamiltonian::H  
    N_fact::Float64     

    function GPAnsatz{V}(hamiltonian_or_address) where {V}
        addr = _starting_address(hamiltonian_or_address)
        addr isa SingleComponentFockAddress || throw(ArgumentError(
            "GPAnsatz requires a single-component Fock address (e.g. BoseFS)"
        ))
        M = num_modes(addr)
        N_fact = Float64(factorial(big(num_particles(addr))))
        return new{typeof(addr),V,M,typeof(hamiltonian_or_address)}(hamiltonian_or_address, N_fact)
    end 
end

# default constructor with Float64 valtype
GPAnsatz(hamiltonian_or_address; valtype = Float64) = GPAnsatz{valtype}(hamiltonian_or_address)

# helper functions to extract starting address from Hamiltonian or Fock address
_starting_address(h::AbstractHamiltonian) = starting_address(h)
_starting_address(addr::AbstractFockAddress) = addr

Rimu.starting_address(gpe::GPAnsatz) = _starting_address(gpe.hamiltonian)
Rimu.build_basis(gpe::GPAnsatz) = build_basis(gpe.hamiltonian)

Base.show(io::IO, gpe::GPAnsatz{A,V,N}) where {A,V,N} = 
    print(io, "GPAnsatz{$V, modes=$N}($(gpe.hamiltonian))")

# evaluate GP ansatz amplitude
function (gpe::GPAnsatz)(addr, params)
    ref_addr = _starting_address(gpe.hamiltonian)
    if num_particles(addr) != num_particles(ref_addr) || num_modes(addr) != num_modes(ref_addr)
        return zero(valtype(gpe))
    end

    orb_prod = one(promote_type(valtype(gpe), eltype(params)))
    for (n, m, _) in occupied_modes(addr)
        c_m = params[m]
        iszero(c_m) && return zero(orb_prod)
        orb_prod *= c_m^n
    end
    # return coefficient sqrt(N! / prod(n_m!)) * prod(c_m^n_m)
    return sqrt(gpe.N_fact * Gutzwiller.multinomial_weight(addr)) * orb_prod
end

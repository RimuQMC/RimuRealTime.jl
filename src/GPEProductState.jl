module GPEProductState

using Rimu
using RimuRealTime
using LinearAlgebra

export gpe_product_state

"""
    gpe_product_state(orbital, address; style = default_style(ComplexF64))
    gpe_product_state(orbital, ::Type{BoseFS{N,M}}; kwargs...)
    gpe_product_state(orbital, hamiltonian; kwargs...)

Construct the Gross-Pitaevskii product state |Ψ_GPE⟩ as a `PDVec` over the
Fock basis of `address`:

    |Ψ_GPE⟩ = ∑_n C_n |n₁, n₂, …, n_M⟩

where the projection amplitudes are given by

    C_n = √(N! / (n₁! n₂! ⋯ n_M!)) ∏_{k=1}^M c_k^{n_k}
"""
function gpe_product_state(
    orbital::AbstractVector,
    address::BoseFS{N,M};
    style = default_style(ComplexF64)
) where {N,M}
    length(orbital) == M || throw(DimensionMismatch("orbital length $(length(orbital)) != modes $M"))

    c = ComplexF64.(orbital ./ norm(orbital))

    # Precompute single-mode factors c_m^n / sqrt(n!)
    factors = Matrix{ComplexF64}(undef, M, N + 1)
    for m in 1:M
        factors[m, 1] = 1.0
        for n in 1:N
            factors[m, n + 1] = factors[m, n] * c[m] / sqrt(n)
        end
    end

    basis = ExactDiagonalization.build_basis(address)
    amplitudes = Dict{eltype(basis), ComplexF64}()
    sizehint!(amplitudes, length(basis))

    for b in basis
        amp = one(ComplexF64)
        for (n, m, _) in occupied_modes(b)
            amp *= factors[m, n + 1]
        end
        if !iszero(amp)
            amplitudes[b] = amp
        end
    end

    state = PDVec(amplitudes; style = style)
    normalize!(state)
    return state
end

gpe_product_state(orbital::AbstractVector, ::Type{<:BoseFS{N,M}}; kwargs...) where {N,M} =
    gpe_product_state(orbital, near_uniform(BoseFS{N,M}); kwargs...)

gpe_product_state(orbital::AbstractVector, h::AbstractHamiltonian; kwargs...) =
    gpe_product_state(orbital, starting_address(h); kwargs...)

end # module GPEProductState


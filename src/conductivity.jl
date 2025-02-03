using KernelAbstractions
using UnPack

"""
When using the diffusion solver to solve the heat equation, this can be used to
update the thermal conductivity of the mesh at each cell-center.

# Arguments
 - `diffusivity::AbstractArray`: diffusivity of the problem
 - `temp::AbstractArray`: temperature
 - `ρ::AbstractArray`: density
 - `κ::Function`: function to determine thermal conductivity, e.g. κ(ρ,T) = κ0 * ρ * T^(5/2)
 - `cₚ::Real`: heat capacity at constant pressure
"""
function update_conductivity!(
  scheme, mesh, temperature, density, cₚ::Real, κ::F
) where {F<:Function}
  @unpack diff_domain, domain = _domain_pairs(scheme, mesh)

  α = @view scheme.α[diff_domain]
  T = @view temperature[domain]
  ρ = @view density[domain]

  backend = scheme.backend
  conductivity_kernel(backend)(α, T, ρ, cₚ, κ; ndrange=size(α))

  # # KernelAbstractions.synchronize(backend)

  return nothing
end

function update_conductivity!(
  scheme, mesh, temperature, density, cₚ::AbstractArray, κ::F
) where {F<:Function}
  @unpack diff_domain, domain = _domain_pairs(scheme, mesh)

  α = @view scheme.α[diff_domain]
  T = @view temperature[domain]
  _cₚ = @view cₚ[domain]
  ρ = @view density[domain]

  backend = scheme.backend
  conductivity_kernel(backend)(α, T, ρ, _cₚ, κ; ndrange=size(α))

  # # KernelAbstractions.synchronize(backend)

  return nothing
end

function _domain_pairs(scheme::PseudoTransientSolver, mesh)
  diff_domain = scheme.iterators.full.cartesian
  domain = mesh.iterators.cell.full

  return (; diff_domain, domain)
end

function _domain_pairs(scheme::ImplicitScheme, mesh)
  diff_domain = scheme.iterators.full.cartesian
  domain = scheme.iterators.mesh

  return (; diff_domain, domain)
end

# conductivity with array-valued cₚ
@kernel function conductivity_kernel(
  α, temperature, density, cₚ::AbstractArray{T,N}, κ::F
) where {T,N,F<:Function}
  idx = @index(Global)

  @inbounds begin
    α[idx] = κ(density[idx], temperature[idx]) / (density[idx] * cₚ[idx])
  end
end

# conductivity with single-valued cₚ
@kernel function conductivity_kernel(
  α, temperature, density, cₚ::Real, κ::F
) where {F<:Function}
  idx = @index(Global)

  @inbounds begin
    α[idx] = abs(κ(density[idx], temperature[idx]) / (density[idx] * cₚ))
  end
end

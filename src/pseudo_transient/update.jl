
@kernel inbounds = true function _update_kernel!(
  u,
  @Const(u_prev),
  cache,
  cell_center_metrics, # applying @Const to a struct array causes problems
  @Const(flux),
  @Const(dτ_ρ),
  @Const(source_term),
  @Const(dt),
  @Const(I0),
)
  idx = @index(Global, Cartesian)
  idx += I0

  @inline ∇q = flux_divergence(flux, cache, cell_center_metrics, idx)

  u[idx] = (
    (u[idx] + dτ_ρ[idx] * (u_prev[idx] / dt - ∇q + source_term[idx])) / (1 + dτ_ρ[idx] / dt)
  )
end

@kernel inbounds = true function _update_kernel_tz!(
  u,
  @Const(u_prev),
  cache,
  θr_dτ,
  α,
  cell_center_metrics, # applying @Const to a struct array causes problems
  coords,
  @Const(nhalo),
  @Const(nnodes),
  @Const(flux),
  @Const(dτ_ρ),
  @Const(source_term),
  @Const(dt),
  @Const(ρ),
  @Const(cₚ),
  @Const(I0),
  @Const(t),
  @Const(tz_type),
  @Const(tz_dim),
  @Const(no_t),
)
  idx = @index(Global, Cartesian)
  idx += I0

  @inline ∇q = flux_divergence(flux, cache, cell_center_metrics, idx)

  @inline f = define_forcing(
    cache,
    cell_center_metrics,
    coords,
    nhalo,
    nnodes,
    α,
    ρ,
    cₚ,
    source_term,
    t,
    tz_type,
    tz_dim,
    no_t,
    idx,
  )

  u[idx] = (
    (u[idx] + dτ_ρ[idx] * (u_prev[idx] / dt - ∇q + source_term[idx] + f)) /
    (1 + dτ_ρ[idx] / dt)
  )
end

function compute_update!(solver::PseudoTransientSolver{N,T}, mesh, Δt; ρ, cₚ, t=0, tz=false, tz_dim=1, no_t=true) where {N,T}
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  if tz == false
    _update_kernel!(solver.backend)(
      solver.u,
      solver.u_prev,
      solver.cache,
      mesh.cell_center_metrics,
      solver.q,
      solver.dτ_ρ,
      solver.source_term,
      Δt,
      idx_offset;
      ndrange=size(domain),
    )
  else
    tz_type = :wavy
    _update_kernel_tz!(solver.backend)(
      solver.u,
      solver.u_prev,
      solver.cache,
      solver.θr_dτ,
      solver.α,
      mesh.cell_center_metrics,
      mesh.centroid_coordinates,
      mesh.nhalo,
      mesh.nnodes,
      solver.q,
      solver.dτ_ρ,
      solver.source_term,
      Δt,
      ρ,
      cₚ,
      idx_offset,
      t,
      tz_type,
      tz_dim,
      no_t;
      ndrange=size(domain),
    )
  end

  # KernelAbstractions.synchronize(solver.backend)
  return nothing
end

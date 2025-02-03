
function update_metric_cache!(solver, mesh::CurvilinearGrid2D)

  #
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _update_2d_metric_cache!(solver.backend)(
    solver.cache.α,
    solver.cache.β,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    idx_offset;
    ndrange=size(domain),
  )

  # KernelAbstractions.synchronize(solver.backend)
  return nothing
end

function update_metric_cache!(solver, mesh::CurvilinearGrid3D)   #
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  _update_3d_metric_cache!(solver.backend)(
    solver.cache.α,
    solver.cache.β,
    solver.cache.γ,
    mesh.cell_center_metrics,
    mesh.edge_metrics,
    idx_offset;
    ndrange=size(domain),
  )

  # KernelAbstractions.synchronize(solver.backend)
  return nothing
end

@kernel inbounds = true function _update_2d_metric_cache!(
  αᵢⱼ, βᵢⱼ, cell_center_metrics, edge_metrics, I0
)

  #
  idx = @index(Global, Cartesian)
  idx += I0

  idim, jdim, = (1, 2)
  ᵢ₋₁ = shift(idx, idim, -1)
  ⱼ₋₁ = shift(idx, jdim, -1)

  Jᵢ₊½ = edge_metrics.i₊½.J[idx]
  Jⱼ₊½ = edge_metrics.j₊½.J[idx]

  Jᵢ₋½ = edge_metrics.i₊½.J[ᵢ₋₁]
  Jⱼ₋½ = edge_metrics.j₊½.J[ⱼ₋₁]

  ξx = cell_center_metrics.ξ.x₁[idx]
  ξy = cell_center_metrics.ξ.x₂[idx]

  ηx = cell_center_metrics.η.x₁[idx]
  ηy = cell_center_metrics.η.x₂[idx]

  ξxᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[idx] / Jᵢ₊½
  ξyᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[idx] / Jᵢ₊½

  ξxᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[ᵢ₋₁] / Jᵢ₋½
  ξyᵢ₋½ = edge_metrics.i₊½.ξ̂.x₂[ᵢ₋₁] / Jᵢ₋½

  ηxᵢ₊½ = edge_metrics.i₊½.η̂.x₁[idx] / Jᵢ₊½
  ηyᵢ₊½ = edge_metrics.i₊½.η̂.x₂[idx] / Jᵢ₊½

  ηxᵢ₋½ = edge_metrics.i₊½.η̂.x₁[ᵢ₋₁] / Jᵢ₋½
  ηyᵢ₋½ = edge_metrics.i₊½.η̂.x₂[ᵢ₋₁] / Jᵢ₋½

  ξxⱼ₊½ = edge_metrics.j₊½.ξ̂.x₁[idx] / Jⱼ₊½
  ξyⱼ₊½ = edge_metrics.j₊½.ξ̂.x₂[idx] / Jⱼ₊½

  ξxⱼ₋½ = edge_metrics.j₊½.ξ̂.x₁[ⱼ₋₁] / Jⱼ₋½
  ξyⱼ₋½ = edge_metrics.j₊½.ξ̂.x₂[ⱼ₋₁] / Jⱼ₋½

  ηxⱼ₊½ = edge_metrics.j₊½.η̂.x₁[idx] / Jⱼ₊½
  ηyⱼ₊½ = edge_metrics.j₊½.η̂.x₂[idx] / Jⱼ₊½

  ηxⱼ₋½ = edge_metrics.j₊½.η̂.x₁[ⱼ₋₁] / Jⱼ₋½
  ηyⱼ₋½ = edge_metrics.j₊½.η̂.x₂[ⱼ₋₁] / Jⱼ₋½

  αᵢⱼ[idx] = (
    ξx * (ξxᵢ₊½ - ξxᵢ₋½) +
    ξy * (ξyᵢ₊½ - ξyᵢ₋½) +
    #
    ηx * (ξxⱼ₊½ - ξxⱼ₋½) +
    ηy * (ξyⱼ₊½ - ξyⱼ₋½)
  )

  βᵢⱼ[idx] = (
    ξx * (ηxᵢ₊½ - ηxᵢ₋½) +
    ξy * (ηyᵢ₊½ - ηyᵢ₋½) +
    #
    ηx * (ηxⱼ₊½ - ηxⱼ₋½) +
    ηy * (ηyⱼ₊½ - ηyⱼ₋½)
  )
end

@kernel inbounds = true function _update_3d_metric_cache!(
  αᵢⱼₖ, βᵢⱼₖ, γᵢⱼₖ, cell_center_metrics, edge_metrics, I0
)

  #
  idx = @index(Global, Cartesian)
  idx += I0

  idim, jdim, kdim = (1, 2, 3)
  ᵢ₋₁ = shift(idx, idim, -1)
  ⱼ₋₁ = shift(idx, jdim, -1)
  ₖ₋₁ = shift(idx, kdim, -1)

  Jᵢ₊½ = edge_metrics.i₊½.J[idx]
  Jⱼ₊½ = edge_metrics.j₊½.J[idx]
  Jₖ₊½ = edge_metrics.k₊½.J[idx]

  Jᵢ₋½ = edge_metrics.i₊½.J[ᵢ₋₁]
  Jⱼ₋½ = edge_metrics.j₊½.J[ⱼ₋₁]
  Jₖ₋½ = edge_metrics.k₊½.J[ₖ₋₁]

  ξx = cell_center_metrics.ξ.x₁[idx]
  ξy = cell_center_metrics.ξ.x₂[idx]
  ξz = cell_center_metrics.ξ.x₃[idx]

  ηx = cell_center_metrics.η.x₁[idx]
  ηy = cell_center_metrics.η.x₂[idx]
  ηz = cell_center_metrics.η.x₃[idx]

  ζx = cell_center_metrics.ζ.x₁[idx]
  ζy = cell_center_metrics.ζ.x₂[idx]
  ζz = cell_center_metrics.ζ.x₃[idx]

  ξxᵢ₊½ = edge_metrics.i₊½.ξ̂.x₁[idx] / Jᵢ₊½
  ξyᵢ₊½ = edge_metrics.i₊½.ξ̂.x₂[idx] / Jᵢ₊½
  ξzᵢ₊½ = edge_metrics.i₊½.ξ̂.x₃[idx] / Jᵢ₊½

  ξxᵢ₋½ = edge_metrics.i₊½.ξ̂.x₁[ᵢ₋₁] / Jᵢ₋½
  ξyᵢ₋½ = edge_metrics.i₊½.ξ̂.x₂[ᵢ₋₁] / Jᵢ₋½
  ξzᵢ₋½ = edge_metrics.i₊½.ξ̂.x₃[ᵢ₋₁] / Jᵢ₋½

  ηxᵢ₊½ = edge_metrics.i₊½.η̂.x₁[idx] / Jᵢ₊½
  ηyᵢ₊½ = edge_metrics.i₊½.η̂.x₂[idx] / Jᵢ₊½
  ηzᵢ₊½ = edge_metrics.i₊½.η̂.x₃[idx] / Jᵢ₊½

  ηxᵢ₋½ = edge_metrics.i₊½.η̂.x₁[ᵢ₋₁] / Jᵢ₋½
  ηyᵢ₋½ = edge_metrics.i₊½.η̂.x₂[ᵢ₋₁] / Jᵢ₋½
  ηzᵢ₋½ = edge_metrics.i₊½.η̂.x₃[ᵢ₋₁] / Jᵢ₋½

  ζxᵢ₊½ = edge_metrics.i₊½.ζ̂.x₁[idx] / Jᵢ₊½
  ζyᵢ₊½ = edge_metrics.i₊½.ζ̂.x₂[idx] / Jᵢ₊½
  ζzᵢ₊½ = edge_metrics.i₊½.ζ̂.x₃[idx] / Jᵢ₊½

  ζxᵢ₋½ = edge_metrics.i₊½.ζ̂.x₁[ᵢ₋₁] / Jᵢ₋½
  ζyᵢ₋½ = edge_metrics.i₊½.ζ̂.x₂[ᵢ₋₁] / Jᵢ₋½
  ζzᵢ₋½ = edge_metrics.i₊½.ζ̂.x₃[ᵢ₋₁] / Jᵢ₋½

  ξxⱼ₊½ = edge_metrics.j₊½.ξ̂.x₁[idx] / Jⱼ₊½
  ξyⱼ₊½ = edge_metrics.j₊½.ξ̂.x₂[idx] / Jⱼ₊½
  ξzⱼ₊½ = edge_metrics.j₊½.ξ̂.x₃[idx] / Jⱼ₊½

  ξxⱼ₋½ = edge_metrics.j₊½.ξ̂.x₁[ⱼ₋₁] / Jⱼ₋½
  ξyⱼ₋½ = edge_metrics.j₊½.ξ̂.x₂[ⱼ₋₁] / Jⱼ₋½
  ξzⱼ₋½ = edge_metrics.j₊½.ξ̂.x₃[ⱼ₋₁] / Jⱼ₋½

  ηxⱼ₊½ = edge_metrics.j₊½.η̂.x₁[idx] / Jⱼ₊½
  ηyⱼ₊½ = edge_metrics.j₊½.η̂.x₂[idx] / Jⱼ₊½
  ηzⱼ₊½ = edge_metrics.j₊½.η̂.x₃[idx] / Jⱼ₊½

  ηxⱼ₋½ = edge_metrics.j₊½.η̂.x₁[ⱼ₋₁] / Jⱼ₋½
  ηyⱼ₋½ = edge_metrics.j₊½.η̂.x₂[ⱼ₋₁] / Jⱼ₋½
  ηzⱼ₋½ = edge_metrics.j₊½.η̂.x₃[ⱼ₋₁] / Jⱼ₋½

  ζxⱼ₊½ = edge_metrics.j₊½.ζ̂.x₁[idx] / Jⱼ₊½
  ζyⱼ₊½ = edge_metrics.j₊½.ζ̂.x₂[idx] / Jⱼ₊½
  ζzⱼ₊½ = edge_metrics.j₊½.ζ̂.x₃[idx] / Jⱼ₊½

  ζxⱼ₋½ = edge_metrics.j₊½.ζ̂.x₁[ⱼ₋₁] / Jⱼ₋½
  ζyⱼ₋½ = edge_metrics.j₊½.ζ̂.x₂[ⱼ₋₁] / Jⱼ₋½
  ζzⱼ₋½ = edge_metrics.j₊½.ζ̂.x₃[ⱼ₋₁] / Jⱼ₋½

  ξxₖ₊½ = edge_metrics.k₊½.ξ̂.x₁[idx] / Jₖ₊½
  ξyₖ₊½ = edge_metrics.k₊½.ξ̂.x₂[idx] / Jₖ₊½
  ξzₖ₊½ = edge_metrics.k₊½.ξ̂.x₃[idx] / Jₖ₊½

  ξxₖ₋½ = edge_metrics.k₊½.ξ̂.x₁[ₖ₋₁] / Jₖ₋½
  ξyₖ₋½ = edge_metrics.k₊½.ξ̂.x₂[ₖ₋₁] / Jₖ₋½
  ξzₖ₋½ = edge_metrics.k₊½.ξ̂.x₃[ₖ₋₁] / Jₖ₋½

  ηxₖ₊½ = edge_metrics.k₊½.η̂.x₁[idx] / Jₖ₊½
  ηyₖ₊½ = edge_metrics.k₊½.η̂.x₂[idx] / Jₖ₊½
  ηzₖ₊½ = edge_metrics.k₊½.η̂.x₃[idx] / Jₖ₊½

  ηxₖ₋½ = edge_metrics.k₊½.η̂.x₁[ₖ₋₁] / Jₖ₋½
  ηyₖ₋½ = edge_metrics.k₊½.η̂.x₂[ₖ₋₁] / Jₖ₋½
  ηzₖ₋½ = edge_metrics.k₊½.η̂.x₃[ₖ₋₁] / Jₖ₋½

  ζxₖ₊½ = edge_metrics.k₊½.ζ̂.x₁[idx] / Jₖ₊½
  ζyₖ₊½ = edge_metrics.k₊½.ζ̂.x₂[idx] / Jₖ₊½
  ζzₖ₊½ = edge_metrics.k₊½.ζ̂.x₃[idx] / Jₖ₊½

  ζxₖ₋½ = edge_metrics.k₊½.ζ̂.x₁[ₖ₋₁] / Jₖ₋½
  ζyₖ₋½ = edge_metrics.k₊½.ζ̂.x₂[ₖ₋₁] / Jₖ₋½
  ζzₖ₋½ = edge_metrics.k₊½.ζ̂.x₃[ₖ₋₁] / Jₖ₋½

  αᵢⱼₖ[idx] = (
    ξx * (ξxᵢ₊½ - ξxᵢ₋½) +
    ξy * (ξyᵢ₊½ - ξyᵢ₋½) +
    ξz * (ξzᵢ₊½ - ξzᵢ₋½) +
    #
    ηx * (ξxⱼ₊½ - ξxⱼ₋½) +
    ηy * (ξyⱼ₊½ - ξyⱼ₋½) +
    ηz * (ξzⱼ₊½ - ξzⱼ₋½) +
    #
    ζx * (ξxₖ₊½ - ξxₖ₋½) +
    ζy * (ξyₖ₊½ - ξyₖ₋½) +
    ζz * (ξzₖ₊½ - ξzₖ₋½)
  )

  βᵢⱼₖ[idx] = (
    ξx * (ηxᵢ₊½ - ηxᵢ₋½) +
    ξy * (ηyᵢ₊½ - ηyᵢ₋½) +
    ξz * (ηzᵢ₊½ - ηzᵢ₋½) +
    #
    ηx * (ηxⱼ₊½ - ηxⱼ₋½) +
    ηy * (ηyⱼ₊½ - ηyⱼ₋½) +
    ηz * (ηzⱼ₊½ - ηzⱼ₋½) +
    #
    ζx * (ηxₖ₊½ - ηxₖ₋½) +
    ζy * (ηyₖ₊½ - ηyₖ₋½) +
    ζz * (ηzₖ₊½ - ηzₖ₋½)
  )

  γᵢⱼₖ[idx] = (
    ξx * (ζxᵢ₊½ - ζxᵢ₋½) +
    ξy * (ζyᵢ₊½ - ζyᵢ₋½) +
    ξz * (ζzᵢ₊½ - ζzᵢ₋½) +
    #
    ηx * (ζxⱼ₊½ - ζxⱼ₋½) +
    ηy * (ζyⱼ₊½ - ζyⱼ₋½) +
    ηz * (ζzⱼ₊½ - ζzⱼ₋½) +
    #
    ζx * (ζxₖ₊½ - ζxₖ₋½) +
    ζy * (ζyₖ₊½ - ζyₖ₋½) +
    ζz * (ζzₖ₊½ - ζzₖ₋½)
  )
end
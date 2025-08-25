function L2_norm(A, ::GPU)
  # _norm = sqrt(mapreduce(x -> (x^2), +, A) / length(A))
  # return _norm
  # return norm(A) / sqrt(length(A))
  return norm(A, Inf) / sqrt(length(A))
end

function L2_norm(A, ::CPU)
  _L2_norm(A, Val(nthreads()))
end

function _L2_norm(a, ::Val{nchunks}) where {nchunks}
  _numer = @MVector zeros(nchunks)

  @batch for idx in eachindex(a)
    ichunk = threadid()

    _numer[ichunk] += a[idx]^2
  end

  return sqrt(sum(_numer) / length(a))
end

@kernel inbounds = true function _update_resid_kernel!(
  residuals,
  cache,
  cell_center_metrics,
  @Const(u),
  @Const(u_prev),
  @Const(flux),
  @Const(source_term),
  @Const(dt),
  @Const(ϵ),
  @Const(I0),
)
  idx = @index(Global, Cartesian)
  idx += I0

  @inline ∇q = flux_divergence(flux, cache, cell_center_metrics, idx)

  uⁿ = u[idx]
  uⁿ⁻¹ = u_prev[idx]
  du = uⁿ - uⁿ⁻¹
  du = du * !isapprox(uⁿ, uⁿ⁻¹; rtol=ϵ)

  residuals[idx] = -du / dt - ∇q + source_term[idx]
end

@kernel inbounds = true function _update_resid_kernel_tz!(
  residuals,
  cache,
  α,
  cell_center_metrics,
  coords,
  @Const(nhalo),
  @Const(nnodes),
  @Const(u),
  @Const(u_prev),
  @Const(flux),
  @Const(source_term),
  @Const(dt),
  @Const(ρ),
  @Const(cₚ),
  @Const(ϵ),
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

  uⁿ = u[idx]
  uⁿ⁻¹ = u_prev[idx]
  du = uⁿ - uⁿ⁻¹
  du = du * !isapprox(uⁿ, uⁿ⁻¹; rtol=ϵ)

  residuals[idx] = -du / dt - ∇q + source_term[idx] + f
end

"""
    update_residual!(solver::PseudoTransientSolver, mesh, Δt)

"""
function update_residual!(
  solver::PseudoTransientSolver{N,T}, mesh, Δt, ϵ=eps(T); ρ,
  cₚ,
  t=0,
  tz=false,
  tz_dim=1,
  no_t=true,
) where {N,T}
  #
  domain = solver.iterators.domain.cartesian
  idx_offset = first(domain) - oneunit(first(domain))

  if tz == false
    _update_resid_kernel!(solver.backend)(
      solver.res,
      solver.cache,
      mesh.cell_center_metrics,
      solver.u,
      solver.u_prev,
      solver.q′,
      solver.source_term,
      Δt,
      ϵ,
      idx_offset;
      ndrange=size(domain),
    )
  else
    tz_type = :wavy
    _update_resid_kernel_tz!(solver.backend)(
      solver.res,
      solver.cache,
      solver.α,
      mesh.cell_center_metrics,
      mesh.centroid_coordinates,
      mesh.nhalo,
      mesh.nnodes,
      solver.u,
      solver.u_prev,
      solver.q′,
      solver.source_term,
      Δt,
      ρ,
      cₚ,
      ϵ,
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

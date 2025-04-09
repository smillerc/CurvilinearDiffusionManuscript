
include("stencils.jl")
include("inner_operators.jl")
# include("boundary_operators.jl")

function assemble!(
  A::SparseMatrixCSC, u::AbstractArray{T,2}, scheme::ImplicitScheme{2}, mesh, Δt
) where {T}
  backend = scheme.backend
  workgroup = (64,)

  nrows = length(scheme.iterators.full.cartesian)
  _assembly_kernel_2d(backend, workgroup)(
    A,
    scheme.linear_problem.b,
    scheme.source_term,
    u,
    scheme.α,
    Δt,
    mesh.cell_center_metrics.J,
    mesh.edge_metrics,
    scheme.iterators.mesh,
    scheme.iterators.full.cartesian,
    scheme.iterators.full.linear,
    scheme.limits,
    scheme.mean_func,
    scheme.bcs;
    ndrange=nrows,
  )

  # # KernelAbstractions.synchronize(backend)

  return nothing
end

function assemble!(
  A::HYPREMatrix, u::AbstractArray{T,2}, scheme::ImplicitScheme{2}, mesh, Δt
) where {T}


  b = scheme.linear_problem.b
  source_term = scheme.source_term
  α = scheme.α
  cell_center_jacobian = mesh.cell_center_metrics.J
  edge_metrics = mesh.edge_metrics
  mesh_indices = scheme.iterators.mesh
  diffusion_prob_indices = scheme.iterators.full.cartesian
  matrix_indices = scheme.iterators.full.linear
  limits = scheme.limits
  mean_func = scheme.mean_func
  bcs = scheme.bcs

  #
  # Assemble A matrix
  #
  A_assembler = HYPRE.start_assemble!(A)

  @unpack ilo, ihi, jlo, jhi = scheme.limits


  for idx in LinearIndices(scheme.iterators.full.cartesian)
    row = matrix_indices[idx]
    mesh_idx = mesh_indices[idx]
    diff_idx = diffusion_prob_indices[idx]
    i, j = diff_idx.I

    onbc = i == ilo || i == ihi || j == jlo || j == jhi

    if !onbc
      J = cell_center_jacobian[mesh_idx]
      edge_α = edge_diffusivity(α, diff_idx, mean_func)
      A_coeffs = _inner_diffusion_operator(edge_α, Δt, J, edge_metrics, mesh_idx)

      # @inbounds for icol in 1:9
      # icol = 5
      # ij = diff_idx.I .+ offsets2d[icol]
      # col = matrix_indices[ij...]

      ᵢ₋₁ = ᵢ = ᵢ₊₁ = row

      ⱼ₋₁ = matrix_indices[(diff_idx.I .+ offsets2d[2])...]
      ⱼ = matrix_indices[(diff_idx.I .+ offsets2d[5])...]
      ⱼ₊₁ = matrix_indices[(diff_idx.I .+ offsets2d[8])...]
      # @show ij, row, col
      a = reshape(A_coeffs, 3, 3) |> collect

      HYPRE.assemble!(
        A_assembler,
        [ᵢ₋₁, ᵢ, ᵢ₊₁],
        [ⱼ₋₁, ⱼ, ⱼ₊₁],
        a)

      # @show [ᵢ₋₁, ᵢ, ᵢ₊₁]
      # @show [ⱼ₋₁, ⱼ, ⱼ₊₁]

      # error("done")

    end
  end

  HYPRE.finish_assemble!(A_assembler)

  #
  # Assemble RHS b vector
  #
  b_assembler = HYPRE.start_assemble!(b)

  for idx in LinearIndices(scheme.iterators.full.cartesian)
    row = matrix_indices[idx]
    mesh_idx = mesh_indices[idx]
    diff_idx = diffusion_prob_indices[idx]
    i, j = diff_idx.I

    onbc = i == ilo || i == ihi || j == jlo || j == jhi

    if !onbc
      J = cell_center_jacobian[mesh_idx]
      rhs_coeff = (source_term[diff_idx] * Δt + u[mesh_idx]) * J
    else
      rhs_coeff = bc_operator(bcs, diff_idx, limits, T)
    end

    # b[row] = rhs_coeff
    HYPRE.assemble!(b_assembler, [row], [rhs_coeff])
  end

  HYPRE.finish_assemble!(b_assembler)

  return nothing
end



function assemble!(
  A::SparseMatrixCSC, u::AbstractArray{T,3}, scheme::ImplicitScheme{3}, mesh, Δt
) where {T}
  backend = scheme.backend
  workgroup = (64,)

  nrows = length(scheme.iterators.full.cartesian)
  _assembly_kernel_3d(backend, workgroup)(
    A,
    scheme.linear_problem.b,
    scheme.source_term,
    u,
    scheme.α,
    Δt,
    mesh.cell_center_metrics.J,
    mesh.edge_metrics,
    scheme.iterators.mesh,
    scheme.iterators.full.cartesian,
    scheme.iterators.full.linear,
    scheme.limits,
    scheme.mean_func,
    scheme.bcs;
    ndrange=nrows,
  )

  # # KernelAbstractions.synchronize(backend)

  return nothing
end

module PseudoTransientScheme

using LinearAlgebra: norm
using TimerOutputs: @timeit
using CartesianDomains: expand, shift, expand_lower, haloedge_regions
using CurvilinearGrids:
  CurvilinearGrid1D, CurvilinearGrid2D, CurvilinearGrid3D, cellsize_withhalo, coords
using KernelAbstractions
using KernelAbstractions: CPU, GPU, @kernel, @index
using Polyester: @batch
using UnPack: @unpack
using Printf
using StaticArrays
using WriteVTK
using .Threads

using ..TimeStepControl

using ..BoundaryConditions

include("../averaging.jl")
include("../validity_checks.jl")
include("../edge_terms.jl")

export PseudoTransientSolver

struct PseudoTransientSolver{N,T,BE,AA<:AbstractArray{T,N},CA,NT1,DM,B,F}
  u::AA
  u_prev::AA
  source_term::AA
  q::NT1
  q′::NT1
  res::AA
  cache::CA
  α::AA # diffusivity
  θr_dτ::AA
  dτ_ρ::AA
  spacing::NTuple{N,T}
  L::T
  iterators::DM
  bcs::B # boundary conditions
  mean::F
  backend::BE
  last_err_check::Vector{Int}
end

function PseudoTransientSolver(
  mesh, bcs; backend=CPU(), mean=:harmonic, T=Float64, kwargs...
)
  #
  #         u
  #     |--------|--------|--------|--------|
  #           q_i+1/2
  #

  iterators = (
    domain=(cartesian=mesh.iterators.cell.domain,),
    full=(cartesian=mesh.iterators.cell.full,),
  )

  u = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  u_prev = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  S = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)) # source term
  residual = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  α = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  θr_dτ = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))
  dτ_ρ = KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full))

  # edge-based; same dims as the cell-based arrays, but each entry stores the {i,j,k}+1/2 value
  q = flux_tuple(mesh, backend, T)
  q′ = flux_tuple(mesh, backend, T)

  L, spacing = phys_dims(mesh, T)

  if mean === :harmonic
    mean_func = harmonic_mean # from ../averaging.jl
  else
    mean_func = arithmetic_mean # from ../averaging.jl
  end

  metric_cache = get_metric_cache(mesh, backend, T)
  last_err_check = [0]
  return PseudoTransientSolver(
    u,
    u_prev,
    S,
    q,
    q′,
    residual,
    metric_cache,
    α,
    θr_dτ,
    dτ_ρ,
    spacing,
    L,
    iterators,
    bcs,
    mean_func,
    backend,
    last_err_check,
  )
end

get_metric_cache(mesh, backend, T) = nothing

function get_metric_cache(mesh::CurvilinearGrid2D, backend, T)
  return (;
    α=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    β=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
  )
end

function get_metric_cache(mesh::CurvilinearGrid3D, backend, T)
  return (;
    α=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    β=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    γ=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
  )
end

function phys_dims(mesh::CurvilinearGrid2D, T)
  x, y = coords(mesh)
  # spacing = (minimum(diff(x; dims=1)), minimum(diff(y; dims=2))) .|> T

  @views begin
    ds = sqrt(minimum(mesh.cell_center_metrics.J[mesh.iterators.cell.domain]))
    spacing = (ds, ds)
  end

  min_x, max_x = extrema(x)
  min_y, max_y = extrema(y)
  # L = max(abs(max_x - min_x), abs(max_y - min_y)) |> T
  L = min(abs(max_x - min_x), abs(max_y - min_y)) |> T
  # @show L
  # L = maximum(spacing)
  # error("checkme!")
  return L, spacing
end

function phys_dims(mesh::CurvilinearGrid3D, T)
  x, y, z = coords(mesh)
  spacing =
    (minimum(diff(x; dims=1)), minimum(diff(y; dims=2)), minimum(diff(z; dims=3))) .|> T

  min_x, max_x = extrema(x)
  min_y, max_y = extrema(y)
  min_z, max_z = extrema(z)

  L = min(abs(max_x - min_x), abs(max_y - min_y), abs(max_z - min_z)) |> T
  # L = maximum(spacing)
  return L, spacing
end

function flux_tuple(mesh::CurvilinearGrid2D, backend, T)
  return (
    x=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    y=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
  )
end

function flux_tuple(mesh::CurvilinearGrid3D, backend, T)
  return (
    x=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    y=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
    z=KernelAbstractions.zeros(backend, T, size(mesh.iterators.cell.full)),
  )
end

include("conductivity.jl")
include("flux_divergence.jl")
include("flux/fluxes.jl")
# include("flux.jl")
include("iteration_parameters.jl")
include("residuals.jl")
include("update.jl")
include("mesh_metric_cache.jl")

# solve a single time-step dt
function step!(
  solver::PseudoTransientSolver{N,DT},
  mesh,
  T,
  ρ,
  cₚ,
  κ,
  dt;
  max_iter=1500,
  rtol=1e-5,
  atol=sqrt(eps(DT)),
  error_check_interval=2,
  apply_cutoff=false,
  calculate_next_dt=true,
  subcycle_conductivity=true,
  write_diagnostic_vtk=false,
  enforce_positivity=false,
  CFL=1 / sqrt(3),
  kwargs...,
) where {N,DT}
  ##
  rt = @elapsed begin
    last_iter_count = solver.last_err_check[1]
    min_err_check = Int(round(last_iter_count * 0.9))
    err_interval = max(1, Int(ceil(last_iter_count - min_err_check) ÷ 2))

    ##

    #
    domain = solver.iterators.domain.cartesian
    nhalo = 1
    # nhalo = mesh.nhalo

    norm_iter = 0
    iter = 0
    rel_error = 2 * rtol
    abs_error = 2 * atol
    init_L₂ = Inf

    Vpdτ = CFL * min(solver.spacing...)

    @assert dt > 0
    @assert all(solver.spacing .> 0)
    @assert Vpdτ > 0

    copy!(solver.u, T)
    copy!(solver.u_prev, T)

    @timeit "update_metric_cache" update_metric_cache!(solver, mesh)

    # @timeit "applybcs! (u)" applybcs!(solver.bcs, mesh, solver.u, nhalo)

    # @timeit "update_conductivity!" update_conductivity!(solver, mesh, solver.u, ρ, cₚ, κ)

    @timeit "validate_scalar (α)" validate_scalar(
      solver.α, domain, nhalo, :diffusivity; enforce_positivity=enforce_positivity
    )

    # @timeit "applybcs! (α)" applybcs!(solver.bcs, mesh, solver.α, nhalo)

    # @timeit "validate_scalar (u)" validate_scalar(
    #   solver.u, domain, nhalo, :u; enforce_positivity=enforce_positivity
    # )

    @timeit "validate_scalar (source_term)" validate_scalar(
      solver.source_term, domain, nhalo, :source_term; enforce_positivity=false
    )

    # Pseudo-transient iteration
    while true
      @timeit "applybcs! (u)" applybcs!(solver.bcs, mesh, solver.u, nhalo)
      # Diffusion coefficient
      if subcycle_conductivity || iter == 0
        @timeit "update_conductivity!" update_conductivity!(
          solver, mesh, solver.u, ρ, cₚ, κ
        )
        @timeit "update_iteration_params!" update_iteration_params!(solver, ρ, Vpdτ, dt;)
      end

      iter += 1

      # @timeit "update_iteration_params!" update_iteration_params!(solver, ρ, Vpdτ, dt;)

      # @timeit "validate_scalar (θr_dτ)" validate_scalar(
      #   solver.θr_dτ, domain, nhalo, :θr_dτ; enforce_positivity=false
      # )

      # @timeit "validate_scalar (dτ_ρ)" validate_scalar(
      #   solver.dτ_ρ, domain, nhalo, :dτ_ρ; enforce_positivity=false
      # )

      @timeit "compute_flux!" compute_flux!(solver, mesh)
      @timeit "compute_update!" compute_update!(solver, mesh, dt)

      # Apply a cutoff function to remove negative / non-finite values

      if (iter >= min_err_check && iter % err_interval == 0 || iter == 1)
        @timeit "update_residual!" update_residual!(solver, mesh, dt)
        # validate_scalar(solver.res, domain, nhalo, :resid; enforce_positivity=false)

        norm_iter += 1

        @timeit "norm" begin
          L₂ = L2_norm(solver.res, solver.backend)

          if iter == 1
            init_L₂ = L₂
          end

          rel_error = L₂ / init_L₂
          abs_error = L₂
        end
      end

      if write_diagnostic_vtk
        to_vtk(solver, mesh, solver.u, ρ, iter, iter)
      end

      if !isfinite(rel_error) || !isfinite(abs_error)
        to_vtk(solver, mesh, solver.u, ρ, iter, iter)
        error(
          "Non-finite error detected! abs_error = $abs_error, rel_error = $rel_error, exiting...",
        )
      end

      if iter > max_iter
        to_vtk(solver, mesh, solver.u, ρ, iter, iter)
        error(
          "Maximum iteration limit reached ($max_iter), abs_error = $abs_error, rel_error = $rel_error, exiting...",
        )
      end

      if (rel_error <= rtol || abs_error <= atol)
        break
      end
    end

    solver.last_err_check[1] = iter

    if enforce_positivity
      @timeit "enforce_positivity" begin
        solver.u .= abs.(solver.u)
      end
    end

    if apply_cutoff
      @timeit "cutoff!" cutoff!(solver.u, solver.backend)
    end

    # @timeit "validate_scalar (u)" validate_scalar(
    #   solver.u, domain, nhalo, :u; enforce_positivity=enforce_positivity
    # )'
    # @timeit "norrmmm" norm(solver.res)

    if calculate_next_dt
      @timeit "next_dt" begin
        next_Δt = next_dt(solver.u, solver.u_prev, dt; kwargs...)
      end
    else
      next_Δt = dt
    end

    copy!(T, solver.u)
  end

  stats = (rel_err=rel_error, abs_err=abs_error, niter=iter, time=rt)
  return stats, next_Δt
end

# ------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------

function cutoff!(A, ::CPU)
  @batch for idx in eachindex(A)
    a = A[idx]
    A[idx] = (0.5(abs(a) + a)) * isfinite(a)
  end

  return nothing
end

function cutoff!(A, ::GPU)
  @. A = (0.5(abs(A) + A)) * isfinite(A)
end

get_filename(iteration) = file_prefix * @sprintf("%07i", iteration)

function get_filename(iteration, name::String)
  name = replace(name, " " => "_")
  if !endswith(name, "_")
    name = name * "_"
  end
  return name * @sprintf("%07i", iteration)
end

function to_vtk(scheme, mesh, u, ρ, iteration=0, t=0.0, name="diffusion", T=Float32)
  fn = get_filename(iteration, name)
  @info "Writing to $fn"

  domain = mesh.iterators.cell.domain

  _coords = Array{T}.(coords(mesh))

  @views vtk_grid(fn, _coords...) do vtk
    vtk["TimeValue"] = t

    vtk["rho"] = Array{T}(ρ[domain])

    vtk["u"] = Array{T}(u[domain])
    vtk["u_prev"] = Array{T}(scheme.u_prev[domain])
    vtk["residual"] = Array{T}(scheme.res[domain])

    for (i, qi) in enumerate(scheme.q)
      vtk["q$i"] = Array{T}(qi[domain])
    end

    for (i, qi) in enumerate(scheme.q′)
      vtk["q2$i"] = Array{T}(qi[domain])
    end

    vtk["diffusivity"] = Array{T}(scheme.α[domain])

    vtk["dτ_ρ"] = Array{T}(scheme.dτ_ρ[domain])
    vtk["θr_dτ"] = Array{T}(scheme.θr_dτ[domain])

    vtk["source_term"] = Array{T}(scheme.source_term[domain])
  end
end

end # module

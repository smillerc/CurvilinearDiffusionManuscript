using CurvilinearGrids, CurvilinearDiffusion
using Printf, Adapt
using TimerOutputs
using KernelAbstractions
using Glob
using LinearAlgebra
using Plots

# @static if Sys.islinux()
#   using MKL
# elseif Sys.isapple()
#   using AppleAccelerate
# end

# NMAX = Sys.CPU_THREADS
# BLAS.set_num_threads(NMAX)
# BLAS.get_num_threads()

# @show BLAS.get_config()

dev = :CPU
const DT = Float64

if dev === :GPU
    @info "Using CUDA"
    using CUDA
    using CUDA.CUDAKernels
    backend = CUDABackend()
    ArrayT = CuArray
    # CUDA.allowscalar(false)
else
    backend = CPU()
    ArrayT = Array
end

# ------------------------------------------------------------
# Grid Construction
# ------------------------------------------------------------
function wavy_grid(ni, nj)
    Lx = 12
    Ly = 12
    n_xy = 6
    n_yx = 6

    xmin = -Lx / 2
    ymin = -Ly / 2

    Δx0 = Lx / (ni)
    Δy0 = Ly / (nj)

    # Ax = 0.4 / Δx0
    # Ay = 0.8 / Δy0
    Ax = 0.2 / Δx0
    Ay = 0.4 / Δy0

    x = zeros(ni + 1, nj + 1)
    y = zeros(ni + 1, nj + 1)
    for j in 1:(nj+1)
        for i in 1:(ni+1)
            x[i, j] = xmin + Δx0 * ((i - 1) + Ax * sinpi((n_xy * (j - 1) * Δy0) / Ly))
            y[i, j] = ymin + Δy0 * ((j - 1) + Ay * sinpi((n_yx * (i - 1) * Δx0) / Lx))
        end
    end

    return CurvilinearGrid2D(x, y, :meg6)
end

function uniform_grid(nx, ny)
    x0, x1 = (-6, 6)
    y0, y1 = (-6, 6)

    return CurvilinearGrids.RectilinearGrid2D((x0, y0), (x1, y1), (nx, ny), :meg6)
end

function axisymmetric_grid(nr, nz)
    rotational_axis = :y # rotate about the pole axis

    r0, r1 = (0, 6)
    z0, z1 = (0, 6)

    r = LinRange(r0, r1, nr)
    z = LinRange(z0, z1, nz)

    x = zeros(nr, nz)
    y = zeros(nr, nz)

    for i in 1:nr
        for j in 1:nz
            if rotational_axis == :y
                x[i, j] = r[i]
                y[i, j] = z[j]
            else
                x[i, j] = z[i]
                y[i, j] = r[j]
            end
        end
    end

    snap_to_axis = true
    return AxisymmetricGrid2D(x, y, :meg6, snap_to_axis, rotational_axis)
end

function initialize_mesh(kres, k, DT)
    ni = nj = kres^k
    # ni = nj = 101
    nhalo = 1
    return wavy_grid(ni, nj)
    # return axisymmetric_grid(ni, nj)
    # return uniform_grid(ni, nj)
end

function init_state_no_source(scheme, kres, k, kwargs...)

    # Define the conductivity model
    @inline function κ(ρ, temperature)
        if !isfinite(temperature)
            return zero(ρ)
        else
            return 2.5
        end
    end

    mesh = adapt(ArrayT, initialize_mesh(kres, k, DT))

    bcs = (ilo=NeumannBC(), ihi=NeumannBC(), jlo=NeumannBC(), jhi=NeumannBC())

    if scheme === :implicit
        solver = ImplicitScheme(mesh, bcs; backend=backend, kwargs...)
    elseif scheme === :pseudo_transient
        solver = PseudoTransientSolver(mesh, bcs; backend=backend, T=DT, kwargs...)
    else
        error("Must choose either :implict or :pseudo_transient")
    end

    # Temperature and density
    T_cold = 1e-2
    T = ones(DT, cellsize_withhalo(mesh)) * T_cold
    ρ = ones(DT, cellsize_withhalo(mesh))
    cₚ = 1.0

    fwhm = 1.0
    x0 = 0.0
    y0 = 0.0
    xc = Array(mesh.centroid_coordinates.x)
    yc = Array(mesh.centroid_coordinates.y)
    for idx in mesh.iterators.cell.domain
        T[idx] = exp(-(((x0 - xc[idx])^2) / fwhm + ((y0 - yc[idx])^2) / fwhm)) #+ T_cold
    end

    # copy!(solver.u, T)
    return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

function init_state_with_source(scheme, kres, k, kwargs...)

    # Define the conductivity model
    @inline function κ(ρ, temperature)
        if !isfinite(temperature)
            return zero(ρ)
        else
            return 2.5 * abs(temperature)^3
        end
    end

    mesh = initialize_mesh(kres, k, DT)
    bcs = (ilo=NeumannBC(), ihi=NeumannBC(), jlo=NeumannBC(), jhi=NeumannBC())

    if scheme === :implicit
        solver = ImplicitScheme(mesh, bcs; backend=backend, kwargs...)
    elseif scheme === :pseudo_transient
        solver = PseudoTransientSolver(mesh, bcs; backend=backend, T=DT, kwargs...)
    else
        error("Must choose either :implict or :pseudo_transient")
    end

    # Temperature and density
    T_hot = 1e4 |> DT
    T_cold = 1e-2 |> DT
    T = ones(DT, cellsize_withhalo(mesh)) * T_cold
    ρ = ones(DT, cellsize_withhalo(mesh))
    source_term = zeros(DT, size(solver.source_term))
    cₚ = 1.0 |> DT

    fwhm = 1.0 |> DT
    x0 = 0.0 |> DT
    y0 = 0.0 |> DT
    xc = Array(mesh.centroid_coordinates.x)
    yc = Array(mesh.centroid_coordinates.y)
    for (idx, idx2) in zip(mesh.iterators.cell.domain, solver.iterators.domain.cartesian)
        source_term[idx2] =
            T_hot * exp(-(((x0 - xc[idx])^2) / fwhm + ((y0 - yc[idx])^2) / fwhm)) + T_cold
    end

    @show size(solver.source_term), cellsize_withhalo(mesh)
    copy!(solver.source_term, source_term)
    return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function solve_prob(scheme, kres, k, case=:no_source; maxiter=Inf, maxt=0.2, kwargs...)
    casename = "blob"

    if case === :no_source
        scheme, mesh, T, ρ, cₚ, κ = init_state_no_source(scheme, kres, k, kwargs...)
    else
        scheme, mesh, T, ρ, cₚ, κ = init_state_with_source(scheme, kres, k, kwargs...)
    end

    T_init = copy(T)
    global Δt = 1e-6
    global t = 0.0
    global iter = 0
    global io_interval = 0.01
    global io_next = io_interval
    @timeit "update_conductivity!" update_conductivity!(scheme, mesh, T, ρ, cₚ, κ)
    @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, ρ, mesh, iter, t, casename)

    while true
        @printf "cycle: %i t: %.4e, Δt: %.3e\n" iter t Δt
        @timeit "nonlinear_thermal_conduction_step!" begin
            stats, next_dt = nonlinear_thermal_conduction_step!(
                scheme,
                mesh,
                T,
                ρ,
                cₚ,
                κ,
                DT(Δt);
                apply_cutoff=true,
                show_convergence=true,
                precon_iter_threshold=1,
                kwargs...,
            )
        end

        # if t + Δt > io_next
        #   @timeit "save_vtk" CurvilinearDiffusion.save_vtk(
        #     scheme, T, ρ, mesh, iter, t, casename
        #   )
        #   global io_next += io_interval
        # end

        if iter == 0
            reset_timer!()
        end

        if t >= maxt
            break
        end

        global iter += 1
        global t += Δt
        if iter >= maxiter - 1
            break
        end
        Δt = min(next_dt, 1e-4)
    end

    @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, ρ, mesh, iter, t, casename)

    print_timer()
    return scheme, mesh, T, T_init
end

function plot_sol(scheme, mesh, temperature, T_initial)
    xc, yc = centroids(mesh)
    mid = Int(ceil(length(xc[:, 1]) / 2))

    domain = mesh.iterators.cell.domain
    ddomain = scheme.iterators.domain.cartesian
    sim = @view temperature[domain]
    Tinit = @view T_initial[domain]

    # tfinal = 4.9683e-02
    # tfinal = 0.0505
    #sol = u_analytic.(xc, t)
    x = xc[:, mid]
    #Tinit = T_initial
    #println("length of Tinit is ")
    @assert length(Tinit) == length(sim)
    @assert length(xc[:, mid]) == length(Tinit[:, mid])

    #L₂ = sqrt(mapreduce((x, y) -> abs(x^2 - y^2), +, sol, sim) / length(sim))
    #@show L₂

    #p1 = plot(xc[:, mid], Tinit[:, mid]; label="initial", xlabel="x", ylabel="u")
    p = plot(
        xc[:, mid],
        sim[:, mid];
        label="simulated",
        lw=5,
        marker=:dot,
        xlabel="x",
        ylabel="u",
        title="u(x,0,tf) with nx=3^6, T_hot = 1e2, tf = 2.5e-2",
    )
    plot!(p, xc[:, mid], Tinit[:, mid]; label="initial", xlabel="x", ylabel="u")
    #p3 = scatter!(p2, xc[:, 1], sol[:, 1]; ms=0.5, label="analytic")
    #p = plot(p1, p2; layout=(2, 1), show=true)

    s = surface(xc, yc, sim')

    display(p)

    #savefig("steadyState_log.png")
end

# @profview 
begin
    cd(@__DIR__)
    rm.(glob("*.vts"))

    kstart = 3
    kend = 5

    kres = 3
    kplot = kend

    simp = zeros(kres^kplot)
    initp = zeros(kres^kplot)
    x = zeros(kres^kplot)
    x1 = zeros(kres^kstart, kres^kstart)
    x2 = zeros(kres^(kstart + 1), kres^(kstart + 1))
    x3 = zeros(kres^(kstart + 2), kres^(kstart + 2))
    cpoints = zeros(kres^kstart)
    csim = zeros(kres^kstart)
    cpx = zeros(kres^kstart, kres^kstart)
    cpy = zeros(kres^kstart, kres^kstart)
    cps = zeros(kres^kstart, kres^kstart)

    x2D = zeros(kres^kplot, kres^kplot)
    y2D = zeros(kres^kplot, kres^kplot)
    sim2D = zeros(kres^kplot, kres^kplot)
    #mid = Int(floor((kres^kend) / 2))

    #initial constants and arrays for convergence study
    num_runs = (kend - kstart) + 1
    st_indx = zeros(Int64, num_runs)
    st_indy = zeros(Int64, num_runs)

    #sinds = Dict(3 => 2, 4 => 5, 5 => 14, 6 => 41, 7 => 122)
    sinds = Dict(3 => 2, 4 => 5, 5 => 14, 6 => 41, 7 => 122)

    x_grids = Dict()
    y_grids = Dict()

    smallg_sol = Dict()

    for i in 1:num_runs
        if i == 1
            st_indx[i] = 1 #sinds[kstart]
            st_indy[i] = sinds[kstart+1]
        else
            st_indx[i] = st_indx[i-1] * kres - 1
            st_indy[i] = st_indy[i-1] * kres - 1
        end
    end

    # scheme, mesh, temperature = solve_prob(:pseudo_transient, :no_source, 500)
    # scheme, mesh, temperature = solve_prob(
    #   :implicit, :with_source; maxiter=100, direct_solve=false, direct_solver=:pardiso
    # )
    # scheme, mesh, temperature = solve_prob(:implicit, :no_source, 10; direct_solve=true)

    for k in kstart:kend
        ind = k - kstart + 1

        #=
        #No source
        scheme, mesh, temperature, T_init = solve_prob(
          :pseudo_transient,
          kres,
          k,
          #:implicit,
          :no_source;
          maxiter=Inf,
          maxt=0.4,
          direct_solve=false,
          mean=:harmonic,
          error_check_interval=2,
          CFL=0.4,
          refresh_matrix=false,
        )
        =#

        # With source
        scheme, mesh, temperature, T_init = solve_prob(
            :pseudo_transient,
            kres,
            k,
            # :implicit,
            :with_source;
            maxiter=Inf,
            # maxt=1.5e-3,
            # maxt=2e-3,
            maxt=1e-3,
            direct_solve=false,
            mean=:arithmetic,
            apply_cutoff=true,
            enforce_positivity=true,
            error_check_interval=10,
            CFL=0.4, # working
            # CFL=0.5,
            subcycle_conductivity=true,
        )

        xc, yc = centroids(mesh)

        domain = mesh.iterators.cell.domain
        sim = @view temperature[domain]

        #~ Store grids and solution at coinciding grid points for Richardson study
        if k == kstart
            x_grids[ind] = xc[:, st_indy[ind]:(kres^((kstart-2)+ind)):end]
            y_grids[ind] = yc[st_indy[ind]:(kres^((kstart-2)+ind)):end, :]
            smallg_sol[ind] = sim[
                st_indx[ind]:(kres^(ind-1)):end, st_indx[ind]:(kres^(ind-1)):end
            ]
            x1[:, :] = xc[:, :]
        else
            x_grids[ind] = xc[
                st_indx[ind]:(kres^(ind-1)):end, st_indy[ind]:(kres^((kstart-2)+ind)):end
            ]
            y_grids[ind] = yc[
                st_indy[ind]:(kres^((kstart-2)+ind)):end, st_indx[ind]:(kres^(ind-1)):end
            ]
            @assert isapprox(x_grids[ind], x_grids[ind-1])
            @assert isapprox(y_grids[ind], y_grids[ind-1])
            smallg_sol[ind] = sim[
                st_indx[ind]:(kres^(ind-1)):end, st_indx[ind]:(kres^(ind-1)):end
            ]
            if k == kstart + 1
                x2[:, :] = xc[:, :]
            else
                x3[:, :] = xc[:, :]
            end
        end

        plot_sol(scheme, mesh, temperature, T_init)

        if k == kplot
            mid = st_indy[ind] + (kres^((kstart - 2) + ind))
            N = kres^kplot
            cinds = collect(st_indx[ind]:(kres^(ind-1)):N)
            @show cinds
            cpoints[:] = [xc[i, mid] for i in cinds]
            csim[:] = [sim[i, mid] for i in cinds]

            cpx[:, :] = [xc[i, j] for i in cinds, j in cinds]
            cpy[:, :] = [yc[i, j] for i in cinds, j in cinds]
            cps[:, :] = [sim[i, j] for i in cinds, j in cinds]

            simp[:] = sim[:, mid]
            T_init_ng = @view T_init[domain]
            initp[:] = T_init_ng[:, mid]
            x[:] = xc[:, mid]
            x2D[:, :] = xc[:, :]
            y2D[:, :] = yc[:, :]
            sim2D[:, :] = sim[:, :]
        end
        println("Finished run for gres = $(kres^k)")
    end

    #~ compute richardson extrapolation
    @assert isapprox(x_grids[1], x_grids[2])
    @assert isapprox(x_grids[1], x_grids[3])
    @assert isapprox(y_grids[1], y_grids[2])
    @assert isapprox(y_grids[1], y_grids[3])

    R_conv = zeros(num_runs - 2)
    for i in 1:(num_runs-2)
        if num_runs >= 3
            num_err = norm(smallg_sol[i] - smallg_sol[i+1], Inf)
            denom_err = norm(smallg_sol[i+1] - smallg_sol[i+2], Inf)

            R_conv[i] = log(kres, num_err / denom_err)
        else
            @error(
                "Must have at least 3 grid refinements in order to approximate convergence using Ricahrdson extrapolation"
            )
        end
    end

    @show R_conv

    p = plot(
        x,
        simp;
        lw=5,
        marker=:dot,
        label="u(x,0,tf)",
        xlabel="x",
        ylabel="u",
        title="u(x,0,tf) with nx=3⁵, tf = 1e-3",
    )
    midy = maximum(simp) / 2
    #plot!(p, x, initp; label="u(x,0,0)", xlabel="x", ylabel="u", color=:green)
    scatter!(p, cpoints, csim; marker=:circle, color=:red, label="RE Points")
    annotate!(-4.3, midy, text("Conv. Rate = $(round(R_conv[1]; digits=2))", 11))

    #savefig(p, "Richardson_Tlarge_all_4_conv_plot.png")
    display(p)

    #
end

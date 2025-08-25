using Plots
using CurvilinearGrids, CurvilinearDiffusion
using Printf, Adapt
using TimerOutputs
using KernelAbstractions
using Glob
using LinearAlgebra

const π² = π * π
const q0 = 2
const DT = Float64

#set initial condition
function u_initial(x, y, t)
    # Choose TZ constant parameters
    a = 1 / 2
    c = 3.0
    d = 0

    x0 = 0
    y0 = 0
    fwhm = 1.0

    #return c * (x - d) .^ 2 + (x - d) .* sinpi.(a .* y)
    return exp.(-((x - x0) .^ 2 ./ fwhm + (y - y0) .^ 2 ./ fwhm))
end

function q(x, y, t)
    x0 = 1.0
    y0 = 1.0
    fwhm = 1.0
    T_hot = 1e4 |> DT

    return T_hot .* exp.(-((x - x0) .^ 2 ./ fwhm + (y - y0) .^ 2 ./ fwhm))
    # return 0
end

#define exact solution
function u_analytic(x, y, t)
    # Choose TZ constant parameters
    a = 1 / 2
    damp = 4
    c = 3.0
    d = 0

    x0 = 0
    y0 = 0
    fwhm = 1.0

    #return c * (x - d) .^ 2 + (x - d) .* sinpi.(a .* y) .* exp(-b * t)
    return exp.(-((x - x0) .^ 2 ./ fwhm + (y - y0) .^ 2 ./ fwhm)) .* exp(-damp * t)
end

dev = :CPU

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

    #xmin = -Lx / 2
    #ymin = -Ly / 2
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
    #println("x[2,1] is $(x[2,1])")

    return CurvilinearGrid2D(x, y, :meg6)
end

function uniform_grid(nx, ny, nhalo)
    x0, x1 = (0, 4)
    y0, y1 = (4, 8)

    return CurvilinearGrids.RectilinearGrid2D((x0, y0), (x1, y1), (nx, ny), :meg6)
end

function initialize_mesh(k, kres)
    ni = nj = kres^k
    nhalo = 1
    return wavy_grid(ni, nj)
    #return uniform_grid(ni, nj, nhalo)
end

# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------

# Define the conductivity model
#@inline κ(ρ, temperature) = 1.0

function init_state(k, kres)
    #@inline κ(ρ, temperature) = 1.0
    # Define the conductivity model

    @inline function κ(ρ, temperature)
        if !isfinite(temperature)
            return zero(ρ)
        else
            return (temperature)^(5 / 2)
        end
    end

    # Initialize mesh
    mesh = adapt(ArrayT, initialize_mesh(k, kres))
    xc = Array(mesh.centroid_coordinates.x)
    yc = Array(mesh.centroid_coordinates.y)

    # Define boundary conditions
    bcs = (
        #ilo=ExtrapolationBC(),
        #ihi=ExtrapolationBC(),  #
        #jlo=PeriodicBC(),  #
        #jhi=PeriodicBC(),  #
        #jlo=ExtrapolationBC(),  #
        #jhi=ExtrapolationBC(),  #
        ilo=NeumannBC(),
        ihi=NeumannBC(),
        jlo=NeumannBC(),
        jhi=NeumannBC(),
    )

    # Specify PT solver
    solver = PseudoTransientSolver(mesh, bcs; backend=backend, mean=:harmonic)

    # Temperature and density
    T = zeros(Float64, cellsize_withhalo(mesh))
    source_term = zeros(Float64, cellsize_withhalo(mesh))
    ρ = ones(Float64, cellsize_withhalo(mesh))
    cₚ = 1.0

    # Initialize temperature and source term
    n = size(xc)[1]
    for i in 1:n
        for j in 1:n
            T[i, j] = u_initial(xc[i, j], yc[i, j], 0.0)
            source_term[i, j] = q(xc[i, j], yc[i, j], 0.0) # / mesh.cell_center_metrics.J[idx]
            #source_term[i, j] = 1.0
        end
    end

    scheme_q = @view solver.source_term[solver.iterators.domain.cartesian]
    mesh_q = source_term[mesh.iterators.cell.domain]
    copy!(scheme_q, mesh_q)

    #   fill!(solver.source_term, 0.0)
    #fill!(solver.α, 1.0)

    return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

function init_state_with_source(k, kres)
    @inline κ(ρ, temperature) = 2.5 #(temperature) .^ 3
    #=
    # Define the conductivity model
    @inline function κ(ρ, temperature)
    if !isfinite(temperature)
      return zero(ρ)
    else
      return (temperature)^3
    end
    end
    =#

    # Initialize mesh
    mesh = adapt(ArrayT, initialize_mesh(k, kres))
    xc = Array(mesh.centroid_coordinates.x)
    yc = Array(mesh.centroid_coordinates.y)

    # Define boundary conditions
    bcs = (
        #ilo=ExtrapolationBC(),
        #ihi=ExtrapolationBC(),  #
        #jlo=PeriodicBC(),  #
        #jhi=PeriodicBC(),  #
        #jlo=ExtrapolationBC(),  #
        #jhi=ExtrapolationBC(),  #
        ilo=NeumannBC(),
        ihi=NeumannBC(),
        jlo=NeumannBC(),
        jhi=NeumannBC(),
    )

    # Specify PT solver
    solver = PseudoTransientSolver(mesh, bcs; backend=backend, mean=:harmonic)

    # Temperature and density
    T_hot = 1e4 |> DT
    #T_hot = 1.0
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

    n = size(xc)[1]
    for i in 1:n
        for j in 1:n
            T[i, j] = u_initial(xc[i, j], yc[i, j], 0.0)
            #source_term[i, j] = q(xc[i, j], yc[i, j], 0.0) # / mesh.cell_center_metrics.J[idx]
        end
    end

    for (idx, idx2) in zip(mesh.iterators.cell.domain, solver.iterators.domain.cartesian)
        source_term[idx2] =
            T_hot * exp(-(((x0 - xc[idx])^2) / fwhm + ((y0 - yc[idx])^2) / fwhm)) + T_cold
    end

    @show size(solver.source_term), cellsize_withhalo(mesh)
    copy!(solver.source_term, source_term)

    #   fill!(solver.source_term, 0.0)
    #fill!(solver.α, 1.0)

    return solver, mesh, adapt(ArrayT, T), adapt(ArrayT, ρ), cₚ, κ
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
function run(maxt, k, kres, maxiter=Inf)
    casename = "twilightzone_wavy2D"

    scheme, mesh, T, ρ, cₚ, κ = init_state(k, kres)

    global Δt = 1e-7
    global t = 0.0
    global iter = 0
    global io_interval = 0.01
    global io_next = io_interval
    @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, ρ, mesh, iter, t, casename)

    while true
        if iter == 0
            reset_timer!()
        end

        @printf "cycle: %i t: %.4e, Δt: %.3e\n" iter t Δt

        stats, next_dt = nonlinear_thermal_conduction_step!(
            scheme,
            mesh,
            T,
            ρ,
            cₚ,
            κ,
            Δt;
            t,
            tz=true,
            tz_dim=2,
            no_t=false,   #set true to test spatial convergence 
            #cutoff=false,
            rtol=5e-1,
            atol=1e-8,
            #enforce_positivity=false,
            subcycle_conductivity=true,
        )

        #=
        if t + Δt > io_next
          @timeit "save_vtk" CurvilinearDiffusion.save_vtk(
            scheme, T, ρ, mesh, iter, t, casename
          )
          global io_next += io_interval
        end
        =#

        if iter >= maxiter || t >= maxt
            break
        end

        global iter += 1
        global t += Δt

        if isfinite(next_dt)
            Δt = min(next_dt, 1e-4)
        end
    end

    @timeit "save_vtk" CurvilinearDiffusion.save_vtk(scheme, T, ρ, mesh, iter, t, casename)

    print_timer()

    return scheme, mesh, T
end

function plot_sol(k, kres, simf, solf, xc, yc)
    sim = simf[:, 12]     #grab a 1D cross section of simulation along x
    sol = solf[:, 12]     #grab a 1D cross sectino of the exact solution along x
    simy = simf[12, :]    #grab a 1D cross section of simulation along y
    soly = simf[12, :]    #grab a 1D cross section of the exact solution along y
    x = xc[:, 12]         #1D cell centered array along x 
    y = yc[12, :]         #1D cell centered array along y
    nx = length(x)
    ny = length(y)

    Tinit = zeros(nx, ny)
    for i in 1:nx
        for j in 1:ny
            Tinit[i, j] = u_initial(x[i], y[j], 0)
        end
    end

    L₂ = sqrt(mapreduce((x, y) -> abs(x^2 - y^2), +, sol, sim) / length(sim))
    @show L₂

    my_L₂ = norm(simf - solf, 2)
    @show my_L₂

    abs_err = norm(simf - solf, Inf)
    @show abs_err

    #rel_err = maximum(abs.((sim - sol)./sol))
    #@show rel_err

    p1 = plot(
        x, Tinit[:, 12]; label="initial", xlabel="x", ylabel="u", title="Uniform Grid Solution"
    )
    p2 = plot(x, sim; label="simulated", lw=5, xlabel="x", ylabel="u")
    p3 = scatter!(p2, x, sol; ms=0.5, label="analytic")
    px = plot(p1, p2; layout=(2, 1), show=true)

    p4 = plot(
        y, Tinit[12, :]; label="initial", xlabel="y", ylabel="u", title="Uniform Grid Solution"
    )
    p5 = plot(y, simy; label="simulated", lw=5, xlabel="y", ylabel="u")
    p6 = scatter!(p5, y, soly; ms=0.5, label="analytic")
    py = plot(p4, p5; layout=(2, 1), show=true)

    display(py)

    abs_err = norm(sim - sol, Inf)

    gres = kres^k

    savefig(px, "TZ_sol_x_$(gres)_tests.png")
    savefig(py, "TZ_sol_y_$(gres)_tests.png")
end

begin
    cd(@__DIR__)
    rm.(glob("*.vts"))
    #tfinal = 5e-2
    tfinal = 0.4  #final time

    #for nx = 2^k, chose start and end value for k (for convergence study)
    kstart = 7
    kend = 8

    kplot = kstart

    kres = 2
    mid = Int(floor((kres^kplot) / 2))

    #initial constants and arrays for convergence study
    num_runs = (kend - kstart) + 1
    abs_err = zeros(num_runs)
    my_L₂ = zeros(num_runs)
    rel_err = zeros(num_runs)
    conv_rate = zeros(num_runs - 1)
    st_indx = zeros(Int64, num_runs)
    st_indy = zeros(Int64, num_runs)

    #initialize arrays for plotting
    dx = zeros(num_runs)
    errs = zeros(kres^kplot)
    xs = zeros(kres^kstart, kres^kstart)
    x = zeros(kres^kplot)
    y = zeros(kres^kplot)
    xf = zeros(kres^kplot)
    yf = zeros(kres^kplot)
    x2D = zeros(kres^kplot, kres^kplot)
    y2D = zeros(kres^kplot, kres^kplot)
    simng = zeros(kres^kplot, kres^kplot)

    simf = zeros(kres^kplot + 10)
    simf2D = zeros(kres^kplot + 10, kres^kplot + 10)

    simp = zeros(kres^kplot)
    solp = zeros(kres^kplot)

    x_grids = Dict()
    y_grids = Dict()

    smallg_sol = Dict()

    for i in 1:num_runs
        if i == 1
            st_indx[i] = 5
            st_indy[i] = 14
        else
            st_indx[i] = st_indx[i-1] * kres - 1
            st_indy[i] = st_indy[i-1] * kres - 1
        end
    end

    for k in kstart:kend
        ind = k - kstart + 1
        scheme, mesh, temperature = run(tfinal, k, kres, Inf)

        xc, yc = centroids(mesh)

        domain = mesh.iterators.cell.domain
        sim = @view temperature[domain]

        sol = u_analytic.(xc, yc, tfinal)

        if k == kstart
            x_grids[ind] = xc[st_indx[ind]:end, st_indy[ind]:(kres^(ind)):end]
            y_grids[ind] = yc[st_indy[ind]:(kres^ind):end, st_indx[ind]:end]
            smallg_sol[ind] = sim[st_indy[ind]:(kres^(ind)):end, st_indy[ind]:(kres^(ind)):end]
        else
            #st_ind = st_ind * kres - 1
            #(ind - 1) * kres - 1)
            x_grids[ind] = xc[st_indx[ind]:(kres^(ind-1)):end, st_indy[ind]:(kres^(ind)):end]
            y_grids[ind] = yc[st_indy[ind]:(kres^(ind)):end, st_indx[ind]:(kres^(ind-1)):end]
            smallg_sol[ind] = sim[st_indy[ind]:(kres^(ind)):end, st_indy[ind]:(kres^(ind)):end]
        end

        dx[ind] = xc[2, 1] - xc[1, 1]
        abs_err[ind] = norm(sim[:, :] - sol[:, :], Inf)
        my_L₂[ind] = norm(sim[:, :] - sol[:, :], 2)
        rel_err[ind] = maximum(abs.((sim[:, :] .- sol[:, :]) ./ sol[:, :]))
        println(
            "gres is $(kres^k) and abs_err = $(norm(sim[:, :] - sol[:, :], Inf)), L₂_err = $(my_L₂[ind]), rel_err = $(rel_err[ind])",
        )
        if k > kstart
            conv_rate[ind-1] =
                log.(abs_err[ind] / abs_err[ind-1]) / log.(dx[ind] / dx[ind-1])
        end

        if k == kstart
            xs[:] = xc[:, :]
        end

        if k == kplot
            #mid = Int(floor((2^k) / 2))
            #mid = 22
            simp[:] = sim[:, mid]      #full 1D simulation (on interior domain)
            solp[:] = sol[:, mid]      #full 1D exact solution (on interior domain)
            errs[:] = abs.(sim[:, mid] - sol[:, mid])   #errors along cross section of x dimension
            x[:] = xc[:, 1]                           #cell centered x array on interior domain
            y[:] = yc[1, :]                           #cell sentered y array on interior domain
            x2D[:, :] = xc[:, :]
            y2D[:, :] = yc[:, :]

            dy = yc[mid, 2] - yc[mid, 1]                #grid spacing in y
            simf[:] = temperature[:, mid]              #full 1D simulation along x (including ghost points)
            simf2D[:, :] = temperature[:, :]          #full 2D simulation (including ghost points)
            xf[:] = copy(xc[:, mid])                   #full cell centered x array (to include ghost points)
            yf[:] = copy(yc[mid, :])                   #full cell centered y array (to include ghost points)

            simng[:, :] = sim[:, :]

            #fill in the ghost point values for x and y grid
            ind1 = xf[1]
            inde = xf[end]
            ind1y = yf[1]
            indey = yf[end]
            pushfirst!(
                xf,
                ind1 - 5dx[ind],
                ind1 - 4dx[ind],
                ind1 - 3dx[ind],
                ind1 - 2dx[ind],
                ind1 - dx[ind],
            )
            push!(
                xf,
                inde + dx[ind],
                inde + 2dx[ind],
                inde + 3dx[ind],
                inde + 4dx[ind],
                inde + 5dx[ind],
            )

            pushfirst!(yf, ind1y - 5dy, ind1y - 4dy, ind1y - 3dy, ind1y - 2dy, ind1y - dy)
            push!(yf, indey + dy, indey + 2dy, indey + 3dy, indey + 4dy, indey + 5dy)
        end

        #plot_sol(k, kres, sim, sol, xc, yc)
    end

    #=
    p1 = plot(
        dx,
        (dx .^ 1) / 22;
        label="Reference dx",
        xlabel="dx",
        ylabel="Errors (log10)",
        linestyle=:solid,
        xscale=:log10,
        yscale=:log10,
        xticks=(dx, dx),
      )
    =#

    # Plot error convergence, initial condition, and final solution vs exact solution
    p1 = plot(
        dx,
        (dx .^ 2) / 1.9;
        label="Reference dx^2",
        xlabel="dx",
        ylabel="Errors (log10)",
        linestyle=:solid,
        xscale=:log10,
        yscale=:log10,
        xticks=(round.(dx; digits=3), round.(dx; digits=3)),
        title="2D 'Wavy Mesh 2' Spatial Convergence",
    )
    n = length(x)
    u_init = zeros(n, n)
    for i in 1:n
        for j in 1:n
            u_init[i, j] = u_initial(x2D[i, j], y2D[i, j], 0)
        end
    end
    p3 = plot(x, u_init[:, mid]; label="Initial u", xlabel="x", ylabel="u", linestyle=:solid)
    p4 = plot(x, simp; label="simulated (nx=ny=2⁷)", lw=5, xlabel="x", ylabel="u")
    p5 = scatter!(
        p4, x, solp; ms=2.0, markershape=:diamond, markercolor=:black, label="analytic"
    )
    plot!(
        p1,
        dx,
        abs_err;
        label="Abs error",
        xlabel="dx",
        ylabel="Errors (log10)",
        linestyle=:dash,
        marker=:circle,
        xscale=:log10,
        yscale=:log10,
    )
    p2 = plot(x, errs; label="errors", lw=5, xlabel="x", ylabel="Errors")
    p = plot(p1, p3, p4; layout=(3, 1))
    #p_paper_space_conv = plot(p1, p4; layout=(2, 1))
    formatted_abs = @sprintf("%.2e", abs_err[1])
    formatted_rel = @sprintf("%.2e", rel_err[1])
    p_paper_t_errors = plot!(
        p4,
        x,
        u_init[:, mid];
        color=:red,
        lw=1.5,
        label="Initial u",
        title="2D 'Wavy Mesh 2' Non-Linear MS",
    )
    annotate!(-3.5, 0.55, text("Abs. error = $(formatted_abs)", 12))
    annotate!(-3.5, 0.45, text("Rel. error = $(formatted_rel)", 12))
    @show conv_rate

    # Plot 1D snapshot along x including ghost points
    pf = plot(
        xf,
        simf;
        label="full sim(with halo) in x",
        lw=5,
        xlabel="x",
        ylabel="Temp",
        marker=:square,
    )

    # Plot 1D snapshot along y including ghost points
    pfy = plot(
        yf,
        simf2D[6, :];
        label="full sim(with halo) in y",
        lw=5,
        xlabel="y",
        ylabel="Temp",
        marker=:square,
    )

    # Plot full 2D surface (including ghost points)
    s = surface(xf, yf, simf2D')

    # Plot full 2D errors (including ghost points)
    solf = zeros(kres^kplot + 10, kres^kplot + 10)
    for i in 1:length(xf)
        for j in 1:length(xf)
            solf[i, j] = u_analytic(xf[i], yf[j], t)
        end
    end

    errf = abs.(simf2D - solf)
    se = surface(xf, yf, errf')

    solng = zeros(kres^kplot, kres^kplot)
    for i in 1:length(x)
        for j in 1:length(y)
            solng[i, j] = u_analytic(x2D[i, j], y2D[i, j], t)
        end
    end

    # Plot 2D errors (no ghost points)
    errng = abs.(simng - solng)
    sng = surface(x2D, y2D, simng')
    seng = surface(
        x2D,
        y2D,
        errng';
        xlabel="x",
        ylabel="y",
        zlabel="u",
        xlimits=(-6, 6),
        ylimits=(-6, 6),
        title="2D Curvilinear TZ Errors",
    )
    #@show yf

    display(p_paper_t_errors)
    #display(pf)
    #savefig(p, "TZ_convergence_curvilinear_2D.png")
    #savefig(seng, "TZ_curvilinear_2D_errors.png")
    #savefig(p_paper_space_conv, "TZ_wavy_2D_conv_paper.png")
    #savefig(p_paper_t_errors, "TZ_wavy_witht_error.png")

    @show abs_err
end
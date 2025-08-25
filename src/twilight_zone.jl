
"""
    define_forcing(
  (αᵢⱼ, βᵢⱼ), cell_center_metrics, coords, nhalo, nnodes, α, t, tz_type, tz_dim, no_t, idx
)

 Defines the forcing function for twilight zone verification
    tz_type: Defines which grid type TZ is being done on. :cart -> uniform cartesian, 
             :axisym -> 2D axisymmetric, :wavy -> distorted mesh
    tz_dim: Specifies the number of spatial dimensions the exact solution is. Either 1 or 2
    no_t: Specifies if the exact solution is dependent on t or not. no_t = true checks spatial convergence
"""
function define_forcing(
    (αᵢⱼ, βᵢⱼ),
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
    if tz_type == :wavy
        f = define_forcing_wavy(
            (αᵢⱼ, βᵢⱼ),
            cell_center_metrics,
            coords,
            nhalo,
            nnodes,
            α,
            ρ,
            cₚ,
            source_term,
            t,
            tz_dim,
            no_t,
            idx,
        )
    else
        @error(
            "Manufactured solutions are not implemented for this grid type. The current option is :wavy. You provided $(tz_type)."
        )
    end

    return f
end

function define_forcing_wavy(
    (αᵢⱼ, βᵢⱼ),
    cell_center_metrics,
    coords,
    nhalo,
    nnodes,
    α,
    ρ,
    cₚ,
    source_term,
    t,
    tz_dim,
    no_t,
    idx,
)
    @inbounds begin
        i, j = idx.I    #loop over interior domain

        nx, ny = nnodes #number of nodes (edges, NOT cell centers)

        x = coords.x[i, j]  #cell centered x coordinate
        y = coords.y[i, j]  #cell centered y coordinate

        s = source_term[i, j] #source term at index i,j

        #curvilinear grid metrics
        ξx = cell_center_metrics.ξ.x₁[i, j]
        ξy = cell_center_metrics.ξ.x₂[i, j]
        ηx = cell_center_metrics.η.x₁[i, j]
        ηy = cell_center_metrics.η.x₂[i, j]

        #specifies if the diffusivity is linear or nonlinear. nonlinear = true
        non_lin_α = true

        #grid construction parameters (needed for derivatives of grid)
        Lx = 12
        Ly = 12
        xmin = -Lx / 2
        ymin = -Ly / 2
        n_xy = 6
        n_yx = 6
        Δx0 = Lx / (nx - 1)
        Δy0 = Ly / (ny - 1)
        Ax = 0.2 / Δx0
        Ay = 0.4 / Δy0

        indi = (i - 5) + 1 / 2
        indj = (j - 5) + 1 / 2

        fwhm = 1.0
        x0 = 0
        y0 = 0

        damp = 4

        #wavy grid point

        xt = xmin + Δx0 * ((indi - 1) + Ax * sinpi(n_xy * (indj - 1) * Δy0 / Ly))
        yt = ymin + Δy0 * ((indj - 1) + Ay * sinpi(n_yx * (indi - 1) * Δx0 / Lx))

        #stored grid metric derivatives
        xξ = cell_center_metrics.x₁.ξ[i, j]
        xη = cell_center_metrics.x₁.η[i, j]
        yξ = cell_center_metrics.x₂.ξ[i, j]
        yη = cell_center_metrics.x₂.η[i, j]

        #grid metric second derivatives
        xξξ = 0
        xξη = 0
        xηη = -Δx0 * Ax * (π * n_xy * Δy0 / Ly)^2 * sinpi(n_xy * (indj - 1) * Δy0 / Ly)
        yξξ = -Δy0 * Ay * (π * n_yx * Δx0 / Lx)^2 * sinpi(n_yx * (indi - 1) * Δx0 / Lx)
        yξη = 0
        yηη = 0

        if no_t == true
            exy = exp(-(((x - x0)^2) / fwhm + ((y - y0)^2) / fwhm))

            ut = 0
            uξ = (-2 / fwhm) * ((x - x0) * xξ + (y - y0) * yξ) * exy
            uξξ =
                (
                    (2 * (x - x0) / fwhm * xξ + 2 * (y - y0) / fwhm * yξ)^2 -
                    (2 / fwhm) * (xξ^2 + yξ^2 + (x - x0) * xξξ + (y - y0) * yξξ)
                ) * exy
            uη = (-2 / fwhm) * ((x - x0) * xη + (y - y0) * yη) * exy
            uηη =
                (
                    (2 * (x - x0) / fwhm * xη + 2 * (y - y0) / fwhm * yη)^2 -
                    (2 / fwhm) * (xη^2 + yη^2 + (x - x0) * xηη + (y - y0) * yηη)
                ) * exy
            uξη =
                (
                    (2 * (x - x0) / fwhm * xξ + 2 * (y - y0) / fwhm * yξ) *
                    (2 * (x - x0) / fwhm * xη + 2 * (y - y0) / fwhm * yη) -
                    (2 / fwhm) * (xξ * xη + yξ * yη + (x - x0) * xξη + (y - y0) * yξη)
                ) * exy
        else
            exy = exp(-(((x - x0)^2) / fwhm + ((y - y0)^2) / fwhm)) * exp(-damp * t)

            ut = -damp * exy
            uξ = (-2 / fwhm) * ((x - x0) * xξ + (y - y0) * yξ) * exy
            uξξ =
                (
                    (2 * (x - x0) / fwhm * xξ + 2 * (y - y0) / fwhm * yξ)^2 -
                    (2 / fwhm) * (xξ^2 + yξ^2 + (x - x0) * xξξ + (y - y0) * yξξ)
                ) * exy
            uη = (-2 / fwhm) * ((x - x0) * xη + (y - y0) * yη) * exy
            uηη =
                (
                    (2 * (x - x0) / fwhm * xη + 2 * (y - y0) / fwhm * yη)^2 -
                    (2 / fwhm) * (xη^2 + yη^2 + (x - x0) * xηη + (y - y0) * yηη)
                ) * exy
            uξη =
                (
                    (2 * (x - x0) / fwhm * xξ + 2 * (y - y0) / fwhm * yξ) *
                    (2 * (x - x0) / fwhm * xη + 2 * (y - y0) / fwhm * yη) -
                    (2 / fwhm) * (xξ * xη + yξ * yη + (x - x0) * xξη + (y - y0) * yξη)
                ) * exy
        end

        #forcing function for 2d
        if non_lin_α == false
            orth_ξ = α[i, j] .* (ξx^2 .+ ξy^2) .* uξξ           #orthogonal in ξ term
            orth_η = α[i, j] .* (ηx^2 .+ ηy^2) .* uηη           #orthogonal in η term
            cross = 2 .* α[i, j] .* (ξx * ηx + ξy * ηy) .* uξη  #cross derivative term
            non_orth_ξ = α[i, j] .* αᵢⱼ[i, j] .* uξ                        #non-orthogonal du/dξ term
            non_orth_η = α[i, j] .* βᵢⱼ[i, j] .* uη                        #non-orthogonal du/dη term
        else
            αξ = ((5 / 2) * exy^(3 / 2) * uξ) / (ρ[i, j] * cₚ)  #ξ derivative of diffusivity
            αη = ((5 / 2) * exy^(3 / 2) * uη) / (ρ[i, j] * cₚ)  #η derivative of diffusivity
            orth_ξ = (ξx^2 .+ ξy^2) * (αξ * uξ + α[i, j] * uξξ) #orthogonal in ξ term
            orth_η = (ηx^2 .+ ηy^2) * (αη * uη + α[i, j] * uηη) #orthogonal in η term
            cross = (ξx * ηx + ξy * ηy) * (2 * α[i, j] * uξη + αη * uξ + αξ * uη) #cross derivative term
            non_orth_ξ = α[i, j] .* αᵢⱼ[i, j] .* uξ #non-orthogonal du/dξ term
            non_orth_η = α[i, j] .* βᵢⱼ[i, j] .* uη #non-orthogonal du/dη term
        end

        f = ut - orth_ξ - orth_η - cross - non_orth_ξ - non_orth_η - s  #forcing function
    end

    return f
end
using CairoMakie
using Unitful
using JSON3

function runtimes(fn)
  json_string = read(fn, String)
  data = JSON3.read(json_string)

  tot =
    Float64(data["inner_timers"]["nonlinear_thermal_conduction_step!"]["total_time_ns"]) *
    u"ns"
  n = data["inner_timers"]["nonlinear_thermal_conduction_step!"]["n_calls"]
  per_cycle = tot / n

  return tot |> u"s", per_cycle |> u"ms", n
end

dev = :GPU
resolution = [51, 101, 201]

cell_resolution = @. (resolution - 1)^3
krylov_gpu = [
  runtimes("$(@__DIR__)/benchmark_results/nonlinear_$(dev)_timing_$(res)_implicit.json") for
  res in resolution
]
pt_gpu = [
  runtimes(
    "$(@__DIR__)/benchmark_results/nonlinear_$(dev)_timing_$(res)_pseudo_transient.json"
  ) for res in resolution
]

dev = :CPU
krylov_cpu = [
  runtimes("$(@__DIR__)/benchmark_results/nonlinear_$(dev)_timing_$(res)_implicit.json") for
  res in resolution
]
pt_cpu = [
  runtimes(
    "$(@__DIR__)/benchmark_results/nonlinear_$(dev)_timing_$(res)_pseudo_transient.json"
  ) for res in resolution
]

fig = Figure()
ax = Axis(
  fig[1, 1];
  yscale=log10,
  xscale=log10,
  xticks=[1e4, 1e5, 1e6, 1e7],
  yticks=[1e3, 1e4, 1e5, 1e6, 1e7],
  xminorticksvisible=true,
  xminorgridvisible=true,
  yminorticksvisible=true,
  yminorgridvisible=true,
  xminorticks=IntervalsBetween(9),
  yminorticks=IntervalsBetween(9),
  xlabel="Total Resolution",
  ylabel="Runtime [ms]",
  title="3D Nonlinear Thermal Conduction (Problem 5)",
)
palette = Makie.wong_colors()

runtime_idx = 1
scatterlines!(
  ax,
  cell_resolution,
  [ustrip(u"ms", c[runtime_idx]) for c in krylov_gpu];
  label="GMRES [GPU]",
  color=palette[1],
)

scatterlines!(
  ax,
  cell_resolution,
  [ustrip(u"ms", c[runtime_idx]) for c in krylov_cpu];
  label="GMRES [CPU]",
  linestyle=:dash,
  color=palette[1],
)

scatterlines!(
  ax,
  cell_resolution,
  [ustrip(u"ms", c[runtime_idx]) for c in pt_gpu];
  label="PT [GPU]",
  color=palette[2],
)

scatterlines!(
  ax,
  cell_resolution,
  [ustrip(u"ms", c[runtime_idx]) for c in pt_cpu];
  label="PT [CPU]",
  linestyle=:dash,
  color=palette[2],
)

lines!(ax, cell_resolution, cell_resolution; label="y=x", color=:black)

xlims!(ax, 10^4.9, 10^7.25)
ylims!(ax, 1e3, 10^7.1)
# scatterlines!(
#   ax,
#   [case[1]^2 for case in accelerated_PT_CPU],
#   [ustrip(u"ms", case[case_idx]) for case in accelerated_PT_CPU];
#   label="PT solver [CPU]",
#   linestyle=:dash,
# )

# scatterlines!(
#   ax,
#   [case[1]^2 for case in krylov_GPU],
#   [ustrip(u"ms", case[case_idx]) for case in krylov_GPU];
#   label="Krylov solver [GPU]",
# )

# scatterlines!(
#   ax,
#   [case[1]^2 for case in accelerated_PT_GPU],
#   [ustrip(u"ms", case[case_idx]) for case in accelerated_PT_GPU];
#   label="PT solver [GPU]",
# )

Legend(fig[1, 2], ax)
display(fig)

save("$(@__DIR__)/nonlinear_3dbenchmarks.png", fig)

println("krylov_cpu");
[round(typeof(1u"ms"), c[2]) for c in krylov_cpu]
println("krylov_gpu");
[round(typeof(1u"ms"), c[2]) for c in krylov_gpu]
println("pt_cpu");
[round(typeof(1u"ms"), c[2]) for c in pt_cpu]
println("pt_gpu");
[round(typeof(1u"ms"), c[2]) for c in pt_gpu]

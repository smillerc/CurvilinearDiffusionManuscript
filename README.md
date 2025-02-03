# CurvilinearDiffusion

This repository is to allow users to reproduce the results presented in the manuscript: "" recently submitted to X.

## Installing

`CurvilinearDiffusion.jl` isn't a registered package at the moment, so it needs to be cloned and run locally.

```bash
git clone https://github.com/smillerc/CurvilinearDiffusionManuscript
cd CurvilinearDiffusionManuscript
julia --project # run using CurvilinearDiffusionManuscript as the local environment
```
and within julia
```julia
julia> ] # switch to package mode
(CurvilinearDiffusion) pkg> resolve # download dependencies
# switch back to normal mode (hit backspace a few times)
julia> using CurvilinearDiffusion # load the package
```

# Examples

Check out the `run.jl` file for each test problem in the `test/integrated` directories.
```bash
cd $HOME/CurvilinearDiffusionManuscript/test/integrated/2d/gaussian
julia -t 16 --project=${HOME}/CurvilinearDiffusionManuscript run.jl
```

The benchmark cases in the manuscript are also within these test directories. These run all three solver methods (implicit direct, implicit GMRES, and accelerated pseudo-transient) for various grid resolutions.

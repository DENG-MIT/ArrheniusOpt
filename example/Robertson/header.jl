using Flux
using Flux.Optimise: update!
using Flux.Losses: mae, mse
using DiffEqFlux
using DiffEqSensitivity
using DifferentialEquations
using LinearAlgebra
using Random
using ProgressBars
using DelimitedFiles
using Plots, Colors, Printf, Profile

BLAS.set_num_threads(4);
Threads.nthreads() = 4;

# settings for plots
Plots.pyplot()
Plots.resetfontsizes()
Plots.scalefontsizes(1.5)
function valid(tsteps, y_true, y_pred; title="", xscale=:log10,
                line1=(3, :scatter), line2=(2, :solid) )
    # IJulia.clear_output(true)
    h = plot(legend=false, title=title, size=(600,350), framestyle=:box,
             palette=palette(:darktest, length(weights)))
    plot!(tsteps, (y_true .* weights)', xscale=xscale, line=line1, msw=0.0)
    plot!(tsteps, (y_pred .* weights)', xscale=xscale, line=line2)
    return h
end

println("[info] Header.jl loaded");

using Arrhenius
using ForwardDiff
using LinearAlgebra
using DiffEqSensitivity
using DifferentialEquations

using Random
using ProgressBars
using DelimitedFiles
using Plots, Colors, Printf, Profile

using Flux
using Flux: crossentropy
using Flux.Losses: mae
using Flux.Optimise: update!
using LatinHypercubeSampling
using Statistics
using DiffEqFlux

BLAS.set_num_threads(4);
Threads.nthreads() = 4;

# settings
P = one_atm;
gas = CreateSolution("../../mechanism/JP10skeletal.yaml");
ns = gas.n_species;
nr = gas.n_reactions;
@show gas.species_names;

solver = TRBDF2();
datasize = 100;
tsteps = 10 .^ range(log10(1e-8), log10(1e-1), length=datasize);
tspan = [0.0, tsteps[end]*1.001];

rng = MersenneTwister(0x7777777);
p_true = zeros(nr);
p_init = 0.5*(2*rand(rng, nr).-1);
p_pred = deepcopy(p_init);
grad_max = 10 ^ (1);

# ODE function
function dudtp!(du, u, p, t)
    T = u[end]
    Y = @view(u[1:ns])

    mean_MW = 1. / dot(Y, 1 ./ gas.MW)
    ρ_mass = P / R / T * mean_MW
    X = Y2X(gas, Y, mean_MW)
    C = Y2C(gas, Y, ρ_mass)
    cp_mole, cp_mass = get_cp(gas, T, X, mean_MW)
    h_mole = get_H(gas, T, Y, X)
    S0 = get_S(gas, T, P, X)

    qdot = wdot_func(gas.reaction, T, C, S0, h_mole, get_qdot = true)
    wdot = gas.reaction.vk * (qdot .* exp.(p))

    Ydot = wdot / ρ_mass .* gas.MW
    Tdot = -dot(h_mole, wdot) / ρ_mass / cp_mass
    du .= vcat(Ydot, Tdot)
end

Y0 = zeros(ns);
Y0[species_index(gas, "C10H16")] = 0.05;
Y0[species_index(gas, "N2")] = 0.95;
T0 = 1200.0;
P = one_atm;
u0 = vcat(Y0, T0);
prob = ODEProblem(dudtp!, u0, tspan);

## prepare plotting
using PyPlot
using Plots: plot, grid
Plots.pyplot()
Plots.resetfontsizes()
Plots.scalefontsizes(2.0)
function valid(t, y_true, y_pred, xscale)
    h1 = plot(legend=false)
    plot!(t, y_true[species_index(gas, "C10H16"),:], line=(1,:scatter), msw=0, label="Groundtruth")
    plot!(t, y_pred[species_index(gas, "C10H16"),:], line=(1,:solid), label="Prediction")
    xticks!([0.0, 0.03, 0.06, 0.09])
    ylabel!(h1, "Y(C10H16)")
    xlabel!(h1, "Time [s]")

    h2 = plot(legend=false)
    plot!(t, y_true[end,:], line=(1,:scatter), msw=0, label="Groundtruth")
    plot!(t, y_pred[end,:], line=(1,:solid), label="Prediction")
    xticks!([0.0, 0.03, 0.06, 0.09])
    ylabel!(h2, "T [K]")
    xlabel!(h2, "Time [s]")

    xlim = [5e-6, 1e-1];
    scale = maximum(abs.(y_true), dims=2)
    scale = max.(scale, 1e-16)
    h3 = plot(palette=palette(:default, length(scale)), xscale=xscale,
            xlim=xlim, legend=false)
    plot!(t[2:end], (y_true[:,2:end]./scale)', line=(1, :scatter), msw=0)
    plot!(t[2:end], (y_pred[:,2:end]./scale)', line=(1, :solid))
    ylabel!(h3, "Normalized Y, T")
    xlabel!(h3, "Time [s]")

    h = plot(h1, h2, h3, layout=@layout[grid(2,1) a{0.7w}], size=(800,350), framestyle=:box, fg_legend=:transparent)
    # display(h)
    sleep(1e-9)
    return h;
end

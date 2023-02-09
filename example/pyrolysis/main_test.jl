include("header.jl")

# sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))
# sensealg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(false))
sensealg = ForwardDiffSensitivity()
# sensealg = ForwardSensitivity(autojacvec=true)
function fsol(p)
    sol = solve(prob, u0=u0, solver, p=p, tspan=tspan, saveat=tsteps,
                reltol=1e-9, abstol=1e-12, sensealg=sensealg)
    return sol[end, end]
end

# solver = KenCarp4()
# solver = Rosenbrock23()
solver = TRBDF2()
println("timing ode solver ...")
@time fsol(p_true);
@time fsol(p_true);
@time Flux.gradient(fsol, p_true);
@time Flux.gradient(fsol, p_true);

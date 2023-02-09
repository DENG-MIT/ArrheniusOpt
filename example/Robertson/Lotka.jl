casename = "Lotka"

# the ODE function
function f(u,p,t)
    dx = p[1]*u[1] - p[2]*u[1]*u[2]
    dy = -p[3]*u[2] + p[4]*u[1]*u[2]
    return [dx, dy]
end

p_true = [1.5, 1.0, 3.0, 1.0];
y0 = [1.0, 1.0];
datasize = 6;
tsteps = range(0.0, 5.0, length=datasize);
tspan = [0.0, tsteps[end]+1e-6];

solver = Rosenbrock23()
function predict(p)
    _sol = solve(prob, solver, p=p, saveat=tsteps);
    _pred = Array(_sol)
    if _sol.retcode == :Success
        nothing
    else
        println("ode solver failed")
    end
    return _pred
end

function loss(p; y_train=y_true)
    return sum(abs2, (predict(p) - y_train) ./ scale) # / length(y_train);
end

# get data with p_true
# forward solve
prob = ODEProblem(f, y0, tspan);
y_true = predict(p_true);
scale = vec(maximum(y_true, dims=2));

# backward solve
y_end = y_true[:,end]
rev_prob = ODEProblem(f, y_end, reverse(tspan));
rev_sol = solve(rev_prob, solver, p=p_true, saveat=tsteps)
plot(rev_sol)

# for valid plot
weights = [1, 1];
xscale = :identity;

# for training
n_epoch = 100;

println("[info] Lotka ODE loaded");

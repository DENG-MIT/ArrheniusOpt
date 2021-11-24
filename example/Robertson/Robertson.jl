casename = "Robertson"

# the ODE function
function f(y, p, t)
    k = [0.04, 3e7, 1e4] .* p
    dydt1 = -k[1]*y[1]+k[3]*y[2]*y[3]
    dydt2 =  k[1]*y[1]-k[2]*y[2]^2-k[3]*y[2]*y[3]
    dydt3 =  k[2]*y[2]^2
    return [dydt1, dydt2, dydt3]
end

p_true = [1, 1, 1];
y0 = [1.0, 0.0, 0.0];
datasize = 100;
tsteps = 10 .^ range(log10(1e-4), log10(1e8), length=datasize);
tspan = (0.0, tsteps[end]+1e-3);

# solver = DifferentialEquations.CVODE_BDF()
solver = KenCarp4();

function predict(p)
    _sol = solve(prob, solver, p=p, saveat=tsteps);
    _pred = Array(_sol)
    if _sol.retcode == :Success
        nothing
    else
        println("ODE solver failed")
    end
    return _pred
end

function loss(p; y_train=y_true)
    return sum(abs2, (predict(p) - y_train) ./ scale) / length(y_train);
end

# get data with p_true
prob = ODEProblem(f, y0, tspan, p_true);
y_true = predict(p_true);

scale = vec(maximum(y_true, dims=2));

# for valid plot
weights = [1, 2e4, 1];
xscale = :log10;

# for training
n_epoch = 300;

println("[info] Robertson ODE loaded")

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
datasize = 50;
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
n_epoch = 200;

# fnt = Plots.font("DejaVu Sans", 10.0)
# default(titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)
function compare_Robertson(tsteps, y_true, y_init, y_pred; xscale=:log10,
                line1=(3, :scatter), line2=(2, :dash), line3=(2, :solid))
    fig, axs = PyPlot.subplots(3,1, figsize=(4,5))
    for i in 1:3
        axs[i].plot(tsteps, y_true[i,:], "k.", label="Data")
        axs[i].plot(tsteps, y_init[i,:], "b--", label="Initial")
        axs[i].plot(tsteps, y_pred[i,:], "r-", label="Optimized")
        axs[i].set_xscale("log")
        axs[i].set_ylabel("\$y_{$i}\$")
    end
    axs[2].legend(loc="best", frameon=false)
    axs[3].set_xlabel("Time [s]")
    fig.subplots_adjust(left=0.15, right=0.98, hspace=0.35, top=0.98)
    return fig
end

println("[info] Robertson ODE loaded")

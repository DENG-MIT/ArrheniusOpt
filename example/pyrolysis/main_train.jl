include("header.jl")

## generate datasets
n_exp_train = 20;
n_exp_valid = 5;
n_exp = n_exp_train + n_exp_valid;

noise = 0;
abstol = 1e-9;
reltol = 1e-12;

rng = MersenneTwister(0x7777777);

u0_list = zeros(Float64, (n_exp, ns+1));
randvals = randomLHC(rng, n_exp, 2) ./ n_exp;
for i in range(1, length=n_exp)
    Y0 = zeros(ns);
    if i <= n_exp_train
        T0 = 1000 + 200*randvals[i,1]; # 1000-1200K
        val = 0.02 + 0.18*randvals[i,2]; # Y[JP10] = 0.02-0.2
    else
        T0 = 1200 + 200*randvals[i,1]; # 1200-1400K
        val = 0.2 + 0.1*randvals[i,2]; # Y[JP10] = 0.2-0.3
    end
    Y0[species_index(gas, "C10H16")] = val;
    Y0[species_index(gas, "N2")] = 1 - val;
    u0 = vcat(Y0, T0);
    u0_list[i,:] = u0;
end
P = one_atm;

datasize = 100;
tsteps = 10 .^ range(log10(1e-8), log10(1e-1), length=datasize);
tspan = [0.0, tsteps[end]*1.001];

y_true_list = zeros(Float64, (n_exp, ns+1, datasize));
y_init_list = zeros(Float64, (n_exp, ns+1, datasize));
yscale_list = [];

max_min(y) = maximum(y, dims=2) .- minimum(y, dims=2);
for i = 1:n_exp
    u0 = u0_list[i, :]
    _prob = ODEProblem(dudtp!, u0, tspan, p_true)
    y_true = Array(solve(_prob, solver, saveat=tsteps, abstol=abstol, reltol=reltol))
    y_true += noise * randn(size(y_true)) .* y_true
    y_true_list[i, :, :] = y_true
    push!(yscale_list, max_min(y_true))
end

for i = 1:n_exp
    u0 = u0_list[i, :]
    _prob = ODEProblem(dudtp!, u0, tspan, p_init)
    y_init = Array(solve(_prob, solver, saveat=tsteps, abstol=abstol, reltol=reltol))
    y_init_list[i, :, :] = y_init
end

yscale_raw = maximum(abs.(hcat(yscale_list...)), dims=2);
yscale = max.(yscale_raw, 1e-16);
# show(stdout, "text/plain", yscale');

y_true = y_true_list[1,:,:]
y_init = y_init_list[1,:,:]
h = valid(tsteps, y_true, y_init, :log10)
Plots.savefig(h, "figures/JP10sk_initial_train_1.svg")

y_true = y_true_list[n_exp_train+1,:,:]
y_init = y_init_list[n_exp_train+1,:,:]
h = valid(tsteps, y_true, y_init, :log10)
Plots.savefig(h, "figures/JP10sk_initial_valid_1.svg")

# Regularization of parameters
vecnorm(x) = sum(abs2,x)./length(x)
vecnorm(p_init-p_true)

sensealg = ForwardDiffSensitivity()
function predict_ode(u0, p; sample = datasize)
    _prob = remake(prob, u0=u0, p=p, tspan=[0, tsteps[sample]])
    sol = solve(_prob, solver, saveat=tsteps[1:sample], sensalg=sensealg,
                reltol=reltol, abstol=abstol, verbose=false)
    pred = Array(sol)
    if sol.retcode == :Success
        nothing
    else
        println("ode solver failed")
    end
    return pred
end

function loss_ode(p, i_exp; abstol=1e-12, sample = datasize)
    y_pred = predict_ode(u0_list[i_exp,:], p; sample)
    y_true = y_true_list[i_exp,:,1:sample]
    yscale = max.(yscale_list[i_exp], abstol);
    loss = mae(y_true./yscale, y_pred./yscale) + vecnorm(p-p_init)/100
    return loss
end

@show loss_ode(p_init, 1)

@show loss_ode(p_init, n_exp_train+1; abstol=1e-9)


## prepare training
losses_y_train = Vector{Float64}();
losses_y_valid = Vector{Float64}();
losses_p = Vector{Float64}();
history_p_pred = Vector{Array{Float64}}();

# save losses
loss_epoch = zeros(Float64, n_exp);
for i_exp in 1:n_exp
    loss_epoch[i_exp] = loss_ode(p_pred, i_exp; abstol=abstol)
end
loss_y_train = mean(loss_epoch[1:n_exp_train]);
loss_y_valid = mean(loss_epoch[n_exp_train+1:end]);
loss_p = mae(p_pred, p_true)
push!(losses_y_train, loss_y_train)
push!(losses_y_valid, loss_y_valid)
push!(losses_p, loss_p)
push!(history_p_pred, p_init);

function train(opt; n_epoch=10, batchsize=50, reltol=1e-6, abstol=1e-9)
    epochs = ProgressBar(1:n_epoch);
    loss_epoch = zeros(Float64, n_exp);
    grad_norm = zeros(Float64, n_exp_train);
    for epoch in epochs
        # update parameters
        global p_pred
        for i_exp in randperm(n_exp_train)
            sample = rand(batchsize:datasize)
            grad = ForwardDiff.gradient(
                        x -> loss_ode(x, i_exp; abstol=abstol, sample),
                        p_pred)
            grad_norm[i_exp] = norm(grad, 2)
            if grad_norm[i_exp] > grad_max
                grad = grad ./ grad_norm[i_exp] .* grad_max
            end
            update!(opt, p_pred, grad)
        end

        # save losses
        for i_exp in 1:n_exp
            loss_epoch[i_exp] = loss_ode(p_pred, i_exp; abstol=abstol)
        end
        loss_y_train = mean(loss_epoch[1:n_exp_train]);
        loss_y_valid = mean(loss_epoch[n_exp_train+1:end]);
        loss_p = mae(p_pred, p_true)
        push!(history_p_pred, deepcopy(p_pred))
        push!(losses_y_train, loss_y_train)
        push!(losses_y_valid, loss_y_valid)
        push!(losses_p, loss_p)

        # show results
        if epoch%5==0
            u0 = u0_list[n_exp_train+1,:]
            _prob = remake(prob, u0=u0, p=p_pred, tspan=tspan)
            y_pred = Array(solve(_prob, solver, saveat=tsteps, sensalg=sensealg,
                        reltol=reltol, abstol=abstol, verbose=false))
            valid(tsteps, y_true_list[n_exp_train+1,:,:], y_pred, :log10)
        end
        set_description(epochs, string(@sprintf("Loss ytrain %.3e yvalid %.3e p %.3e gnorm %.3e",
                    loss_y_train, loss_y_valid, loss_p, mean(grad_norm))))
    end
end

opt = ADAMW(0.05, (0.9, 0.999), 1.f-6);
train(opt; n_epoch=10, batchsize=10, reltol=1e-6, abstol=1e-9)

opt = ADAMW(0.02, (0.9, 0.999), 1.f-6);
train(opt; n_epoch=40, batchsize=50, reltol=1e-6, abstol=1e-9)

opt = ADAMW(0.01, (0.9, 0.999), 1.f-6);
train(opt; n_epoch=50, batchsize=100, reltol=1e-6, abstol=1e-9)

## show loss resutls
h1 = plot(yscale=:log10, xscale=:log10)
plot!(losses_y_train, lc=1, label="y loss train")
plot!(losses_y_valid, lc=2, label="y loss valid")
xlabel!("Epoch")
xticks!([1, 10, 100])

h2 = plot(yscale=:log10, xscale=:log10)
plot!(losses_p, lc=1, label="p loss")
xlabel!("Epoch")
xticks!([1, 10, 100])

h = plot(h1, h2, legend=true, size=(800,200), framestyle=:box)
Plots.savefig(h, "figures/JP10sk_losses.svg")

## show training results
reltol = 1e-9;
abstol = 1e-12;
i_exp = 1
u0 = u0_list[i_exp,:]
y_true = y_true_list[i_exp,:,:]
_prob = remake(prob, u0=u0, p=p_pred, tspan=tspan)
y_pred = solve(_prob, solver, saveat=tsteps, sensalg=sensealg,
            reltol=reltol, abstol=abstol, verbose=false)
h = valid(tsteps, y_true, y_pred, :log10)
Plots.savefig(h, "figures/JP10sk_trained_train_1.svg")

## show training resutls
i_exp = n_exp_train + 5
u0 = u0_list[i_exp,:]
y_true = y_true_list[i_exp,:,:]
_prob = remake(prob, u0=u0, p=p_pred, tspan=tspan)
y_pred = solve(_prob, solver, saveat=tsteps, sensalg=sensealg,
            reltol=reltol, abstol=abstol, verbose=false)
h = valid(tsteps, y_true, y_pred, :log10)
Plots.savefig(h, "figures/JP10sk_trained_valid_1.svg")

## show parameters results
p_indexs = Vector{Int64}();
plot(p_pred - p_init)
for i in 1:nr
    if abs(p_pred[i] - p_init[i]) > 0.01
        push!(p_indexs, i)
    end
end
using Plots: scatter
sens = sqrt.(sqrt.(sqrt.(abs.(grad_nominal)))) ./ 3 # .* sign.(grad_nominal)
h = plot(legend=:outerright, size=(1000,200))
scatter!(p_indexs, (p_init[p_indexs]),markersize=8, label="p_init")
scatter!(p_indexs, (p_pred[p_indexs]),markersize=8, label="p_pred")
Plots.bar!(1:232, sens', color=:green, mse=0., alpha = 0.3, label="sensitivity")
xlabel!("Index")
Plots.savefig(h, "figures/JP10sk_p_update_sens.svg")

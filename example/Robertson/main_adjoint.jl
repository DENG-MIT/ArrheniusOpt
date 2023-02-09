include("header.jl")

include("Lotka.jl")
include("backsolveAdjoint.jl") # not working for Robertson
# include("discreteAdjoint.jl")
# solver = KenCarp4()
# include("interpolatingAdjoint.jl")

# include("Robertson.jl")
# include("discreteAdjoint.jl")
# include("interpolatingAdjoint.jl")

function train(p_init, y_true; n_epoch=100, opt = ADAMW(0.1,(0.9,0.999),1e-6), doplot=false)
    p_pred = deepcopy(p_init);
    y_pred = predict(p_pred)
    losses_y = Vector{Float64}([loss(p_init; y_train=y_true)]);
    history_p = Vector{Array{Float64}}([p_init]);

    display(valid(tsteps, y_true, y_pred; xscale=xscale))

    epochs = ProgressBar(1:n_epoch);
    for epoch in epochs
        y_pred = predict(p_pred);

        grad = grad_adjoint(p_pred, y_pred, y_true)
        # grad = Flux.gradient(x -> loss(x), p_pred)[1]

        update!(opt, p_pred, grad)

        loss_y = loss(p_pred; y_train=y_true)
        push!(losses_y, deepcopy(loss_y))
        push!(history_p, deepcopy(p_pred))

        if epoch%10 == 0
            display(valid(tsteps, y_true, y_pred; xscale=xscale))
        end
        set_description(epochs, string(@sprintf("loss_y = %.3e, loss_p %.3e grad %.3e",
                    loss_y, norm(p_pred-p_true)^2, norm(grad))))
    end
    return losses_y, history_p;
end

# initialize
noise_level = 1e-1;
rng = MersenneTwister(Int32(floor(1e7*noise_level)));
p_init = exp.((rand(rng, length(p_true))*1 .- 0.5) * 2) .* p_true;
y_init = predict(p_init);
y_noise = y_true + noise_level .* (rand(rng, length(y0), length(tsteps)).-0.5) .* scale;
h = valid(tsteps, y_noise, y_init; xscale=xscale)
Plots.savefig(h, string(@sprintf("figures/%s_noise=%.0e_init.svg", casename, noise_level)));

# # show adjoint state
grad_adj, h = grad_adjoint(p_init, y_init, y_noise, doplot=true);
Plots.savefig(h, string(@sprintf("figures/%s_noise=%.0e_AdjointState.svg", casename, noise_level)))

# # time performance
# @time grad_ad = Flux.gradient(x -> loss(x; y_train=y_true), p_init)[1];
# @time grad_adj = grad_adjoint(p_init, y_init, y_true; doplot=false);
# @show grad_ad';
# @show grad_adj';

# training
losses_y, history_p = train(p_init, y_noise; n_epoch=n_epoch);
losses_p = vcat(sum(abs2.(hcat(history_p...) .- p_true), dims=1)...);

# y & p loss
h1 = plot(xlabel="Epochs", ylabel="y loss", size=(400,200), legend=false);
plot!(losses_y, yscale=:log10, line=(1, :solid), color=:black);
h2 = plot(xlabel="Epochs", ylabel="p loss", size=(400,200), legend=false);
plot!(losses_p, yscale=:log10, line=(1, :solid), color=:black);
h = plot(h1, h2, layout=(2,1), size=(400,400), framestyle=:box);
Plots.savefig(h, string(@sprintf("figures/%s_noise=%.0e_loss.svg", casename, noise_level)));

# prediction
p_pred = history_p[end];
y_pred = predict(p_pred);
h = valid(tsteps, y_noise, y_pred; xscale=xscale);
Plots.savefig(h, string(@sprintf("figures/%s_noise=%.0e_pred.svg", casename, noise_level)));

# # show adjoint state
grad_adj, h = grad_adjoint(p_true, y_true, y_noise, doplot=true);
Plots.savefig(h, string(@sprintf("figures/%s_noise=%.0e_AdjointState_trained.svg", casename, noise_level)))

# h = compare_Robertson(tsteps, y_noise, y_init, y_pred; xscale=xscale)
# Plots.savefig(h, string(@sprintf("figures/%s_noise=%.0e_comp.svg", casename, noise_level)));

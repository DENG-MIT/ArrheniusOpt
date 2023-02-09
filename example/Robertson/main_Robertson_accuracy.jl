include("header.jl")
include("Robertson.jl")

# training under a series of noises
rng = MersenneTwister(0x7777777);
n_exp = 100;
n_epoch = 300;
p_init_arr = exp.((rand(rng, length(p_true), n_exp) .* 2 .- 1));
noise_level_arr = 10 .^ (rand(rng, n_exp) .* (log10(0.2) - log10(10^-3)) .+ log10(10^-3)); # [(0.1% ~ 20%)]

losses_y_exps = zeros(n_exp, n_epoch+1);
losses_p_exps = zeros(n_exp, n_epoch+1);
history_p_exps = zeros(n_exp, n_epoch+1, length(p_true));

for i in 1:n_exp
    p_init = p_init_arr[:,i];
    noise_level = noise_level_arr[i];

    rng = MersenneTwister(Int32(floor(1e7*noise_level)));
    y_noise = y_true + noise_level .* (rand(rng, length(y0), length(tsteps)).-0.5) .* scale;

    losses_y, history_p = train(p_init, y_noise; n_epoch=n_epoch);
    losses_p = vcat(sum(abs2.(hcat(history_p...) .- p_true), dims=1)...);

    losses_y_exps[i,:] = deepcopy(losses_y);
    losses_p_exps[i,:] = deepcopy(losses_p);
    history_p_exps[i,:,:] = deepcopy(hcat(history_p...)');
    @show i, losses_y[end], losses_p[end];
end

# accuracy plot
h1 = plot(xlabel="Noise", ylabel="Final y loss", size=(400,200), legend=false);
plot!(noise_level_arr, sqrt.(losses_y_exps[:,end]./datasize ./ 3),  xscale=:log10, yscale=:log10, line=(3, :scatter), color=:black)
h2 = plot(xlabel="Noise", ylabel="Final p loss", size=(400,200), legend=false);
plot!(noise_level_arr, sqrt.(losses_p_exps[:,end]./3),  xscale=:log10, yscale=:log10, line=(3, :scatter), color=:black)
h = plot(h1, h2, layout=(2,1), size=(400,400), framestyle=:box)
Plots.savefig(h, "figures/Robertson_noises_accuracy.svg")

# losses plot
h1 = plot(xlabel="Epochs", ylabel="y loss", size=(400,200), legend=false);
for i=1:n_exp
    a = sqrt(noise_level_arr[i] / 10^-0.5)
    plot!(1:n_epoch+1, sqrt.(losses_y_exps[i, :] ./ datasize ./ 3), yscale=:log10, line=(1, :solid), alpha=a, color=:black)
end
h2 = plot(xlabel="Epochs", ylabel="k loss", size=(400,200), legend=false);
for i=1:n_exp
    a = sqrt(noise_level_arr[i] / 10^-0.5)
    plot!(1:n_epoch+1, sqrt.(losses_p_exps[i, :] ./ 3), yscale=:log10, line=(1, :solid), alpha=a, color=:black)
end
h = plot(h1, h2, layout=(2,1), size=(400,400), framestyle=:box)
Plots.savefig(h, "figures/Robertson_noises_losses.svg")

# new accuracy plot
fig, axs = PyPlot.subplots(2,2, figsize=(8, 4))
axs[1,1].set_yscale("log")
axs[1,1].set_ylabel("\$L_y(\\theta)\$")
axs[2,1].set_yscale("log")
axs[2,1].set_xlabel("Epochs")
axs[2,1].set_ylabel("\$L_k(\\theta)\$")
for i=1:n_exp
    a = sqrt(noise_level_arr[i] / 10^-0.5)
    axs[1,1].plot(1:n_epoch+1, sqrt.(losses_y_exps[i, :] ./ datasize ./ 3), lw=1, alpha=a, color="k")
    axs[2,1].plot(1:n_epoch+1, sqrt.(losses_p_exps[i, :] ./ 3), lw=1, alpha=a, color="k")
end
axs[1,1].set_xlim([0,300])
axs[2,1].set_xlim([0,300])
axs[1,2].plot(noise_level_arr, sqrt.(losses_y_exps[:,end].* datasize ./ 3), "k.")
axs[1,2].plot(noise_level_arr, noise_level_arr, "r-", lw=0.5)
axs[1,2].text(1e-2, 3e-2, "\$y=x\$", color="r")
axs[1,2].set_xscale("log")
axs[1,2].set_yscale("log")
axs[1,2].set_ylabel("\$y_{err}\$")
axs[2,2].plot(noise_level_arr, sqrt.(losses_p_exps[:,end]./3), "k.")
axs[2,2].plot(noise_level_arr, noise_level_arr, "r-", lw=0.5)
axs[2,2].text(1e-2, 3e-2, "\$y=x\$", color="r")
axs[2,2].set_xscale("log")
axs[2,2].set_yscale("log")
axs[2,2].set_ylabel("\$k_{err}\$")
axs[2,2].set_xlabel("Noise Level")
fig.subplots_adjust(left=0.1, right=0.98, hspace=0.3, top=0.98, bottom=0.15, wspace=0.25)
fig.savefig(string(@sprintf("figures/%s_noises_loss&accuracy.svg", casename)))
display(fig)

Plots.pyplot()
Plots.resetfontsizes()
fm, fs = "Computer Modern", 11
fnt = Plots.font(fs, fm)
default(titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt, legendtitlefontsize=fs)

# define gradients of ODE function f
dfdu(u,p,t) = Zygote.jacobian(x->f(x,p,t), u)[1];
dfdp(u,p,t) = Zygote.jacobian(x->f(u,x,t), p)[1];
dfdt(u,p,t) = Zygote.jacobian(x->f(u,p,x), t)[1];

# augmented ODE for adjoint state
function aug_ode!(du, u, p, t)
    Np = length(p)
    Nu = (length(u) - Np - 1) >> 1
    y = @views(u[1:Nu])
    au = @views(u[Nu+1:Nu+Nu])
    aθ = @views(u[Nu+Nu+1:Nu+Nu+Np])
    at = @views(u[end])
    dydt = f(y,p,t)
    daudt = - au' * dfdu(y,p,t)
    daθdt = - au' * dfdp(y,p,t)
    datdt = - au' * dfdt(y,p,t)
    du .= vcat(dydt, vec(daudt), vec(daθdt), datdt)
end

# backward adjoint
function grad_adjoint(p_pred, y_pred, y_true; doplot=false)
    Nu, Np = length(y_pred[:,1]), length(p_pred)
    y_t = y_pred[:,end]
    au_t = zeros(Nu)
    aθ_t = zeros(Np)
    at_t = zeros(1)
    aug_t = vcat(y_t, au_t, aθ_t, at_t);
    aug_prob = ODEProblem(aug_ode!, aug_t, reverse(tspan), p_pred);
    if doplot==true
        au_arr = Vector{Matrix{Float64}}();
        u_arr = Vector{Matrix{Float64}}();
        t_arr = Vector{Array{Float64}}();
    end
    denom = scale .* scale; #* length(y_true);
    for i in range(length(tsteps),2,step=-1)
        y_t = y_pred[:,i]
        au_t += 2 * (y_t .- y_true[:,i]) ./ denom;
        aug_t = vcat(y_t, au_t, aθ_t, at_t);

        tspan_i = (tsteps[i],tsteps[i-1]);
        aug_sol = solve(aug_prob, solver, u0=aug_t, tspan=tspan_i, p=p_pred)
        aug_u = Array(aug_sol)

        y_t = aug_u[1:Nu,end]
        au_t = aug_u[Nu+1:Nu+Nu,end]
        aθ_t = aug_u[Nu+Nu+1:Nu+Nu+Np,end]
        at_t = aug_u[end,end]

        if doplot==true
            push!(au_arr, aug_u[Nu+1:Nu+Nu,:])
            push!(u_arr, aug_u[1:Nu,:])
            push!(t_arr, aug_sol.t)
        end
    end

    if doplot==true
        # plot state u
        h1 = plot(xformatter=_->"", grid=false, showaxis=:xy, palette=palette(:darktest, length(weights)))
        plot!(tsteps, y_noise', line=:scatter, m=:square, msw=0, ms=4, label="")
        for (i,ti) in enumerate(t_arr)
            plot!(ti, u_arr[i]', label="", xscale=xscale)
        end
        plot!([t_arr[1][2],t_arr[1][1]], [u_arr[1][1,2],u_arr[1][1,1]], arrow=(:closed, 0.6), label="", xscale=xscale)
        plot!([t_arr[1][2],t_arr[1][1]], [u_arr[1][2,2],u_arr[1][2,1]], arrow=(:closed, 0.6), label="", xscale=xscale)
        ylabel!("u(t)")
        annotate!(5, 7., text("State", :right, fs))
        annotate!(0.5, 2., text(L"$u_1$", :right, palette(:darktest, length(weights))[1], fs))
        annotate!(0.5, 0., text(L"$u_2$", :right, palette(:darktest, length(weights))[2], fs))

        # plot adjoint state a_u
        h2 = plot(xlabel="t [s]", grid=false, showaxis=:xy, palette=palette(:darktest, length(weights)))
        for (i,ti) in enumerate(t_arr)
            plot!(ti, au_arr[i]', label="", xscale=xscale)
            plot!([ti[end-1], ti[end]], [au_arr[i][1,end-1], au_arr[i][1,end]],
                    arrow=(:closed, 0.6), label="", xscale=xscale)
            plot!([ti[end-1], ti[end]], [au_arr[i][2,end-1], au_arr[i][2,end]],
                    arrow=(:closed, 0.6), label="", xscale=xscale)
            if i>1
                t_intp = [t_arr[i-1][end], t_arr[i][1]]
                au1_intp = [au_arr[i-1][1,end], au_arr[i][1,1]]
                au2_intp = [au_arr[i-1][2,end], au_arr[i][2,1]]
                plot!(t_intp, au1_intp, line=:dot, label="")#, arrow=(:open, 0.6))
                plot!(t_intp, au2_intp, line=:dot, label="")#, arrow=(:open, 0.6))
            end
        end
        ylabel!(L"$\lambda$(t)")
        annotate!(5, 0.33, text("Adjoint State", :right, fs))
        annotate!(4.5, 0.1, text(L"$\lambda_1$", :right, palette(:darktest, length(weights))[1], fs))
        annotate!(4.5,-0.15, text(L"$\lambda_2$", :right, palette(:darktest, length(weights))[2], fs))
        h = plot(h1, h2, layout=(2,1), framestyle=:box, size=(400, 350))
        display(h)
        return aθ_t, h
    end

    return aθ_t
end

# initialize
noise_level = 5e-1;
rng = MersenneTwister(77777);
y_noise = y_true + noise_level .* (rand(rng, length(y0), length(tsteps)).-0.5) .* scale;

# show adjoint state
grad_adj, h = grad_adjoint(p_true, y_true, y_noise, doplot=true);
Plots.savefig(h, string(@sprintf("figures/%s_noise=%.0e_AdjointState_trained.svg",
                casename, noise_level)))

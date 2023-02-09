using Dierckx

# define gradients of ODE function f
dfdu(u,p,t) = Zygote.jacobian(x->f(x,p,t), u)[1];
dfdp(u,p,t) = Zygote.jacobian(x->f(u,x,t), p)[1];
dfdt(u,p,t) = Zygote.jacobian(x->f(u,p,x), t)[1];

# adjoint ode with interpolated u
# solve au, aθ, at
function aug_ode!(du, u, p, t)
    Np = length(p)
    Nu = length(u) - Np - 1
    y = [interp_u(t) for interp_u in interp_u_s]
    au = @views(u[1:Nu])
    aθ = @views(u[Nu+1:Nu+Np])
    at = @views(u[end])
    daudt = - au' * dfdu(y,p,t)
    daθdt = - au' * dfdp(y,p,t)
    datdt = - au' * dfdt(y,p,t)
    du .= vcat(vec(daudt), vec(daθdt), datdt)
end

# interpolating adjoint
function grad_adjoint(p_pred, y_pred, y_true; doplot=false)
    Nu, Np = length(y_pred[:,1]), length(p_pred)
    interp_u_s = [Spline1D(tsteps, y_pred[i,:]) for i in 1:Nu]
    au_t = zeros(Nu)
    aθ_t = zeros(Np)
    at_t = zeros(1)
    aug_t = vcat(au_t, aθ_t, at_t);
    aug_prob = ODEProblem(aug_ode!, aug_t, reverse(tspan), p_pred);
    if doplot==true
        au_arr = Vector{Matrix{Float64}}();
        t_arr = Vector{Array{Float64}}();
    end
    denom = scale .* scale * length(y_true);
    for i in range(length(tsteps),2,step=-1)
        tspan_i = (tsteps[i],tsteps[i-1]);
        aug_t[1:Nu] += 2*(y_pred[:,i] .- y_true[:,i]) ./ denom;
        aug_sol = solve(aug_prob, solver, u0=aug_t, tspan=tspan_i, p=p_pred);
        aug_u = Array(aug_sol);
        aug_t = aug_u[:,end];
        if doplot==true
            push!(au_arr, aug_u[Nu+1:Nu+Nu,:])
            push!(t_arr, aug_sol.t)
        end
    end
    if doplot==true
        h = plot(title="Adjoint State", xlabel="t [s]",
                palette=palette(:darktest, length(weights)))
        for (i,ti) in enumerate(t_arr)
            plot!(ti, au_arr[i]', label="", xscale=xscale)
        end
        display(h)
        return aug_t[Nu+1:Nu+Np], h
    end
    return aug_t[Nu+1:Nu+Np]
end

interp_u_s = [Spline1D(tsteps, y_true[i,:]) for i in 1:length(y_true[:,1])]

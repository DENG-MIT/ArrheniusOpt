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
        t_arr = Vector{Array{Float64}}();
    end
    for i in range(length(tsteps),2,step=-1)
        y_t = y_pred[:,i]
        au_t += 2*(y_t .- y_true[:,i]) ./ scale;
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
            push!(t_arr, aug_sol.t)
        end
    end

    if doplot==true
        h = plot(title="Adjoint State", xlabel="t [s]")
        for (i,ti) in enumerate(t_arr)
            plot!(ti, au_arr[i]', label="")
        end
        display(h)
        return aθ_t, h
    end

    return aθ_t
end

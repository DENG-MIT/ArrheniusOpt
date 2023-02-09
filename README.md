## ArrheniusOpt

This repo is used to validate the ability of kinetics parameter optimization using `Arrhenius.jl` and `DiffEqFlux`.

#### 1. Adjoint sensitivity

By defining adjoint state as:

$$
\boldsymbol a(t) := -\frac{\partial L}{\partial \boldsymbol u(t)}
$$

one can get the evolution ODE for the adjoint state:

$$
\frac{d\boldsymbol a(t)}{dt} = -\boldsymbol a(t) \frac{\partial f(\boldsymbol u(t),\boldsymbol \theta,t)}{\partial \boldsymbol u}
$$

Similarly, denoting the adjoint states for parameters and times:

$$
\boldsymbol a_\theta(t) := -\frac{\partial L}{\partial \boldsymbol \theta(t)} \\ \boldsymbol a_t(t) := \frac{\partial L}{\partial t(t)}
$$

And finally one can get the parameters' gradients against loss function:

$$
\frac{\partial L}{\partial \boldsymbol \theta} = \int_{t_1}^{t_0} \boldsymbol a(t) \frac{\partial f(\boldsymbol u(t),\boldsymbol \theta,t)}{\partial \boldsymbol \theta} dt
$$

The adjoint sensitivity method is implemented in folder `example/Robertson` with three different  variants:

+ `backsolveAdjoint.jl`: Direct integration of the augmented adjoint states

  $$
  \boldsymbol a_{aug}(t) = [\boldsymbol u(t), \boldsymbol a_u(t), \boldsymbol a_\theta(t), \boldsymbol a_t(t)]
  $$

+ `interpolatingAdjoint.jl`: Backward solving $\boldsymbol u(t)$ is hard for stiff ODEs, so this method interpolates the forward computed $\boldsymbol u(t)$ in the backward process.

+ `discreteAdjoint.jl`: Interpolations on high-dimensional $t\to\boldsymbol u(t)$ is time consuming. This method  use the discrete $\boldsymbol u(t_{i+1})$ as a simplification of $\boldsymbol u(t), t\in[t_{i+1},t_{i}]$ when backward solving the augmented states.
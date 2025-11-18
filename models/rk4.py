import torch

# differentiable RK4

def rk4_step(x, I, dt, f, *f_args):
    # x = rk4_step(x, I[:, k], self.dt, f_state_2rc, R1, C1, R2, C2)
    k1 = f(x, I, *f_args)
    k2 = f(x + 0.5*dt*k1, I, *f_args)
    k3 = f(x + 0.5*dt*k2, I, *f_args)
    k4 = f(x + dt*k3,     I, *f_args)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
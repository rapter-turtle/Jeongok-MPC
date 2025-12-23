import numpy as np
import matplotlib.pyplot as plt
from casadi import SX, DM, Function, exp, tanh, log, cos

# ==========================================================
# BLACK-BOX GP SURGE MODEL BUILDER
# ==========================================================

def build_gp_surge_model(npz_path):
    """
    Returns:
        gp_surge_model(u, T, delta) -> SX
    """

    P = np.load(npz_path)

    # -------- scalers --------
    X_mean = float(P["X_mean"])
    X_std  = float(P["X_std"])
    Y_mean = float(P["Y_mean"])
    Y_std  = float(P["Y_std"])

    # -------- GP params --------
    Z   = DM(P["Z"].reshape(-1))
    tmp = DM(P["tmp"].reshape(-1))
    l   = float(P["lengthscale"])
    os  = float(P["outputscale"])
    M   = int(Z.numel())

    # -------- alpha params --------
    raw_min   = float(P["raw_min"])
    raw_range = float(P["raw_range"])
    a3        = float(P["a3"])
    a2_raw    = float(P["a2_raw"])

    # -------- helper --------
    def sigmoid(x):
        return 1.0 / (1.0 + exp(-x))

    def softplus(x):
        return log(1.0 + exp(x))

    # -------- alpha(u) --------
    def alpha_u(u):
        alpha_min = 0.01 + 0.09 * sigmoid(raw_min)
        alpha_max = alpha_min + 0.001 + 0.1 * sigmoid(raw_range)
        a2 = softplus(a2_raw)
        sigma = tanh(a2 * (u - a3))
        return alpha_min + (alpha_max - alpha_min) * 0.5 * (1.0 + sigma)

    # -------- f(u) --------
    def f_u(u):
        un = (u - X_mean) / (X_std + 1e-12)
        s = 0
        for i in range(M):
            d = un - Z[i]
            k = os * exp(-0.5 * (d * d) / (l * l))
            s = s + k * tmp[i]
        return s * Y_std + Y_mean

    # -------- FINAL BLACK BOX --------
    def gp_surge_model(u, T, delta):
        return f_u(u) + alpha_u(u) * T * cos(2 * delta/500.0)

    return gp_surge_model


# ==========================================================
# PLOT f(u), alpha(u) USING ONLY BLACK-BOX MODEL
# ==========================================================

def plot_f_alpha_blackbox(npz_path, u_min=0.0, u_max=12.0, N=400):

    gp_surge_model = build_gp_surge_model(npz_path)

    u = SX.sym('u')
    T = SX.sym('T')
    delta = SX.sym('delta')

    # f(u): T = 0
    f_expr = gp_surge_model(u, 0.0, 0.0)

    # alpha(u): linear coefficient of T
    a_expr = gp_surge_model(u, 1.0, 0.0) - gp_surge_model(u, 0.0, 0.0)

    f_fun = Function("f_fun", [u], [f_expr])
    a_fun = Function("a_fun", [u], [a_expr])

    u_grid = np.linspace(u_min, u_max, N)

    f_vals = np.zeros_like(u_grid)
    a_vals = np.zeros_like(u_grid)

    for i, ui in enumerate(u_grid):
        f_vals[i] = float(f_fun(ui))
        a_vals[i] = float(a_fun(ui))

    # ---- f(u) ----
    plt.figure(figsize=(12,4))
    plt.plot(u_grid, f_vals, linewidth=3)
    plt.xlabel("u [m/s]")
    plt.ylabel("f(u)")
    plt.title("f(u) from BLACK-BOX GP surge model")
    plt.grid()
    plt.show()

    # ---- alpha(u) ----
    plt.figure(figsize=(10,4))
    plt.plot(u_grid, a_vals, linewidth=3)
    plt.xlabel("u [m/s]")
    plt.ylabel("alpha(u)")
    plt.title("alpha(u) from BLACK-BOX GP surge model")
    plt.grid()
    plt.show()


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    NPZ_PATH = "/home/user/aura_ws/src/aura_mpc/GP_MPC/surge_model/surge_acados.npz"
    plot_f_alpha_blackbox(NPZ_PATH)

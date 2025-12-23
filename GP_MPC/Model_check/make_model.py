#!/usr/bin/env python3
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

from casadi import SX, DM, Function, exp, tanh, log

# ==========================================================
# USER PATH
# ==========================================================
MODEL_DIR = "/home/user/aura_ws/src/aura_mpc/GP_MPC/surge_model"
OUT_NPZ   = MODEL_DIR + "/surge_acados.npz"

GP_PATH    = MODEL_DIR + "/gp.pth"
LIK_PATH   = MODEL_DIR + "/likelihood.pth"
ALPHA_PATH = MODEL_DIR + "/alpha.pth"
Z_PATH     = MODEL_DIR + "/inducing_points.pth"
SCALER     = np.load(MODEL_DIR + "/scaler.npz")

DEVICE = "cpu"

X_mean = float(SCALER["X_mean"])
X_std  = float(SCALER["X_std"])
Y_mean = float(SCALER["Y_mean"])
Y_std  = float(SCALER["Y_std"])

# ==========================================================
# 1) CasADi GP + alpha builder (acados-consistent)
# ==========================================================
def make_surge_gp_alpha_from_npz(npz_path):

    P = np.load(npz_path)

    # scalers
    X_mean = float(P["X_mean"])
    X_std  = float(P["X_std"])
    Y_mean = float(P["Y_mean"])
    Y_std  = float(P["Y_std"])

    # GP params
    Z = DM(P["Z"].reshape(-1))
    tmp = DM(P["tmp"].reshape(-1))
    l = float(P["lengthscale"])
    os = float(P["outputscale"])
    M = int(Z.numel())

    # alpha params
    raw_min   = float(P["raw_min"])
    raw_range = float(P["raw_range"])
    a3        = float(P["a3"])
    a2_raw    = float(P["a2_raw"])

    def sigmoid(x):
        return 1.0 / (1.0 + exp(-x))

    def softplus(x):
        return log(1.0 + exp(x))

    # alpha(u)
    def alpha_u(u):
        alpha_min = 0.01 + 0.09 * sigmoid(raw_min)
        alpha_max = alpha_min + 0.001 + 0.1 * sigmoid(raw_range)
        a2 = softplus(a2_raw)
        sigma = tanh(a2 * (u - a3))
        return alpha_min + (alpha_max - alpha_min) * 0.5 * (1.0 + sigma)

    # f(u)
    def f_u(u):
        un = (u - X_mean) / (X_std + 1e-12)
        s = 0
        for i in range(M):
            d = un - Z[i]
            k = os * exp(-0.5 * (d * d) / (l * l))
            s = s + k * tmp[i]
        return s * Y_std + Y_mean

    return f_u, alpha_u


# ==========================================================
# 2) Torch models
# ==========================================================
class AlphaLearn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_min = torch.nn.Parameter(torch.tensor(0.0))
        self.raw_range = torch.nn.Parameter(torch.tensor(0.0))
        self.a3 = torch.nn.Parameter(torch.tensor(5.0))
        self.a2_raw = torch.nn.Parameter(torch.tensor(1.0))
        self.sigmoid = torch.nn.Sigmoid()
        self.softplus = torch.nn.Softplus()

    def forward(self, u):
        alpha_min = 0.01 + 0.09 * self.sigmoid(self.raw_min)
        alpha_max = alpha_min + 0.001 + 0.1 * self.sigmoid(self.raw_range)
        a2 = self.softplus(self.a2_raw)
        sigma = torch.tanh(a2 * (u - self.a3))
        return alpha_min + (alpha_max - alpha_min) * 0.5 * (1 + sigma)


class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, Z):
        qdist = gpytorch.variational.CholeskyVariationalDistribution(Z.size(0))
        qstr = gpytorch.variational.VariationalStrategy(
            self, Z, qdist, learn_inducing_locations=True
        )
        super().__init__(qstr)
        self.mean = gpytorch.means.ZeroMean()
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean(x), self.kernel(x))


# ==========================================================
# 3) Export torch → npz (ONE TIME)
# ==========================================================
Z = torch.load(Z_PATH, map_location=DEVICE)
gp = SVGPModel(Z).to(DEVICE)
lik = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
alpha = AlphaLearn().to(DEVICE)

gp.load_state_dict(torch.load(GP_PATH, map_location=DEVICE))
lik.load_state_dict(torch.load(LIK_PATH, map_location=DEVICE))
alpha.load_state_dict(torch.load(ALPHA_PATH, map_location=DEVICE))

gp.eval(); lik.eval(); alpha.eval()

lengthscale = float(gp.kernel.base_kernel.lengthscale.detach().numpy().reshape(-1)[0])
outputscale = float(gp.kernel.outputscale.detach().numpy().reshape(-1)[0])
whitened = bool(getattr(gp.variational_strategy, "whitened", True))

Z_np = gp.variational_strategy.inducing_points.detach().numpy().reshape(-1)
m_np = gp.variational_strategy.variational_distribution.mean.detach().numpy().reshape(-1)

def rbfK(z1, z2, l, os):
    z1 = z1[:,None]; z2 = z2[None,:]
    return os*np.exp(-0.5*(z1-z2)**2/(l*l))

Kzz = rbfK(Z_np, Z_np, lengthscale, outputscale) + 1e-6*np.eye(len(Z_np))
Lzz = np.linalg.cholesky(Kzz)

if whitened:
    tmp = np.linalg.solve(Lzz.T, m_np)
else:
    v = np.linalg.solve(Lzz, m_np)
    tmp = np.linalg.solve(Lzz.T, v)

a_sd = alpha.state_dict()
raw_min   = float(a_sd["raw_min"])
raw_range = float(a_sd["raw_range"])
a3        = float(a_sd["a3"])
a2_raw    = float(a_sd["a2_raw"])

np.savez(
    OUT_NPZ,
    X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std,
    Z=Z_np, tmp=tmp,
    lengthscale=lengthscale, outputscale=outputscale,
    whitened=int(whitened),
    raw_min=raw_min, raw_range=raw_range, a3=a3, a2_raw=a2_raw
)

print("✔ Saved:", OUT_NPZ)

# ==========================================================
# 4) Plot f(u), alpha(u) (acados-consistent)
# ==========================================================
def plot_f_alpha(npz_path, u_min=0.0, u_max=12.0, N=400):

    f_u, alpha_u = make_surge_gp_alpha_from_npz(npz_path)

    u = SX.sym('u')
    f_fun = Function("f_fun", [u], [f_u(u)])
    a_fun = Function("a_fun", [u], [alpha_u(u)])

    u_grid = np.linspace(u_min, u_max, N)
    f_vals = np.zeros_like(u_grid)
    a_vals = np.zeros_like(u_grid)

    for i, ui in enumerate(u_grid):
        f_vals[i] = float(f_fun(ui))
        a_vals[i] = float(a_fun(ui))

    plt.figure(figsize=(12,4))
    plt.plot(u_grid, f_vals, linewidth=3)
    plt.xlabel("u [m/s]")
    plt.ylabel("f(u)")
    plt.title("GP residual f(u) used in acados")
    plt.grid()
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(u_grid, a_vals, linewidth=3)
    plt.xlabel("u [m/s]")
    plt.ylabel("alpha(u)")
    plt.title("Learned alpha(u)")
    plt.grid()
    plt.show()


# ==========================================================
# 5) Run
# ==========================================================
if __name__ == "__main__":
    plot_f_alpha(OUT_NPZ)

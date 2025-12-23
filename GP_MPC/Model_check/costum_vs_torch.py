

#!/usr/bin/env python3
import os
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

# ==========================================================
# 0) Set model directory HERE
# ==========================================================
MODEL_DIR = "/home/user/aura_ws/src/aura_mpc/GP_MPC/surge_model"   # <-- 여기만 네 폴더로 바꾸면 됨

GP_PATH    = os.path.join(MODEL_DIR, "gp.pth")
LIK_PATH   = os.path.join(MODEL_DIR, "likelihood.pth")
ALPHA_PATH = os.path.join(MODEL_DIR, "alpha.pth")
Z_PATH     = os.path.join(MODEL_DIR, "inducing_points.pth")
SCALER_PATH= os.path.join(MODEL_DIR, "scaler.npz")

S = np.load(SCALER_PATH)
X_mean = float(S["X_mean"]); X_std = float(S["X_std"])
Y_mean = float(S["Y_mean"]); Y_std = float(S["Y_std"])

# ==========================================================
# 1) Alpha(u)
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

alpha = AlphaLearn().to(DEVICE)
alpha.load_state_dict(torch.load(ALPHA_PATH, map_location=DEVICE))
alpha.eval()

# ==========================================================
# 2) GPtorch model (reference)
# ==========================================================
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

Z = torch.load(Z_PATH, map_location=DEVICE)
gp = SVGPModel(Z).to(DEVICE)
lik = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)

gp.load_state_dict(torch.load(GP_PATH, map_location=DEVICE))
lik.load_state_dict(torch.load(LIK_PATH, map_location=DEVICE))
gp.eval(); lik.eval()

# ==========================================================
# 3) NumPy SVGP mean (acados-friendly)
#    Implements: mu(x) = K_xZ * L_zz^{-T} * m   (whitened)
# ==========================================================
def rbf_kernel_numpy(X1, X2, lengthscale, outputscale):
    # X1: (N,1), X2:(M,1)
    sq = (X1 - X2.T) ** 2
    return outputscale * np.exp(-0.5 * sq / (lengthscale**2))

class NumpySVGPMean:
    """
    Mean-only SVGP prediction matching gpytorch (for whitened strategy).
    O(M^2) precompute, O(N*M) predict.
    """
    def __init__(self, Z, m, lengthscale, outputscale, jitter=1e-6, whitened=True):
        self.Z = Z.reshape(-1,1).astype(np.float64)     # (M,1)
        self.m = m.reshape(-1).astype(np.float64)       # (M,)
        self.l = float(lengthscale)
        self.os = float(outputscale)
        self.whitened = bool(whitened)

        Kzz = rbf_kernel_numpy(self.Z, self.Z, self.l, self.os)
        Kzz = Kzz + jitter * np.eye(Kzz.shape[0])
        self.Lzz = np.linalg.cholesky(Kzz)              # Kzz = L L^T

        if self.whitened:
            # tmp = L^{-T} m  (solve L^T tmp = m)
            self.tmp = np.linalg.solve(self.Lzz.T, self.m)  # (M,)
        else:
            # tmp = Kzz^{-1} m (solve via cholesky)
            v = np.linalg.solve(self.Lzz, self.m)
            self.tmp = np.linalg.solve(self.Lzz.T, v)

    def predict_mean(self, X):
        X = X.reshape(-1,1).astype(np.float64)          # (N,1)
        KxZ = rbf_kernel_numpy(X, self.Z, self.l, self.os)  # (N,M)
        mu = KxZ @ self.tmp                              # (N,)
        return mu

# ---- extract variational mean m and whitened flag ----
# gpytorch stores variational mean at:
m_t = gp.variational_strategy.variational_distribution.mean.detach().cpu().numpy()

# whitened flag: gp.variational_strategy.whitened (usually True)
whitened_flag = bool(getattr(gp.variational_strategy, "whitened", True))

# kernel hyperparameters
lengthscale = gp.kernel.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)[0]
outputscale = gp.kernel.outputscale.detach().cpu().numpy().reshape(-1)[0]

Z_np = gp.variational_strategy.inducing_points.detach().cpu().numpy().reshape(-1,1)

np_svgp = NumpySVGPMean(
    Z=Z_np,
    m=m_t,
    lengthscale=lengthscale,
    outputscale=outputscale,
    whitened=whitened_flag
)

# ==========================================================
# 4) Compare on a grid: f(u) and alpha(u)
# ==========================================================
u_grid = np.linspace(0, 12, 400).astype(np.float32)
u_norm = (u_grid - X_mean) / (X_std + 1e-12)

U_t = torch.tensor(u_norm[:,None], dtype=torch.float32, device=DEVICE)

# gpytorch mean (normalized y-space)
f_torch_norm = lik(gp(U_t)).mean.detach().cpu().numpy().reshape(-1)

# numpy svgp mean (normalized y-space)
f_numpy_norm = np_svgp.predict_mean(u_norm)

# denormalize to real residual space
f_torch = f_torch_norm * Y_std + Y_mean
f_numpy = f_numpy_norm * Y_std + Y_mean

# alpha(u)
alpha_u = alpha(torch.tensor(u_grid, device=DEVICE)).detach().cpu().numpy().reshape(-1)

# error metrics
abs_err = np.abs(f_torch - f_numpy)
print("=== f(u) torch vs numpy(SVGP-formula) ===")
print(f"max|err|  : {abs_err.max():.6e}")
print(f"mean|err| : {abs_err.mean():.6e}")

# ==========================================================
# 5) Plots
# ==========================================================
plt.figure(figsize=(12,4))
plt.plot(u_grid, f_torch, linewidth=3, label="GPtorch f(u)")
plt.plot(u_grid, f_numpy, "--", linewidth=3, label="NumPy SVGP f(u)")
plt.xlabel("u"); plt.ylabel("f(u)")
plt.grid(); plt.legend()
plt.title("f(u) comparison (should overlap)")
plt.show()

plt.figure(figsize=(12,3))
plt.plot(u_grid, abs_err, linewidth=2)
plt.xlabel("u"); plt.ylabel("|f_torch - f_numpy|")
plt.grid()
plt.title("Absolute error")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(u_grid, alpha_u, linewidth=3)
plt.xlabel("u"); plt.ylabel("alpha(u)")
plt.grid()
plt.title("alpha(u)")
plt.show()

# ==========================================================
# 6) acados-friendly callable (mean-only)
# ==========================================================
def f_u_numpy(u_real):
    un = (u_real - X_mean) / (X_std + 1e-12)
    mu_norm = np_svgp.predict_mean(np.array([un], dtype=np.float64))[0]
    return float(mu_norm * Y_std + Y_mean)

print("✔ ready: f_u_numpy(u) gives compact mean-only f(u) for MPC/acados")

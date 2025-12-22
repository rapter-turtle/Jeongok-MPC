import numpy as np

# ==========================================================
# NumPy GP Surge Model (DOB / ROS)
# ==========================================================

class GPSurgeModelNumpy:
    def __init__(self, npz_path):
        P = np.load(npz_path)

        # scalers
        self.X_mean = float(P["X_mean"])
        self.X_std  = float(P["X_std"])
        self.Y_mean = float(P["Y_mean"])
        self.Y_std  = float(P["Y_std"])

        # GP params
        self.Z   = P["Z"].reshape(-1)      # (M,)
        self.tmp = P["tmp"].reshape(-1)    # (M,)
        self.l   = float(P["lengthscale"])
        self.os  = float(P["outputscale"])
        self.M   = len(self.Z)

        # alpha params
        self.raw_min   = float(P["raw_min"])
        self.raw_range = float(P["raw_range"])
        self.a3        = float(P["a3"])
        self.a2_raw    = float(P["a2_raw"])

    # -----------------------------
    # helpers
    # -----------------------------
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def softplus(x):
        return np.log1p(np.exp(x))

    # -----------------------------
    # alpha(u)
    # -----------------------------
    def alpha(self, u):
        alpha_min = 0.01 + 0.09 * self.sigmoid(self.raw_min)
        alpha_max = alpha_min + 0.001 + 0.1 * self.sigmoid(self.raw_range)
        a2 = self.softplus(self.a2_raw)
        sigma = np.tanh(a2 * (u - self.a3))
        return alpha_min + (alpha_max - alpha_min) * 0.5 * (1.0 + sigma)

    # -----------------------------
    # f(u)
    # -----------------------------
    def f(self, u):
        un = (u - self.X_mean) / (self.X_std + 1e-12)
        s = 0.0
        for i in range(self.M):
            d = un - self.Z[i]
            k = self.os * np.exp(-0.5 * d * d / (self.l * self.l))
            s += k * self.tmp[i]
        return s * self.Y_std + self.Y_mean

    # -----------------------------
    # FINAL surge model
    # -----------------------------
    def __call__(self, u, T, delta):
        return self.f(u) + self.alpha(u) * T

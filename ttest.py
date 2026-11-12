#%%
import time
import torch
import torch.nn as nn
from Models.wesn import WESN
import torch._dynamo

torch._dynamo.config.suppress_errors = True

# 这里假设 ESNCell 和 WESN 已经在同一文件中定义好
# 或者你可以 from 其他模块 import
# from wesn import ESNCell, WESN


def create_model(hidden_dim, win_len, output_dim, device, dtype):
    """Helper to create a WESN model with fixed hyperparameters."""
    model = WESN(
        hidden_dim=hidden_dim,
        connectivity=0.1,
        leaky=1.0,
        spectral_radius=0.9,
        use_norm2=False,
        use_bias=True,
        activation="tanh",
        win_len=win_len,
        W_in_initializer="xavier_uniform",
        W_h_initializer="xavier_uniform",
        bias_initializer="zeros",
        output_dim=output_dim,
    )
    model.to(device=device, dtype=dtype)
    return model


def benchmark_one(model, x, n_warmup=10, n_iters=50, do_backward=False, name=""):
    """
    Benchmark forward (and optional backward) runtime of a model on given input x.

    Args:
        model:   nn.Module (WESN or compiled WESN)
        x:       [B, T, C] tensor on correct device/dtype
        n_warmup: number of warmup iterations (not timed)
        n_iters:  number of iterations for timing
        do_backward: whether to include backward pass in timing
        name:    label for print
    """
    device = x.device
    use_cuda = device.type == "cuda"

    model.train() if do_backward else model.eval()

    # Warmup
    for _ in range(n_warmup):
        y = model(x)
        if do_backward:
            loss = y.mean()
            loss.backward()
            model.zero_grad(set_to_none=True)

    if use_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(n_iters):
        y = model(x)
        if do_backward:
            loss = y.mean()
            loss.backward()
            model.zero_grad(set_to_none=True)

    if use_cuda:
        torch.cuda.synchronize()

    end = time.perf_counter()
    avg_time_ms = (end - start) * 1000.0 / n_iters
    mode = "fwd+bwd" if do_backward else "fwd-only"
    print(f"[{name:<15s}] {mode:8s}  avg = {avg_time_ms:6.3f} ms / iter")


def main():
    # ----------------- 基本配置 -----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # 你可以根据实际应用调整这些参数
    B = 32      # batch size
    T = 64      # time steps
    C = 16      # input dim
    H = 64      # ESN hidden_dim
    out_dim = 4 # output_dim
    win_len = 3 # window length: use [x_t, x_{t-1}, x_{t-2}]

    print(f"Device: {device}, dtype: {dtype}")
    print(f"B={B}, T={T}, C={C}, H={H}, win_len={win_len}, out_dim={out_dim}")

    # 随机输入
    x = torch.randn(B, T, C, device=device, dtype=dtype)

    # ----------------- 创建模型（eager 版本） -----------------
    model_eager = create_model(H, win_len, out_dim, device, dtype)

    # ----------------- 创建 compiled 版本（如果可用） -----------------
    model_compiled = None
    has_compile = hasattr(torch, "compile")
    if has_compile:
        # torch.compile 仅在 PyTorch >= 2.0 可用
        model_compiled = torch.compile(create_model(H, win_len, out_dim, device, dtype))
    else:
        print("torch.compile not available (PyTorch < 2.0).")

    # ----------------- 只测 forward -----------------
    print("\n=== Forward-only benchmark ===")
    benchmark_one(model_eager, x, n_warmup=10, n_iters=50, do_backward=False, name="eager")

    if model_compiled is not None:
        benchmark_one(model_compiled, x, n_warmup=10, n_iters=50, do_backward=False, name="compiled")

    # ----------------- forward + backward -----------------
    print("\n=== Forward + Backward benchmark ===")
    benchmark_one(model_eager, x, n_warmup=10, n_iters=30, do_backward=True, name="eager")

    if model_compiled is not None:
        benchmark_one(model_compiled, x, n_warmup=10, n_iters=30, do_backward=True, name="compiled")


if __name__ == "__main__":
    main()


#%% 3D Gaussian function
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters for a 3D Gaussian N(mu, Sigma) ---
# mu = np.array([0.5, -0.2, 0.3])
# Sigma = np.array([
#     [1.0,  0.6, 0.2],
#     [0.6,  1.5, 0.3],
#     [0.2,  0.3, 0.7]
# ], dtype=float)

mu = np.array([0, 0, 0])
Sigma = np.array([
    [1.0,  0, 0],
    [0,  1, 0],
    [0,  0, 1]
], dtype=float)

# Ensure Sigma is symmetric positive definite
Sigma = (Sigma + Sigma.T) / 2
eigvals, eigvecs = np.linalg.eigh(Sigma)

# Level set: (x-mu)^T Sigma^{-1} (x-mu) = k corresponds to constant density
# For a visual "1-sigma" ellipsoid, k=1. For a more visible ellipsoid, k=3 is common.
k = 3.0

# Create a unit sphere parameterization
u = np.linspace(0, 2*np.pi, 80)
v = np.linspace(0, np.pi, 40)
uu, vv = np.meshgrid(u, v)

xs = np.cos(uu) * np.sin(vv)
ys = np.sin(uu) * np.sin(vv)
zs = np.cos(vv)

sphere = np.stack([xs, ys, zs], axis=-1)  # shape (V,U,3)

# Map sphere to ellipsoid: x = mu + eigvecs * diag(sqrt(k*eigvals)) * sphere_point
radii = np.sqrt(k * eigvals)
A = eigvecs @ np.diag(radii)

ellipsoid = sphere @ A.T + mu  # broadcasted

Xe = ellipsoid[..., 0]
Ye = ellipsoid[..., 1]
Ze = ellipsoid[..., 2]

# Sample points from the Gaussian for intuition
n_samples = 1500
samples = np.random.multivariate_normal(mu, Sigma, size=n_samples)

# --- Plot ---
fig = plt.figure(figsize=(7.5, 6.5))
ax = fig.add_subplot(111, projection='3d')

# Wireframe ellipsoid (no explicit color set; matplotlib default used)
ax.plot_wireframe(Xe, Ye, Ze, rstride=3, cstride=3, linewidth=0.6)

# Sample points
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=6, alpha=0.25)

# Mean point
ax.scatter([mu[0]], [mu[1]], [mu[2]], s=60)

ax.set_title("3D Gaussian visualization: samples + constant-density ellipsoid")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Equal-ish aspect ratio
mins = np.minimum(ellipsoid.reshape(-1, 3).min(axis=0), samples.min(axis=0))
maxs = np.maximum(ellipsoid.reshape(-1, 3).max(axis=0), samples.max(axis=0))
center = 0.5 * (mins + maxs)
span = (maxs - mins).max()
ax.set_xlim(center[0] - 0.55*span, center[0] + 0.55*span)
ax.set_ylim(center[1] - 0.55*span, center[1] + 0.55*span)
ax.set_zlim(center[2] - 0.55*span, center[2] + 0.55*span)

plt.show()


#%% 3D Gaussian function + color
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
mu = np.array([0.5, -0.2, 0.3])
Sigma = np.array([
    [1.0,  0.6, 0.2],
    [0.6,  1.5, 0.3],
    [0.2,  0.3, 0.7]
], dtype=float)
Sigma = (Sigma + Sigma.T) / 2
Sigma_inv = np.linalg.inv(Sigma)

k = 5  # ellipsoid level

# --- Helper: Gaussian value (unnormalized) ---
def gaussian_val(x, mu, Sigma_inv):
    d = x - mu
    return np.exp(-0.5 * np.einsum('...i,ij,...j', d, Sigma_inv, d))

# --- Ellipsoid surface from eigendecomposition ---
eigvals, eigvecs = np.linalg.eigh(Sigma)
u = np.linspace(0, 2*np.pi, 90)
v = np.linspace(0, np.pi, 45)
uu, vv = np.meshgrid(u, v)
sphere = np.stack([
    np.cos(uu) * np.sin(vv),
    np.sin(uu) * np.sin(vv),
    np.cos(vv)
], axis=-1)

radii = np.sqrt(k * eigvals)
A = eigvecs @ np.diag(radii)
ellipsoid = sphere @ A.T + mu
Xe, Ye, Ze = ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2]

# --- Sample points ---
np.random.seed(0)
samples = np.random.multivariate_normal(mu, Sigma, size=2500)
vals = gaussian_val(samples, mu, Sigma_inv)

# Normalize for color mapping
vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)

# --- Plot ---
fig = plt.figure(figsize=(7.5, 6.5))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
                c=vals_norm, s=8, alpha=0.7, cmap='viridis')
ax.plot_wireframe(Xe, Ye, Ze, rstride=3, cstride=3, linewidth=0.6)

ax.scatter([mu[0]], [mu[1]], [mu[2]], s=80)
ax.set_title("3D Gaussian: color indicates G(x), wireframe is constant-density ellipsoid")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.1)
cbar.set_label("Normalized G(x)")

# Set bounds
mins = np.minimum(ellipsoid.reshape(-1, 3).min(axis=0), samples.min(axis=0))
maxs = np.maximum(ellipsoid.reshape(-1, 3).max(axis=0), samples.max(axis=0))
center = 0.5 * (mins + maxs)
span = (maxs - mins).max()
ax.set_xlim(center[0] - 0.55*span, center[0] + 0.55*span)
ax.set_ylim(center[1] - 0.55*span, center[1] + 0.55*span)
ax.set_zlim(center[2] - 0.55*span, center[2] + 0.55*span)

plt.show()


#%% 3D to 2D
import numpy as np
import matplotlib.pyplot as plt

# ---- 3D Gaussian parameters ----
mu = np.array([0.5, -0.2, 1.0])
Sigma = np.array([
    [1.0,  0.4, 0.3],
    [0.4,  1.2, 0.2],
    [0.3,  0.2, 0.8]
], dtype=float)
Sigma = (Sigma + Sigma.T) / 2

# Inverse for G(x)
Sigma_inv = np.linalg.inv(Sigma)

def G3(x):
    d = x - mu
    return np.exp(-0.5 * (d @ Sigma_inv @ d))

# ---- Sample points in 3D ----
np.random.seed(0)
N = 2500
X3 = np.random.multivariate_normal(mu, Sigma, size=N)

# ---- Orthographic projection to image plane: x' = P x (drop z) ----
P = np.array([[1, 0, 0],
              [0, 1, 0]])

X2 = (P @ X3.T).T

# ---- Induced 2D Gaussian parameters ----
mu2 = P @ mu
Sigma2 = P @ Sigma @ P.T
Sigma2_inv = np.linalg.inv(Sigma2)

def G2(x):
    d = x - mu2
    return np.exp(-0.5 * (d @ Sigma2_inv @ d))

# =========================================================
# Figure 1: 3D Gaussian samples
# =========================================================
fig1 = plt.figure(figsize=(7, 6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(X3[:, 0], X3[:, 1], X3[:, 2], s=6, alpha=0.3)
ax1.scatter([mu[0]], [mu[1]], [mu[2]], s=60)
ax1.set_title("3D Gaussian samples G(x) in space")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
plt.tight_layout()
plt.show()

# =========================================================
# Figure 2: 2D projected Gaussian on image plane
# =========================================================
# grid for contours
x_min, x_max = X2[:, 0].min()-1, X2[:, 0].max()+1
y_min, y_max = X2[:, 1].min()-1, X2[:, 1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.stack([xx, yy], axis=-1)

G_vals = np.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        G_vals[i, j] = G2(grid[i, j])

fig2 = plt.figure(figsize=(6, 6))
plt.scatter(X2[:, 0], X2[:, 1], s=6, alpha=0.3)
plt.contour(xx, yy, G_vals, levels=8)
plt.scatter([mu2[0]], [mu2[1]], s=60)
plt.title("Projected 2D Gaussian G'(x') on image plane")
plt.xlabel("x'")
plt.ylabel("y'")
plt.axis("equal")
plt.tight_layout()
plt.show()

print("3D mean mu =", mu)
print("3D covariance Sigma =\n", Sigma)
print("\nProjected 2D mean mu' =", mu2)
print("Projected 2D covariance Sigma' =\n", Sigma2)
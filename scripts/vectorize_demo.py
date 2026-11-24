"""Compare pure-Python and NumPy standardisation; demonstrate a broadcasting error."""
import time

import numpy as np

from mlkit.numops import standardize


def standardize_loop(rows):
    n, d = len(rows), len(rows[0])
    means = [sum(r[j] for r in rows) / n for j in range(d)]
    stds = [(sum((r[j] - means[j]) ** 2 for r in rows) / n) ** 0.5 for j in range(d)]
    return [[(r[j] - means[j]) / stds[j] for j in range(d)] for r in rows]


rng = np.random.default_rng(0)
X = rng.normal(loc=50, scale=10, size=(200_000, 5))
rows = X.tolist()

t0 = time.perf_counter()
Z_loop = standardize_loop(rows)
t_loop = time.perf_counter() - t0

t0 = time.perf_counter()
Z_vec, mean, std = standardize(X)
t_vec = time.perf_counter() - t0

print(f"X shape: {X.shape}, mean shape: {mean.shape}, std shape: {std.shape}")
print(f"pure Python: {t_loop:.3f}s  NumPy: {t_vec:.4f}s  speedup: {t_loop / t_vec:.0f}x")
print("results match:", np.allclose(np.array(Z_loop), Z_vec))

# A common bug: centring each ROW with axis=1 but forgetting keepdims
try:
    X - X.mean(axis=1)
except ValueError as err:
    print("broadcasting error:", err)
print("row-wise fix shape:", (X - X.mean(axis=1, keepdims=True)).shape)

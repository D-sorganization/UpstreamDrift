import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

T_basis = 1.8138888888888889

def bernstein(k, n, u):
    return comb(n, k) * (u**k) * ((1.0 - u)**(n - k))

def d_bernstein(k, n, u):
    # derivative with respect to u
    term1 = k * (u**(k-1)) * ((1.0 - u)**(n-k)) if k > 0 else 0
    term2 = -(n - k) * (u**k) * ((1.0 - u)**(n - k - 1)) if n - k > 0 else 0
    return comb(n, k) * (term1 + term2)

n = 6
t_eval = np.array([0.0, 0.20, 0.40, 0.60, 0.80, 1.05, 1.15, 1.233333, 1.813889])
u_eval = t_eval / T_basis

print("=== BERNSTEIN BASIS POLYNOMIAL VALUES (Degree 6, T_basis = 1.813889s) ===")
header = f"{'t (s)':8s} | {'u':6s} | " + " | ".join([f"B_{k}(u)" for k in range(7)])
print(header)
print("-" * len(header))

for t, u in zip(t_eval, u_eval):
    b_vals = [bernstein(k, n, u) for k in range(7)]
    row = f"{t:8.4f} | {u:6.4f} | " + " | ".join([f"{b:6.4f}" for b in b_vals])
    print(row)

print("\n=== BASIS AUTHORITY PEAK LOCATIONS ===")
for k in range(7):
    u_peak = k / n
    t_peak = u_peak * T_basis
    b_peak = bernstein(k, n, u_peak)
    print(f"k = {k}: Peak at u = {u_peak:.4f} (t = {t_peak:.4f}s), max B_{k} = {b_peak:.4f}")

# Detailed analysis on delivery window [1.05s, 1.233s]
t_window = np.linspace(1.05, 1.233333, 100)
u_window = t_window / T_basis

print("\n=== BASIS AUTHORITY IN DELIVERY WINDOW [1.05s, 1.233s] ===")
for k in range(7):
    b_k = np.array([bernstein(k, n, u) for u in u_window])
    print(f"k = {k}: min = {b_k.min():.4f}, max = {b_k.max():.4f}, mean = {b_k.mean():.4f}, at t=1.233s = {b_k[-1]:.4f}")

print("\n=== RELATIVE BASIS CONTRIBUTIONS AT t = 1.233s (u = 0.680) ===")
u_impact = 1.233333 / T_basis
b_impact = np.array([bernstein(k, n, u_impact) for k in range(7)])
for k in range(7):
    print(f"k = {k}: {b_impact[k]:.4f} ({b_impact[k]*100:.1f}%)")

sum_frozen = sum(b_impact[:4])
sum_k45 = sum(b_impact[4:6])
k6_val = b_impact[6]
print(f"\nFrozen history (k=0,1,2,3) contribution at impact: {sum_frozen:.4f} ({sum_frozen*100:.1f}%)")
print(f"k=4, 5 contribution at impact:                      {sum_k45:.4f} ({sum_k45*100:.1f}%)")
print(f"k=6 contribution at impact:                         {k6_val:.4f} ({k6_val*100:.1f}%)")

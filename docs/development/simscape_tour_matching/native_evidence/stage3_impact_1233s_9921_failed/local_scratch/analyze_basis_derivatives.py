import numpy as np
from scipy.special import comb

T_basis = 1.8138888888888889

def d_bernstein(k, n, u):
    # derivative w.r.t t: dB/dt = (1/T_basis) * dB/du
    # dB/du = n * (B_{k-1, n-1}(u) - B_{k, n-1}(u))
    term_prev = comb(n - 1, k - 1) * (u**(k - 1)) * ((1.0 - u)**(n - k)) if (k > 0 and n - k >= 0) else 0.0
    term_curr = comb(n - 1, k) * (u**k) * ((1.0 - u)**(n - 1 - k)) if (k < n and n - 1 - k >= 0) else 0.0
    return (n / T_basis) * (term_prev - term_curr)

n = 6
t = 1.233333
u = t / T_basis

print("=== DERIVATIVES (RATE OF EFFORT INJECTION dB_k/dt) AT t = 1.233s ===")
for k in range(7):
    db_dt = d_bernstein(k, n, u)
    print(f"k = {k}: dB_{k}/dt = {db_dt:+.4f} s^-1")

print("\n=== DERIVATIVES AT DELIVERY HORIZONS ===")
for t_h in [1.05, 1.15, 1.233333]:
    u_h = t_h / T_basis
    print(f"\nHorizon t = {t_h:.3f}s (u = {u_h:.4f}):")
    for k in [4, 5, 6]:
        db_dt = d_bernstein(k, n, u_h)
        print(f"  k = {k}: dB/dt = {db_dt:+.4f} s^-1")

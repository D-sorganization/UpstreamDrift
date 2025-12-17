"""
Torque polynomial fitting tool.
- Fits tau(t) with an n-th degree polynomial via least squares.
- Plots original vs fitted data.
- Optionally saves polynomial coefficients to a .npy file.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np  # noqa: TID253
import numpy.typing as npt  # noqa: TID253


def fit_torque_poly(
    t: npt.ArrayLike, tau: npt.ArrayLike, degree: int = 6
) -> npt.NDArray[np.float64]:
    """
    Fit polynomial: tau(t) ≈ p(t) = c0 + c1 t + ... + cN t^N
    Returns coefficients (highest degree first, like np.polyval).
    """
    t_arr = np.asarray(t, dtype=np.float64).flatten()
    tau_arr = np.asarray(tau, dtype=np.float64).flatten()
    if t_arr.shape != tau_arr.shape:
        msg = "t and tau must have same shape"
        raise ValueError(msg)
    coeffs = np.polyfit(t_arr, tau_arr, degree)
    return np.asarray(coeffs, dtype=np.float64)


def evaluate_torque_poly(
    coeffs: npt.NDArray[np.float64], t: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    t_arr = np.asarray(t, dtype=np.float64)
    result = np.polyval(coeffs, t_arr)
    return np.asarray(result, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit polynomial to torque vs time data."
    )
    parser.add_argument("csv", help="CSV file with columns t, tau")
    parser.add_argument(
        "-d", "--degree", type=int, default=6, help="Polynomial degree (default: 6)"
    )
    parser.add_argument(
        "-o", "--out", type=str, default="", help="Output .npy file for coefficients"
    )
    args = parser.parse_args()

    data = np.loadtxt(args.csv, delimiter=",", skiprows=1)
    t = data[:, 0]
    tau = data[:, 1]

    coeffs = fit_torque_poly(t, tau, degree=args.degree)

    # Plot
    t_dense = np.linspace(t.min(), t.max(), 1000)
    tau_fit = evaluate_torque_poly(coeffs, t_dense)

    plt.figure()
    plt.plot(t, tau, ".", label="data")
    plt.plot(t_dense, tau_fit, "-", label=f"poly deg {args.degree}")
    plt.xlabel("time")
    plt.ylabel("torque")
    plt.legend()
    plt.title("Torque polynomial fit")
    plt.grid(True)
    plt.show()

    if args.out:
        np.save(args.out, coeffs)


if __name__ == "__main__":
    main()

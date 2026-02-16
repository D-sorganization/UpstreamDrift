"""Nonlinear dynamics and complexity analysis.

Includes Lyapunov exponents, correlation dimension, recurrence quantification,
entropy measures, fractal dimension, and local divergence rates.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

from src.shared.python.analysis.dataclasses import RQAMetrics


class NonlinearDynamicsMixin:
    """Mixin for nonlinear dynamics and complexity analysis.

    Expects the following attributes to be available on the instance:
    - times: np.ndarray
    - joint_positions: np.ndarray
    - joint_velocities: np.ndarray
    - dt: float
    """

    times: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    dt: float

    def compute_local_divergence_rate(
        self,
        joint_idx: int = 0,
        tau: int = 1,
        dim: int = 3,
        window: int = 50,
        data_type: str = "velocity",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute local divergence rate over time (Local Lyapunov proxy).

        Instead of a single exponent, this returns a time series showing how
        fast nearby trajectories are diverging *at each point in time*.

        Args:
            joint_idx: Joint index
            tau: Time lag
            dim: Embedding dimension
            window: Theiler window
            data_type: 'position' or 'velocity'

        Returns:
            Tuple of (times, divergence_rates)
        """
        if data_type == "position":
            data = self.joint_positions[:, joint_idx]
        else:
            data = self.joint_velocities[:, joint_idx]

        N = len(data)
        M = N - (dim - 1) * tau
        if window + 1 > M:
            return np.array([]), np.array([])

        # Reconstruct Phase Space
        orbit = np.zeros((M, dim))
        for d in range(dim):
            orbit[:, d] = data[d * tau : d * tau + M]

        # Find Nearest Neighbors
        from scipy.spatial.distance import cdist

        dists_mat = cdist(orbit, orbit, metric="euclidean")

        # Apply Theiler window
        for i in range(M):
            start = max(0, i - window)
            end = min(M, i + window + 1)
            dists_mat[i, start:end] = np.inf
            dists_mat[i, i] = np.inf

        nearest_neighbors = np.argmin(dists_mat, axis=1)

        # Compute Divergence Rate
        lookahead = min(10, int(0.1 / self.dt)) if self.dt > 0 else 5
        divergence_rates = np.zeros(M - lookahead)

        indices = np.arange(M - lookahead)
        nn_indices = nearest_neighbors[indices]

        valid_mask = nn_indices < (M - lookahead)
        valid_i = indices[valid_mask]
        valid_nn = nn_indices[valid_mask]

        if len(valid_i) > 0:
            diff_0 = orbit[valid_i] - orbit[valid_nn]
            dist_sq_0 = np.sum(diff_0**2, axis=1)

            diff_t = orbit[valid_i + lookahead] - orbit[valid_nn + lookahead]
            dist_sq_t = np.sum(diff_t**2, axis=1)

            safe_mask = (dist_sq_0 > 1e-18) & (dist_sq_t > 1e-18)
            denom = lookahead * self.dt

            if denom > 0:
                divergence_rates[valid_i[safe_mask]] = (
                    0.5 * np.log(dist_sq_t[safe_mask] / dist_sq_0[safe_mask]) / denom
                )

        valid_times = self.times[: len(divergence_rates)]
        return valid_times, divergence_rates

    def compute_recurrence_matrix(
        self,
        threshold_ratio: float = 0.1,
        metric: str = "euclidean",
        use_sparse: bool = False,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.int_]]:
        """Compute Recurrence Plot matrix.

        Args:
            threshold_ratio: Threshold distance as ratio of max phase space diameter.
            metric: Distance metric (e.g., 'euclidean', 'cityblock').
            use_sparse: If True, uses cKDTree for memory-efficient computation.

        Returns:
            Binary recurrence matrix (N, N).
        """
        if (
            self.joint_positions.shape[1] == 0
            or self.joint_velocities.shape[1] == 0
            or len(self.times) < 2
        ):
            return np.zeros((0, 0), dtype=np.int_)

        state_vec = np.hstack((self.joint_positions, self.joint_velocities))

        mean = np.mean(state_vec, axis=0)
        std = np.std(state_vec, axis=0)
        std[std < 1e-6] = 1.0
        normalized_state = (state_vec - mean) / std

        N = len(normalized_state)

        if use_sparse and metric == "euclidean" and N > 100:
            tree = cKDTree(normalized_state)

            sample_size = min(100, N)
            rng = np.random.default_rng(0)
            sample_indices = rng.choice(N, sample_size, replace=False)
            sample_dists: list[Any] = []
            for i in sample_indices:
                dists_i, _ = tree.query(normalized_state[i], k=min(10, N))
                sample_dists.extend(dists_i[1:])
            estimated_max = np.max(sample_dists) * 2
            threshold = threshold_ratio * estimated_max

            recurrence_matrix = np.zeros((N, N), dtype=np.int_)
            for i in range(N):
                neighbors = tree.query_ball_point(normalized_state[i], threshold)
                for j in neighbors:
                    if j >= i:
                        recurrence_matrix[i, j] = 1
                        recurrence_matrix[j, i] = 1

            return cast(
                np.ndarray[tuple[int, int], np.dtype[np.int_]], recurrence_matrix
            )

        dists = pdist(normalized_state, metric=metric)
        dist_matrix = squareform(dists)

        if threshold_ratio is None:
            threshold_ratio = 0.1
        threshold = threshold_ratio * np.max(dist_matrix)

        recurrence_matrix = (dist_matrix < threshold).astype(np.int_)
        return cast(np.ndarray[tuple[int, int], np.dtype[np.int_]], recurrence_matrix)

    def compute_cross_recurrence_matrix(
        self,
        joint_idx_1: int,
        joint_idx_2: int,
        threshold_ratio: float = 0.1,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.int_]]:
        """Compute Cross Recurrence Plot matrix between two joints.

        Args:
            joint_idx_1: First joint index
            joint_idx_2: Second joint index
            threshold_ratio: Threshold distance as ratio of max distance

        Returns:
            Binary recurrence matrix (N, N)
        """
        s1 = np.column_stack(
            (
                self.joint_positions[:, joint_idx_1],
                self.joint_velocities[:, joint_idx_1],
            )
        )
        s2 = np.column_stack(
            (
                self.joint_positions[:, joint_idx_2],
                self.joint_velocities[:, joint_idx_2],
            )
        )

        s1 = (s1 - np.mean(s1, axis=0)) / (np.std(s1, axis=0) + 1e-9)
        s2 = (s2 - np.mean(s2, axis=0)) / (np.std(s2, axis=0) + 1e-9)

        from scipy.spatial.distance import cdist

        dist_matrix = cdist(s1, s2, metric="euclidean")
        threshold = threshold_ratio * np.max(dist_matrix)
        recurrence_matrix = (dist_matrix < threshold).astype(np.int_)

        return cast(np.ndarray[tuple[int, int], np.dtype[np.int_]], recurrence_matrix)

    def compute_rqa_metrics(
        self,
        recurrence_matrix: np.ndarray,
        min_line_length: int = 2,
    ) -> RQAMetrics | None:
        """Compute Recurrence Quantification Analysis (RQA) metrics.

        Args:
            recurrence_matrix: Binary recurrence matrix (N, N)
            min_line_length: Minimum length to consider a line

        Returns:
            RQAMetrics object or None
        """
        if recurrence_matrix.size == 0:
            return None

        N = recurrence_matrix.shape[0]
        if N < 2:
            return None

        n_recurrence_points = np.sum(recurrence_matrix) - N
        rr = n_recurrence_points / (N * N - N) if N > 1 else 0.0

        diagonal_lengths: list[int] = []
        for k in range(1, N):
            diag = np.diagonal(recurrence_matrix, offset=k)
            d = np.concatenate((np.array([0]), diag, np.array([0])))
            diffs = np.diff(d)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            lengths = ends - starts
            diagonal_lengths.extend(lengths[lengths >= min_line_length])

        n_diag_points = np.sum(diagonal_lengths)
        det = n_diag_points / n_recurrence_points if n_recurrence_points > 0 else 0.0
        l_max = np.max(diagonal_lengths) if len(diagonal_lengths) > 0 else 0

        vertical_lengths: list[int] = []
        for i in range(N):
            col = recurrence_matrix[:, i].copy()
            col[i] = 0

            c = np.concatenate((np.array([0]), col, np.array([0])))
            diffs = np.diff(c)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            lengths = ends - starts
            vertical_lengths.extend(lengths[lengths >= min_line_length])

        n_vert_points = np.sum(vertical_lengths)
        lam = n_vert_points / n_recurrence_points if n_recurrence_points > 0 else 0.0
        tt = float(np.mean(vertical_lengths)) if len(vertical_lengths) > 0 else 0.0

        return RQAMetrics(
            recurrence_rate=float(rr),
            determinism=float(det),
            laminarity=float(lam),
            longest_diagonal_line=int(l_max),
            trapping_time=tt,
        )

    def compute_correlation_dimension(
        self, data: np.ndarray, tau: int = 1, dim: int = 3
    ) -> float:
        """Estimate Correlation Dimension (D2) using Grassberger-Procaccia algorithm.

        Args:
            data: Time series
            tau: Time delay
            dim: Embedding dimension

        Returns:
            Estimated Correlation Dimension
        """
        N = len(data)
        M = N - (dim - 1) * tau
        if M < 20:
            return 0.0

        orbit = np.zeros((M, dim))
        for d in range(dim):
            orbit[:, d] = data[d * tau : d * tau + M]

        from scipy.spatial.distance import pdist as _pdist

        dists = _pdist(orbit, metric="euclidean")

        dists = dists[dists > 1e-9]
        if len(dists) == 0:
            return 0.0

        min_r, max_r = np.min(dists), np.max(dists)
        radii = np.geomspace(min_r * 2, max_r * 0.5, 20)
        c_r = []

        for r in radii:
            count = np.sum(dists < r)
            c_r.append(count / len(dists))

        log_r = np.log(radii)
        log_c = np.log(c_r)

        n_points = len(log_r)
        start = n_points // 4
        end = 3 * n_points // 4

        slope, _ = np.polyfit(log_r[start:end], log_c[start:end], 1)
        return float(slope)

    def estimate_lyapunov_exponent(
        self,
        data: np.ndarray,
        tau: int = 1,
        dim: int = 3,
        window: int = 50,
    ) -> float:
        """Estimate the Largest Lyapunov Exponent (LLE) using Rosenstein's algorithm.

        Args:
            data: 1D time series array
            tau: Time delay (lag)
            dim: Embedding dimension
            window: Minimum temporal separation for nearest neighbors

        Returns:
            Estimated LLE (nats/s)
        """
        N = len(data)
        if window > N:
            return 0.0

        M = N - (dim - 1) * tau
        if M < 1:
            return 0.0

        orbit = np.zeros((M, dim))
        for d in range(dim):
            orbit[:, d] = data[d * tau : d * tau + M]

        nearest_neighbors = np.zeros(M, dtype=int)

        from scipy.spatial.distance import cdist

        dists_mat = cdist(orbit, orbit, metric="euclidean")

        for i in range(M):
            start = max(0, i - window)
            end = min(M, i + window + 1)
            dists_mat[i, start:end] = np.inf
            dists_mat[i, i] = np.inf

        nearest_neighbors = np.argmin(dists_mat, axis=1)

        max_steps = min(M, int(1.0 / self.dt * 0.5)) if self.dt > 0 else 10

        divergence = np.zeros(max_steps)
        counts = np.zeros(max_steps)

        for i in range(max_steps):
            idx1_vec = np.arange(M) + i
            idx2_vec = nearest_neighbors + i

            valid_mask = (idx1_vec < M) & (idx2_vec < M)

            if not np.any(valid_mask):
                continue

            p1 = orbit[idx1_vec[valid_mask]]
            p2 = orbit[idx2_vec[valid_mask]]

            diff = p1 - p2
            dists = np.sqrt(np.sum(diff**2, axis=1))

            valid_dists_mask = dists > 1e-9
            valid_dists = dists[valid_dists_mask]

            if len(valid_dists) > 0:
                divergence[i] += np.sum(np.log(valid_dists))
                counts[i] += len(valid_dists)

        counts[counts == 0] = 1.0
        avg_log_dist = divergence / counts

        t_axis = np.arange(max_steps) * self.dt

        if len(t_axis) > 1:
            slope, _ = np.polyfit(t_axis, avg_log_dist, 1)
            return float(slope)

        return 0.0

    def compute_permutation_entropy(
        self,
        data: np.ndarray,
        order: int = 3,
        delay: int = 1,
    ) -> float:
        """Compute Permutation Entropy.

        Args:
            data: 1D time series
            order: Order of permutation (embedding dimension)
            delay: Time delay

        Returns:
            Entropy value (bits)
        """
        N = len(data)
        M = N - (order - 1) * delay
        if M < 1:
            return 0.0

        matrix = np.zeros((M, order), dtype=data.dtype)
        for i in range(order):
            matrix[:, i] = data[i * delay : i * delay + M]

        ranks = np.argsort(matrix, axis=1)

        if order <= 12:
            packed = np.zeros(M, dtype=np.int64)
            multiplier = 1
            for i in range(order):
                packed += ranks[:, i] * multiplier
                multiplier *= order

            _, counts = np.unique(packed, return_counts=True)
        else:
            _, counts = np.unique(ranks, axis=0, return_counts=True)

        probs = counts / M
        probs = probs[probs > 0]

        pe = -np.sum(probs * np.log2(probs))
        return float(pe)

    def compute_sample_entropy(
        self,
        data: np.ndarray,
        m: int = 2,
        r: float = 0.2,
    ) -> float:
        """Compute Sample Entropy (SampEn).

        Args:
            data: 1D time series
            m: Template length (embedding dimension)
            r: Tolerance (typically 0.2 * std)

        Returns:
            Sample Entropy value
        """
        N = len(data)
        if m + 1 > N:
            return 0.0

        tolerance = r * np.std(data)

        def count_matches(template_len: int) -> int:
            """Count template-matching vector pairs within the tolerance radius."""
            n_vectors = N - template_len

            X = np.zeros((n_vectors, template_len))
            for i in range(template_len):
                X[:, i] = data[i : i + n_vectors]

            tree = cKDTree(X)
            count = tree.count_neighbors(tree, r=tolerance, p=np.inf)
            B = (count - n_vectors) // 2

            return int(B)

        A = count_matches(m)
        B = count_matches(m + 1)

        if A == 0 or B == 0:
            return 0.0

        return float(-np.log(B / A))

    def compute_multiscale_entropy(
        self,
        data: np.ndarray,
        max_scale: int = 10,
        m: int = 2,
        r: float = 0.15,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Multiscale Entropy (MSE).

        Args:
            data: 1D time series
            max_scale: Maximum scale factor
            m: Template length
            r: Tolerance (ratio of std)

        Returns:
            Tuple of (scales, entropy_values)
        """
        mse_values = []
        scales = np.arange(1, max_scale + 1)

        for scale in scales:
            if scale == 1:
                scaled_data = data
            else:
                n_windows = len(data) // scale
                if n_windows < m + 1:
                    mse_values.append(0.0)
                    continue
                truncated = data[: n_windows * scale]
                reshaped = truncated.reshape(n_windows, scale)
                scaled_data = np.mean(reshaped, axis=1)

            std_current = np.std(scaled_data)
            std_original = np.std(data)

            if std_current < 1e-9:
                mse = 0.0
            else:
                r_ratio = (r * std_original) / std_current
                mse = self.compute_sample_entropy(scaled_data, m=m, r=r_ratio)

            mse_values.append(mse)

        return scales, np.array(mse_values)

    def compute_fractal_dimension(
        self,
        data: np.ndarray,
        k_max: int = 10,
    ) -> float:
        """Compute Fractal Dimension using Higuchi's method.

        Args:
            data: 1D time series
            k_max: Maximum interval time (k)

        Returns:
            Fractal dimension (HFD) approx between 1.0 and 2.0
        """
        N = len(data)
        if k_max + 1 > N:
            return 1.0

        L_k = []
        x_k = []

        for k in range(1, k_max + 1):
            L_m_k = 0.0
            abs_diffs = np.abs(data[k:] - data[:-k])

            for m_val in range(k):
                m_diffs = abs_diffs[m_val::k]
                n_intervals = len(m_diffs)

                if n_intervals > 0:
                    L_m_k += np.sum(m_diffs) * (N - 1) / (n_intervals * k)

            L_k.append(L_m_k / (k * k))
            x_k.append(np.log(1.0 / k))

        y_val = np.log(L_k)
        slope, _ = np.polyfit(x_k, y_val, 1)

        return float(slope)

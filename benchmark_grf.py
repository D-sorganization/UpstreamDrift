
import time
import numpy as np
from shared.python.statistical_analysis import StatisticalAnalyzer, GRFMetrics

def benchmark_grf_metrics():
    # Setup data
    n_samples = 100000
    times = np.linspace(0, 10, n_samples)

    # Random data for other fields required by __init__
    joint_positions = np.zeros((n_samples, 10))
    joint_velocities = np.zeros((n_samples, 10))
    joint_torques = np.zeros((n_samples, 10))

    # CoP Position (N, 2)
    cop_position = np.random.rand(n_samples, 2) * 0.5

    analyzer = StatisticalAnalyzer(
        times=times,
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        joint_torques=joint_torques,
        cop_position=cop_position
    )

    # Warmup
    analyzer.compute_grf_metrics()

    # Benchmark
    start_time = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        analyzer.compute_grf_metrics()
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / iterations
    print(f"Average time per call: {avg_time*1000:.4f} ms")

if __name__ == "__main__":
    benchmark_grf_metrics()

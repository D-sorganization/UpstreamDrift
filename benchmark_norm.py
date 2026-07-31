import numpy as np
import timeit

def bench_linalg_norm(arr):
    return np.linalg.norm(arr, axis=1, keepdims=True)

def bench_einsum(arr):
    return np.sqrt(np.einsum('...i,...i->...', arr, arr))[..., None]

def bench_einsum_axis2(arr):
    return np.sqrt(np.einsum('...i,...i->...', arr, arr))[..., None]

arr = np.random.rand(1000, 3)

t1 = timeit.timeit("bench_linalg_norm(arr)", globals=globals(), number=10000)
t2 = timeit.timeit("bench_einsum(arr)", globals=globals(), number=10000)

print(f"np.linalg.norm: {t1:.4f}s")
print(f"np.einsum: {t2:.4f}s")

arr2 = np.random.rand(10, 1000, 3)
t3 = timeit.timeit("np.linalg.norm(arr2, axis=2, keepdims=True)", globals=globals(), number=1000)
t4 = timeit.timeit("bench_einsum_axis2(arr2)", globals=globals(), number=1000)

print(f"np.linalg.norm axis=2: {t3:.4f}s")
print(f"np.einsum axis=2: {t4:.4f}s")

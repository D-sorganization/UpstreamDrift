# Issue: Implement SimscapeAdapterPool for Parallel Inference (Option 4)

## Summary

Implement `SimscapeAdapterPool`: a process pool of `SimscapeAdapter` workers,
each owning a separate MATLAB Engine, with `map_simulate` and `imap_simulate`
methods. Pool size bounded above by available MATLAB licenses.

## Motivation

See `motion_matching/option4_python_bridge/INTERFACES.md` §"`SimscapeAdapterPool`".
A single `SimscapeAdapter` is not thread-safe (MATLAB Engine is single-threaded),
so concurrent fits or batched inference require multiple worker processes.
This is the bridge equivalent of the parsim option in MATLAB-side multistart (#025).

## Dependencies

- #036 (skeleton).
- #037 (working `simulate_with_coefficients`).

## File targets

- New: `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\simscape_adapter_pool.py`
- New: `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option4_python_bridge\worker.py` (subprocess entry point)
- New: `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\option4\test_simscape_adapter_pool.py`

## Public API

Verbatim from `INTERFACES.md`:

```python
class SimscapeAdapterPool:
    """Pool of SimscapeAdapter instances for parallel inference.

    Each pool worker is a separate Python process owning one MATLAB Engine.
    Pool size is bounded above by the host's MATLAB license count.

    Usage:
        with SimscapeAdapterPool(pool_size=4, model_path=slx_path) as pool:
            outs = pool.map_simulate(thetas)

    Cache: each worker has its own in-process cache. Cross-worker caching
    is an explicit non-goal for v1.
    """

    @precondition(
        lambda self, pool_size, model_path:
            isinstance(pool_size, int) and pool_size >= 1,
        "pool_size must be a positive int",
    )
    @precondition(
        lambda self, pool_size, model_path:
            isinstance(model_path, str) and model_path.endswith(".slx"),
        "model_path must end in .slx",
    )
    def __init__(self, pool_size: int, model_path: str) -> None: ...

    @precondition(
        lambda self, thetas: all(
            isinstance(t, np.ndarray) and t.ndim == 1 and t.size % 7 == 0
            for t in thetas
        ),
        "every theta must be a 1-D numpy array with size multiple of 7",
    )
    def map_simulate(self, thetas: Sequence[np.ndarray]) -> list[SimscapeOutput]:
        """Distribute simulations across pool workers.

        Order of results matches order of inputs.
        """

    def imap_simulate(self, thetas: Iterable[np.ndarray]) -> Iterable[SimscapeOutput]:
        """Streaming variant. Yields results as they complete (any order)."""

    def close(self) -> None:
        """Quit every worker engine. Idempotent."""

    def __enter__(self) -> "SimscapeAdapterPool": ...
    def __exit__(self, *exc_info: object) -> None: ...
```

## Required tests (TDD)

- `test_pool_init_rejects_non_positive_pool_size`
- `test_pool_init_rejects_non_slx_model_path`
- `test_pool_starts_pool_size_worker_processes_each_with_one_matlab_engine`
- `test_pool_map_simulate_returns_n_outputs_in_input_order`
- `test_pool_map_simulate_uses_all_workers_balanced_within_2x`
- `test_pool_imap_simulate_streams_results_as_they_complete`
- `test_pool_handles_worker_crash_with_clear_error_does_not_hang`
- `test_pool_close_quits_every_worker_engine`
- `test_pool_close_is_idempotent`
- `test_pool_context_manager_calls_close_on_exit`
- `test_pool_speedup_versus_serial_within_30_percent_of_pool_size_for_64_thetas`
- `test_pool_each_worker_has_own_cache_no_cross_worker_sharing`
- `test_pool_marked_live_simulation_for_real_matlab_engine_path`

Tests that touch real workers should be marked
`@pytest.mark.live_simulation`. Offline tests use a stub adapter.

## DbC contract

Preconditions inherited verbatim from `INTERFACES.md`:

- `pool_size >= 1`.
- `model_path.endswith(".slx")`.
- Every theta is a 1-D numpy array with size multiple of 7.

Postconditions:

- `len(map_simulate(thetas)) == len(thetas)`.
- Order of `map_simulate` results matches input order.

## Acceptance Criteria

- [ ] `SimscapeAdapterPool` works end-to-end with at least 4 workers.
- [ ] Live tests pass under `pytest -m live_simulation`.
- [ ] Offline tests pass under `pytest -m "not live_simulation"`.
- [ ] Worker crash handling verified (kill a worker mid-batch; pool surfaces
      error and does not hang).
- [ ] Speed-up benchmark recorded in test output for the live path.
- [ ] DbC decorators present and verbatim from `INTERFACES.md`.
- [ ] `ruff check` and `ruff format --check` clean.
- [ ] No file exceeds 1200 lines.
- [ ] No `print()`; use `get_logger`.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `option4`, `python`, `tdd`, `dbc`

## Effort estimate

L (3-7 days). Multiprocessing + MATLAB Engine startup time + license accounting
make this the trickiest issue in Option 4 after #037.

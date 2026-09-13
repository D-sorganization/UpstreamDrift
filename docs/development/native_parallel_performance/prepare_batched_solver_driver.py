"""Create an isolated native solver driver with the qualified batch adapter.

The input must be the immutable run20 driver.  This narrow text transformation
is deliberately fail-closed: every anchor must occur exactly once.  It creates
a new driver and never changes its source or any archived run output.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one driver anchor: {old[:48]}")
    return source.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source.read_text()
    source = _replace_once(
        source,
        "from src.shared.python.motion_matching.multi_shooting_fit import MultipleShootingOptions,fit_multiple_shooting",
        "from src.shared.python.motion_matching.multi_shooting_fit import MultipleShootingOptions,fit_multiple_shooting\n"
        "from src.shared.python.motion_matching.native_window_executor import NativeWindowExecutor\n"
        "from src.engines.physics_engines.pinocchio.python.native_sensitivity_batch import (NativeSensitivityBatchAdapter,NativeSensitivityWindowRequest,evaluate_trusted_native_sensitivity_request)",
    )
    source = _replace_once(
        source,
        "parser.add_argument('--audit-only',action='store_true');parser.add_argument('--horizon'",
        "parser.add_argument('--audit-only',action='store_true');"
        "parser.add_argument('--batch-workers',type=int,choices=(0,2),default=0);"
        "parser.add_argument('--horizon'",
    )
    source = _replace_once(
        source,
        "def full(theta,clock):return replay_candidate(raw,candidate(theta),clock,rtol=1e-11,atol=1e-13,max_step=.00025).markers_m",
        "def full(theta,clock):return replay_candidate(raw,candidate(theta),clock,rtol=1e-11,atol=1e-13,max_step=.00025).markers_m\n"
        "def batch_request(theta,clock,state):\n"
        " c=candidate(theta);start=np.array(c.document['q0']+c.document['qd0']) if state is None else state\n"
        " tangent=None if state is None else state_lookup[state.tobytes()].state_jacobian\n"
        " return NativeSensitivityWindowRequest(raw,c.document,clock,start,tangent,0,args.basis_duration)\n"
        "executor=NativeWindowExecutor(evaluate_trusted_native_sensitivity_request,workers=args.batch_workers)\n"
        "batch_adapter=NativeSensitivityBatchAdapter(batch_request,executor)",
    )
    source = _replace_once(
        source,
        "def checkpoint(theta,states,cost):",
        "def checkpoint(theta,states,cost):",
    )
    source = _replace_once(
        source,
        "options=MultipleShootingOptions(solver='slsqp',",
        "options=MultipleShootingOptions(segmented_forward_batch=batch_adapter,solver='slsqp',",
    )
    source = _replace_once(
        source,
        "(out/'returned-nodes.json').write_text(json.dumps({str(k):v.tolist() for k,v in fit.intermediate_states.items()},indent=2)+'\\n')",
        "(out/'returned-nodes.json').write_text(json.dumps({str(k):v.tolist() for k,v in fit.intermediate_states.items()},indent=2)+'\\n')\nexecutor.close()",
    )
    source = _replace_once(
        source,
        " global evaluation_count",
        " nonlocal evaluation_count",
    )
    entrypoint = "parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)"
    if source.count(entrypoint) != 1:
        raise ValueError("Expected exactly one top-level driver entry point")
    imports, body = source.split(entrypoint, maxsplit=1)
    source = (
        imports
        + "def main():\n "
        + entrypoint
        + body.replace("\n", "\n ")
        + "\n\nif __name__ == '__main__':\n main()\n"
    )
    args.output.write_text(source)


if __name__ == "__main__":
    main()

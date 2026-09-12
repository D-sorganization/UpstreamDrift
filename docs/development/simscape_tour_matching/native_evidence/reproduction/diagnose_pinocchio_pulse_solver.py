"""Compare Pinocchio constrained mobility with an independent dense KKT solve."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pinocchio as pin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("module", "spec", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    loader = importlib.util.spec_from_file_location("native_port", args.module)
    assert loader is not None and loader.loader is not None
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    spec = json.loads(args.spec.read_text())
    case = json.loads(args.baseline.read_text())
    model = module.NativePinocchioModel(spec)
    names = spec["coordinate_order"]
    coordinates = dict(zip(names, case["q"], strict=True))
    zeros = dict.fromkeys(names, 0.0)
    model.accelerations(coordinates, zeros, zeros)
    jacobian = pin.getConstraintJacobian(
        model.model, model.data, model.constraints[0], model.constraint_data[0]
    ).copy()
    q = model.configuration(coordinates)
    upper = pin.crba(model.model, model.data, q).copy()
    mass = np.triu(upper) + np.triu(upper, 1).T
    n, k = model.model.nv, jacobian.shape[0]
    kkt = np.block([[mass, jacobian.T], [jacobian, np.zeros((k, k))]])
    response = np.linalg.solve(kkt, np.vstack([np.eye(n), np.zeros((k, n))]))[:n]
    indices = [model.model.joints[model.model.getJointId(name)].idx_v for name in names]
    result = {
        "body_inertias": [
            {
                "joint": str(model.model.names[index]),
                "mass": float(inertia.mass),
                "com": inertia.lever.tolist(),
                "eigenvalues": np.linalg.eigvalsh(inertia.inertia).tolist(),
                "matrix": inertia.inertia.tolist(),
            }
            for index, inertia in enumerate(model.model.inertias)
        ],
        "mass_eigenvalues": np.linalg.eigvalsh(mass).tolist(),
        "constrained_mobility_eigenvalues": np.linalg.eigvalsh(response).tolist(),
        "dense_primitive_response": response[np.ix_(indices, indices)].tolist(),
        "kkt_residual_max": float(np.max(np.abs(jacobian @ response))),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Drift guard for calc_backend modules synchronized with Tools.
The baseline hashes were captured from the matching files in the sibling
Tools repository and should only change when that upstream source changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
TOOLS_BASELINE_HASHES: dict[str, str] = {
    "src/shared/python/calc_backend/__init__.py": "6623aefdab8848574f39098e42e2eff9bbf3da4101930ea769a81f006ce30620",
    "src/shared/python/calc_backend/app.py": "64bba44bf2005226a327627651afa35b5a6f2514ce0b511ff3857729bbcc5d0e",
    "src/shared/python/calc_backend/contracts/__init__.py": "3694802a72cc8dc97460d2b2d5b647b8c91132ecfc062e2551768929af93bdf0",
    "src/shared/python/calc_backend/contracts/acid_gas_dewpoint.py": "e682a54f598c63b4d23f68bb962096d264319e3b89ebbd97801f76453b36537d",
    "src/shared/python/calc_backend/contracts/baghouse.py": "039e25a4608294b10cf9bd3a88c081221646a660deb6521e0922ff204983c384",
    "src/shared/python/calc_backend/contracts/financial.py": "1dbf571256a7e800813d95625777d866f6eb54b10fe7297fcba01f7ad5b6889f",
    "src/shared/python/calc_backend/contracts/flare.py": "ec1185a8ab1d0d6a5f368c1e7047bb4886e1cc53fd57a8a23b49aae992f21144",
    "src/shared/python/calc_backend/contracts/flow_rate.py": "af3134daa4c96705fa5b81911b6f4825decc28e5b40d6641c75366a2271e316f",
    "src/shared/python/calc_backend/contracts/ode_solver.py": "24f78c0faa6987f072f9de0d6a1c85311279615dbcd23f3605c63e6879268392",
    "src/shared/python/calc_backend/contracts/pressure_drop.py": "8fad7c03d795d00ca352800a172143391dd656bfa639156f66c00906d7b0f136",
    "src/shared/python/calc_backend/contracts/rotation_converter.py": "8b0c70a1c68f20235ebbdb005b4eeca722f82ae2567e38716bec9510b2e20034",
    "src/shared/python/calc_backend/contracts/scrubber.py": "432a5773cf225f26f007649b7ba4a73f6883ccb0bc68c0dd432aace4ef7648f2",
    "src/shared/python/calc_backend/contracts/syngas_water.py": "3cba74c1aedd06018a47fb7175391a57ba029a82125c5772d3f77c3990e1ab7f",
    "src/shared/python/calc_backend/contracts/thermal_profile.py": "1e30310a202a0313d232ed5cda6aa88f776f2eefcf52b788a103690d3dd13d19",
    "src/shared/python/calc_backend/contracts/wgs_reactor.py": "7439830067b0323d2a42eeb45e8c168f106c99c8d5e171b48fce8a6d5aa3dae4",
    "src/shared/python/calc_backend/protocols.py": "71c3df86f0e32bc3a383d28a2a64b4f9bc0f601a4c11b0172f95f9fec6ad2a92",
    "src/shared/python/calc_backend/routers/__init__.py": "5bd0f5e9d92970d889ffea283713c8583b23ce895039aa536fcbc28246687ab5",
    "src/shared/python/calc_backend/routers/acid_gas_dewpoint.py": "101cb520acadd96ecf1176e2a6f6caa36f8c99d86c1ea83edece861d4b6e772d",
    "src/shared/python/calc_backend/routers/baghouse.py": "ae70fbd1fe2b45f9cbd3c32669371029d392100627670cd6ae4ea5a43b5cbd0d",
    "src/shared/python/calc_backend/routers/financial.py": "21938bbc24d7b24ed6b6a42510d6bba897be5ec822b47dd363f3c36581da6ff0",
    "src/shared/python/calc_backend/routers/flare.py": "e00028af6a438a04e57986b559511e043dfda9581dca913922818bab131a6872",
    "src/shared/python/calc_backend/routers/flow_rate.py": "8c70682473ef678d62eb54c2bf34a952a13022776aa43ec96c928f06467e76f8",
    "src/shared/python/calc_backend/routers/ode_solver.py": "6723900227fef37983ca24527488c6cc8841df0cbfb4b9714a8153171c89d9f6",
    "src/shared/python/calc_backend/routers/pressure_drop.py": "a3d9e758a8253189adb99cf8aeb46ea593d05442f35dc611fb23af4c6fd001d4",
    "src/shared/python/calc_backend/routers/rotation_converter.py": "713f4dc8663a40aa435afaac6196b29489d71e77e81f2f8c0b3d61c1d075ce73",
    "src/shared/python/calc_backend/routers/scrubber.py": "71e4c765f224c7c32709a6eefd6afba89c0bd0230134fe8be7562952a9e17fd9",
    "src/shared/python/calc_backend/routers/syngas_water.py": "eaba7f1dc95ed40599e14080d032f6c2bc6ecf73907a984d007649d71b580f89",
    "src/shared/python/calc_backend/routers/thermal_profile.py": "3d28f636092e51ee1bab9222e7b238e41963ba348a086881f9d6022d984b944a",
    "src/shared/python/calc_backend/routers/wgs_reactor.py": "de1c655923eae9609318da51e01e3105b6e5e02e8f110bddb9bd3101c01d61d9",
    "src/shared/python/calc_backend/tests/__init__.py": "43778dbd52483ada78a38cc962215e17682f5bfb2cd0937dd9c47ae466f92add",
    "src/shared/python/calc_backend/tests/test_calc_backend.py": "05ee41125504cd1ac9a0f80af8384b0e180da8598f920d7deda92e74c88b2e14",
    "src/shared/python/calc_backend/tests/test_calc_backend_gaps.py": "4a1bc192239e25573f872907a21e41b5a7cdd04791dc93544fd870564f852aeb",
}


def _runtime_equivalent_source(relative_path: str) -> bytes:
    """Return the source bytes that should match the Tools runtime baseline."""
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    return source.encode("utf-8")


@pytest.mark.parametrize(
    ("relative_path", "expected_sha256"),
    sorted(TOOLS_BASELINE_HASHES.items()),
)
def test_calc_backend_modules_match_tools_baseline(
    relative_path: str,
    expected_sha256: str,
) -> None:
    """Verify the selected leaf modules still match the Tools baseline."""
    path = REPO_ROOT / relative_path
    if not path.exists():
        pytest.fail(f"Missing file: {relative_path}")
    actual_sha256 = hashlib.sha256(
        _runtime_equivalent_source(relative_path)
    ).hexdigest()
    assert actual_sha256 == expected_sha256, relative_path

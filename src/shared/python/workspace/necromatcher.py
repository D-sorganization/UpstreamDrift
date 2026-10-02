"""Persistent historical-player library built on the workspace project spine.

Native model bytes are stored as candidates. Saving a model or authored control
profile never certifies reconstruction accuracy or native dynamics.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from zipfile import ZIP_STORED, ZipFile

from src.shared.python.core.contracts.exceptions import StateError
from src.shared.python.data_io.user_config_root import user_config_path
from src.shared.python.motion_matching.piecewise_polynomial import (
    PiecewisePolynomialTorque,
)
from .artifact_handoff import ArtifactKind, ArtifactReference, compute_file_sha256
from .necromatcher_capture import build_capture_archive
from .necromatcher_fit import FIT_SCHEMA, read_kinematic_fit
from .necromatcher_efforts import (
    EFFORT_SCHEMA,
    AuthoredEffortProfile,
    read_effort_profile,
    read_segments,
    validate_authored_provenance,
)
from .project_store import (
    DatasetMetadata,
    SessionMetadata,
    SessionProjectStore,
    SubjectMetadata,
)

_ARTIFACT_KINDS = {
    "native_model": ArtifactKind.MODEL,
    "torque_profile": ArtifactKind.DRIVING_PROFILE,
    "image_capture": ArtifactKind.OBSERVATION,
    "kinematic_fit": ArtifactKind.TRAJECTORY,
}


class NecromatcherLibrary:
    """Immutable asset versions with hash-checked recall and exclusive writes."""

    def __init__(self, root: Path | str) -> None:
        self._store = SessionProjectStore(root)
        self._store.load_project()

    @classmethod
    def create(cls, root: Path | str) -> NecromatcherLibrary:
        """Create a new library; never overwrite an existing project."""
        SessionProjectStore(root).create_project("necromatcher", "Necromatcher")
        return cls(root)

    @property
    def root(self) -> Path:
        return self._store.root

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        """Reject concurrent mutation rather than silently lose another writer."""
        path = self.root / ".necromatcher-write.lock"
        try:
            handle = path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise StateError(
                "Library write in progress; retry after writer finishes"
            ) from exc
        try:
            with handle:
                handle.write(str(os.getpid()))
                yield
        finally:
            path.unlink()

    def add_player(self, player_id: str, name: str) -> SubjectMetadata:
        """Register one historical player identity without replacement."""
        with self._write_lock():
            if player_id in self._store.load_project().subjects:
                raise StateError(f"Player already exists: {player_id}")
            return self._store.add_subject(player_id, name)

    def players(self) -> list[SubjectMetadata]:
        return sorted(
            self._store.load_project().subjects.values(), key=lambda x: x.subject_id
        )

    def add_swing(self, swing_id: str, player_id: str, name: str) -> SessionMetadata:
        """Create a swing/session under a registered player."""
        with self._write_lock():
            return self._store.create_session(swing_id, player_id, name)

    def swings(self, player_id: str | None = None) -> list[SessionMetadata]:
        return self._store.list_sessions(player_id)

    def assets(self, swing_id: str) -> list[DatasetMetadata]:
        self._store.load_session(swing_id)
        return self._store.list_datasets(swing_id)

    def load_asset(self, asset_id: str) -> DatasetMetadata:
        """Recall a version only when its immutable bytes still match."""
        asset = self._store.load_project().datasets[asset_id]
        reference = ArtifactReference(
            artifact_id=asset.dataset_id,
            path=asset.path,
            hash=asset.metadata["hash"],
            schema=asset.metadata["schema"],
            kind=_ARTIFACT_KINDS[asset.kind],
        )
        reference.verify_on_disk(self.root)
        return asset

    def _save_asset(
        self,
        asset_id: str,
        swing_id: str,
        source: Path,
        kind: str,
        metadata: dict[str, Any],
    ) -> DatasetMetadata:
        """Copy validated bytes before publishing metadata; roll back failed writes."""
        project = self._store.load_project()
        if asset_id in project.datasets:
            raise StateError(f"Asset version already exists: {asset_id}")
        self._store.load_session(swing_id)
        # Reuse the canonical identifier validation before composing a filesystem path.
        ArtifactReference(
            asset_id, str(source), "pending", metadata["schema"], _ARTIFACT_KINDS[kind]
        )
        destination = self.root / "assets" / f"{asset_id}{source.suffix}"
        destination.parent.mkdir(exist_ok=True)
        digest = compute_file_sha256(source)
        if metadata.get("hash", digest) != digest:
            raise ValueError("Source changed after validation")
        created = committed = False
        try:
            with source.open("rb") as incoming, destination.open("xb") as outgoing:
                created = True
                while chunk := incoming.read(65536):
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if compute_file_sha256(destination) != digest:
                raise ValueError("Source changed while copying asset")
            asset = self._store.register_dataset(
                asset_id,
                swing_id,
                destination,
                kind,
                metadata={**metadata, "hash": digest},
            )
            committed = True
            return asset
        finally:
            if created and not committed:
                destination.unlink(missing_ok=True)

    def add_model(
        self,
        model_id: str,
        swing_id: str,
        source: Path,
        *,
        engine: str,
        dofs: tuple[str, ...],
    ) -> DatasetMetadata:
        """Save native model bytes and ordered DOFs as an unqualified candidate."""
        if engine not in {"mujoco", "drake", "pinocchio", "opensim", "simscape"}:
            raise ValueError("Unsupported native model engine")
        if not dofs or len(set(dofs)) != len(dofs) or any(not x.strip() for x in dofs):
            raise ValueError("Model DOFs must be nonempty, named and unique")
        with self._write_lock():
            return self._save_asset(
                model_id,
                swing_id,
                source,
                "native_model",
                {
                    "schema": "necromatcher/native-model/1",
                    "engine": engine,
                    "dofs": list(dofs),
                    "qualification": "unqualified_candidate",
                },
            )

    def add_capture(
        self, capture_id: str, swing_id: str, source: Path
    ) -> DatasetMetadata:
        """Copy verified image evidence into a portable, hash-bound archive."""
        with self._write_lock():
            session = self._store.load_session(swing_id)
            with TemporaryDirectory(prefix="necromatcher-", dir=self.root) as directory:
                archive = Path(directory) / "capture.zip"
                try:
                    receipt = build_capture_archive(source, archive, session.subject_id)
                except (KeyError, TypeError) as exc:
                    raise ValueError("Malformed capture payload") from exc
                return self._save_asset(
                    capture_id,
                    swing_id,
                    archive,
                    "image_capture",
                    {
                        "schema": "necromatcher/image-capture/1",
                        "qualification": "image_observations_only",
                        "source_sha256": receipt["source"]["content_sha256"],
                        "frame_count": receipt["frame_count"],
                        "physical_time_verified": False,
                    },
                )

    def _read_profile(
        self, source: Path, swing_id: str
    ) -> tuple[dict[str, Any], PiecewisePolynomialTorque]:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Torque profile must be a JSON object")
        if payload.get("schema_version") != "necromatcher/torque-profile/1":
            raise ValueError("Unsupported torque-profile schema")
        model = self.load_asset(payload["model_id"])
        if model.kind != "native_model" or model.session_id != swing_id:
            raise ValueError("Profile model must belong to the same swing session")
        if payload.get("dofs") != model.metadata["dofs"]:
            raise ValueError("Profile DOF order must match the model version")
        if (
            payload.get("units") != "N*m"
            or payload.get("timebase") != "physical_seconds"
        ):
            raise ValueError("Profiles require N*m torques in physical_seconds")
        for asset in self.assets(swing_id):
            if (
                asset.kind == "kinematic_fit"
                and asset.metadata["model_id"] == model.dataset_id
            ):
                if "m" in self.load_fit(asset.dataset_id)["coordinate_units"]:
                    raise ValueError(
                        "Legacy torque units cannot represent model translation"
                    )
        validate_authored_provenance(payload)
        torque = read_segments(payload)
        if torque.n_channels != len(model.metadata["dofs"]):
            raise ValueError("Profile channel count must match model DOFs")
        return payload, torque

    def add_fit(self, fit_id: str, swing_id: str, source: Path) -> DatasetMetadata:
        """Preserve source-bound research samples without certifying dynamics."""
        with self._write_lock():
            source_hash = compute_file_sha256(source)
            payload = read_kinematic_fit(source, self, swing_id)
            return self._save_asset(
                fit_id,
                swing_id,
                source,
                "kinematic_fit",
                {
                    "schema": FIT_SCHEMA,
                    "hash": source_hash,
                    "model_id": payload["model_id"],
                    "capture_id": payload["capture_id"],
                    "frame_count": len(payload["frame_indices"]),
                    "qualification": "monocular_research_hypothesis",
                    "physical_time_qualified": False,
                    "dynamics_replayed": False,
                },
            )

    def load_fit(self, fit_id: str) -> dict[str, Any]:
        """Recall detached samples after verifying both immutable parent versions."""
        asset = self.load_asset(fit_id)
        if asset.kind != "kinematic_fit":
            raise ValueError("Fit recall requires a kinematic-fit asset")
        return read_kinematic_fit(Path(asset.path), self, asset.session_id)

    def add_profile(
        self, profile_id: str, swing_id: str, source: Path
    ) -> DatasetMetadata:
        """Save model-bound authored controls without claiming measured torque."""
        with self._write_lock():
            source_hash = compute_file_sha256(source)
            raw = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("schema_version") == EFFORT_SCHEMA:
                payload, _ = read_effort_profile(source, self, swing_id)
            else:
                payload, _ = self._read_profile(source, swing_id)
            return self._save_asset(
                profile_id,
                swing_id,
                source,
                "torque_profile",
                {
                    "schema": payload["schema_version"],
                    "model_id": payload["model_id"],
                    "qualification": "authored_controls",
                    "hash": source_hash,
                },
            )

    def load_torque(self, profile_id: str, model_id: str) -> PiecewisePolynomialTorque:
        """Recall checked controls for exactly the model revision they bind."""
        profile = self.load_asset(profile_id)
        if profile.kind != "torque_profile" or profile.metadata["model_id"] != model_id:
            raise ValueError("Profile is incompatible with requested model revision")
        if profile.metadata["schema"] == EFFORT_SCHEMA:
            raise ValueError(
                "Use load_effort_profile to preserve generalized effort units"
            )
        _, torque = self._read_profile(Path(profile.path), profile.session_id)
        return torque

    def load_effort_profile(
        self, profile_id: str, model_id: str
    ) -> AuthoredEffortProfile:
        """Recall authored generalized controls without discarding channel units."""
        profile = self.load_asset(profile_id)
        if profile.kind != "torque_profile" or profile.metadata["model_id"] != model_id:
            raise ValueError("Profile is incompatible with requested model revision")
        _, controls = read_effort_profile(Path(profile.path), self, profile.session_id)
        return controls

    def export_swing(self, swing_id: str, destination: Path) -> None:
        """Export checked versions and portable identities for downstream tools."""
        session = self._store.load_session(swing_id)
        player = self._store.load_project().subjects[session.subject_id]
        assets = [self.load_asset(x.dataset_id) for x in self.assets(swing_id)]
        for asset in assets:
            if asset.kind == "kinematic_fit":
                self.load_fit(asset.dataset_id)
            elif asset.kind == "torque_profile":
                if asset.metadata["schema"] == EFFORT_SCHEMA:
                    self.load_effort_profile(
                        asset.dataset_id, asset.metadata["model_id"]
                    )
                else:
                    self.load_torque(asset.dataset_id, asset.metadata["model_id"])
        manifest = {
            "schema_version": "necromatcher/swing-package/1",
            "player": asdict(player),
            "swing": asdict(session),
            "assets": [
                {**asdict(x), "path": f"assets/{Path(x.path).name}"} for x in assets
            ],
        }
        with TemporaryDirectory(
            prefix="necromatcher-export-", dir=destination.parent
        ) as directory:
            temporary = Path(directory) / "swing.zip"
            with ZipFile(temporary, "x", compression=ZIP_STORED) as archive:
                archive.writestr(
                    "manifest.json", json.dumps(manifest, allow_nan=False, indent=2)
                )
                for asset in assets:
                    member = f"assets/{Path(asset.path).name}"
                    archive.write(asset.path, member)
            with ZipFile(temporary, "r") as archive:
                for asset in assets:
                    member = f"assets/{Path(asset.path).name}"
                    digest = hashlib.sha256()
                    with archive.open(member) as copied:
                        while chunk := copied.read(65536):
                            digest.update(chunk)
                    if digest.hexdigest() != asset.metadata["hash"].removeprefix(
                        "sha256:"
                    ):
                        raise ValueError("Export asset hash mismatch")
            # Same-filesystem atomic publication, with no overwrite of an existing export.
            os.link(temporary, destination)


@lru_cache(maxsize=1)
def default_necromatcher_library() -> NecromatcherLibrary:
    """Open the one configured local library for every application surface."""
    configured = os.environ.get("NECROMATCHER_LIBRARY_ROOT")
    root = Path(configured) if configured else user_config_path("necromatcher")
    try:
        return NecromatcherLibrary(root)
    except KeyError:
        try:
            return NecromatcherLibrary.create(root)
        except StateError:
            return NecromatcherLibrary(root)

"""Verified, portable image-observation archives for Necromatcher."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile

import numpy as np

from src.shared.python.shadow_tracker.historical_capture import (
    CaptureWindow,
    observation_record,
)
from src.shared.python.shadow_tracker.ingestion import compute_frame_hash
from src.shared.python.shadow_tracker.source_records import FrameIdentity, SourceAsset
from src.shared.python.pose_estimation.interface import PoseEstimationResult


def _check_observation(row: dict[str, Any]) -> None:
    observation = row["observation"]
    if observation.get("coordinate_system") != "normalized_image_xy":
        raise ValueError("Capture requires normalized image observations")
    if observation.get("physical_time_s") is not None:
        raise ValueError("Image observations cannot certify physical time")
    points, visibility = {}, {}
    for name, landmark in observation["landmarks"].items():
        if set(landmark) != {"x", "y", "visibility"}:
            raise ValueError("Only image XY and visibility are capture evidence")
        points[name] = np.array([landmark["x"], landmark["y"]])
        if landmark["visibility"] is not None:
            visibility[name] = landmark["visibility"]
    checked = observation_record(
        PoseEstimationResult(
            {},
            observation["confidence"],
            0,
            raw_keypoints=points,
            raw_confidences=visibility,
        )
    )
    if checked["status"] != observation["status"]:
        raise ValueError("Detection status disagrees with landmarks")


def _check_image(
    source: Path, row: dict[str, Any], frame: FrameIdentity, asset: SourceAsset
) -> bytes:
    import cv2

    name = row["image"]
    if name != f"{frame.frame_id}.png" or Path(name).name != name:
        raise ValueError("Frame image must be a local PNG named by its frame identity")
    path = source / name
    if path.is_symlink() or path.resolve().parent != source.resolve():
        raise ValueError("Frame image must remain inside capture directory")
    image_bytes = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (asset.height_px, asset.width_px):
        raise ValueError("Frame image dimensions do not match source")
    if (
        compute_frame_hash(image.tobytes(), decoder_name=frame.decoder_name)
        != frame.frame_sha256
    ):
        raise ValueError("Frame image hash mismatch")
    return image_bytes


def build_capture_archive(
    source: Path, destination: Path, subject_id: str
) -> dict[str, Any]:
    """Stream validated capture evidence into a ZIP without extracting paths.

    Each archived PNG is checked against its decoded BGR identity. Observations
    retain exact container PTS and their original byte hash. No physical-motion
    qualification is introduced by storing or exporting this archive.
    """
    receipt_bytes = (source / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt.get("schema_version") != "historical-capture/1.0.0":
        raise ValueError("Unsupported capture receipt schema")
    if receipt.get("subject_id") != subject_id:
        raise ValueError("Capture subject must match the swing's player")
    if (
        receipt.get("qualification") != "image_observations_only"
        or receipt.get("physical_time_verified") is not False
    ):
        raise ValueError("Capture import accepts only unqualified image observations")
    asset = SourceAsset.from_dict(receipt["source"])
    window = CaptureWindow(
        start_s=receipt["window_presentation_s"][0],
        end_s=receipt["window_presentation_s"][1],
    )
    digest = hashlib.sha256()
    count = detected = 0
    previous: Fraction | None = None
    names: set[str] = set()
    with ZipFile(destination, "x", compression=ZIP_STORED) as archive:
        archive.writestr("receipt.json", receipt_bytes)
        archive.write(source / "observations.jsonl", "observations.jsonl")
        with archive.open("observations.jsonl", "r") as incoming:
            for line in incoming:
                row = json.loads(line)
                frame = FrameIdentity.from_dict(row["frame"])
                if (
                    frame.asset_id != asset.asset_id
                    or not frame.is_timing_exact
                    or frame.timing_mode != "container_pts"
                ):
                    raise ValueError(
                        "Frame must bind exact container PTS to its source"
                    )
                if frame.physical_time_s is not None or not window.contains(
                    frame.presentation_time
                ):
                    raise ValueError("Frame clock is outside the capture contract")
                if previous is not None and frame.presentation_time <= previous:
                    raise ValueError("Capture timestamps must strictly increase")
                if row["image"] in names:
                    raise ValueError("Duplicate capture image identity")
                _check_observation(row)
                archive.writestr(row["image"], _check_image(source, row, frame, asset))
                names.add(row["image"])
                previous = frame.presentation_time
                digest.update(line)
                count += 1
                detected += row["observation"]["status"] == "detected"
    if digest.hexdigest() != receipt["observations_sha256"]:
        raise ValueError("Capture observations hash mismatch")
    if (
        not count
        or count != receipt["frame_count"]
        or detected != receipt["detected_count"]
    ):
        raise ValueError("Capture counts disagree with receipt")
    return receipt

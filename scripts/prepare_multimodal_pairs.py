#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare one camera/FC/LiDAR training row per measured LiDAR sample.

The output directory is intentionally outside the repository: it contains
flight images and should only be uploaded as a private Kaggle dataset.

Usage:
  python scripts/prepare_multimodal_pairs.py \
    C:/path/to/science_work --out C:/path/to/kaggle_multimodal_data
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


SESSIONS = ("20250918_lidar", "20250918_lidar_evening")


def read_timed_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime")
    return frame.drop_duplicates("datetime", keep="first").reset_index(drop=True)


def session_pairs(root: Path, name: str, out_images: Path) -> pd.DataFrame:
    folder = root / name
    lidar = read_timed_csv(folder / "lidar_data.csv")
    lidar["dist"] = pd.to_numeric(lidar["dist"], errors="coerce")
    lidar = lidar.dropna(subset=["dist"]).rename(columns={"datetime": "lidar_datetime"})

    paths = sorted(
        p for p in (folder / "nadir_camera").iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    camera = pd.DataFrame({
        "source_image": paths,
        "camera_datetime": [
            pd.to_datetime(p.stem, format="%Y-%m-%d_%H-%M-%S_%f", errors="coerce")
            for p in paths
        ],
    }).dropna(subset=["camera_datetime"]).sort_values("camera_datetime")

    paired = pd.merge_asof(
        lidar.sort_values("lidar_datetime"), camera.sort_values("camera_datetime"),
        left_on="lidar_datetime", right_on="camera_datetime", direction="nearest",
        tolerance=pd.Timedelta("250ms"),
    ).dropna(subset=["source_image"]).copy()
    paired["camera_offset_s"] = (
        paired["camera_datetime"] - paired["lidar_datetime"]
    ).dt.total_seconds()

    fc = read_timed_csv(folder / "fc_data.csv")
    for column in fc.columns:
        if column != "datetime":
            fc[column] = pd.to_numeric(fc[column], errors="coerce")
    fc = fc.rename(columns={"datetime": "fc_datetime"})
    paired = pd.merge_asof(
        paired.sort_values("lidar_datetime"), fc.sort_values("fc_datetime"),
        left_on="lidar_datetime", right_on="fc_datetime", direction="nearest",
        tolerance=pd.Timedelta("150ms"), suffixes=("", "_fc"),
    )
    paired = paired.dropna(subset=["fc_datetime", "relative_alt"]).copy()
    paired["fc_offset_s"] = (paired["fc_datetime"] - paired["lidar_datetime"]).dt.total_seconds()

    # Use a circular representation of yaw to avoid the 0/360 degree seam.
    yaw = pd.to_numeric(paired.get("yaw"), errors="coerce")
    paired["yaw_sin"] = np.sin(np.deg2rad(yaw))
    paired["yaw_cos"] = np.cos(np.deg2rad(yaw))

    # Keep a flag from the existing statistical LiDAR audit; it is not treated
    # as proof that a label is wrong and remains available for separate metrics.
    quality_path = root / "drone-distance-study" / "results" / "target_audit" / f"{name}_lidar_quality.csv"
    if quality_path.exists():
        quality = read_timed_csv(quality_path)[["datetime", "large_jump"]]
        paired = paired.merge(
            quality.rename(columns={"datetime": "lidar_datetime"}),
            on="lidar_datetime", how="left",
        )
        paired["large_jump"] = (
            paired["large_jump"].astype(str).str.strip().str.lower().eq("true")
        )
    else:
        paired["large_jump"] = False

    paired["flight"] = name
    paired["label_dist_m"] = paired["dist"].astype(float)
    paired["label_at_sensor_floor"] = np.isclose(paired["label_dist_m"], 0.1, atol=1e-6)

    out_images.mkdir(parents=True, exist_ok=True)
    names = []
    for path in paired["source_image"]:
        new_name = f"{name}__{Path(path).name}"
        shutil.copy2(path, out_images / new_name)
        names.append(new_name)
    paired["image_filename"] = names

    columns = [
        "image_filename", "flight", "lidar_datetime", "camera_datetime", "camera_offset_s",
        "fc_datetime", "fc_offset_s", "label_dist_m", "label_at_sensor_floor", "large_jump",
        "relative_alt", "roll", "pitch", "yaw_sin", "yaw_cos", "xacc", "yacc", "zacc",
    ]
    for column in columns:
        if column not in paired:
            paired[column] = np.nan
    return paired[columns].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Folder containing the two flight folders")
    parser.add_argument("--out", type=Path, required=True, help="Private Kaggle dataset folder")
    args = parser.parse_args()
    root, out = args.root.expanduser().resolve(), args.out.expanduser().resolve()
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    frames = [session_pairs(root, session, image_dir) for session in SESSIONS]
    pairs = pd.concat(frames, ignore_index=True)
    pairs.to_csv(out / "labels.csv", index=False, encoding="utf-8")
    (out / "README.txt").write_text(
        "Private sample prepared for multimodal depth-estimation experiments.\n"
        "One row per LiDAR measurement with a camera frame within 0.25 s and FC telemetry within 0.15 s.\n"
        "Target is lidar dist; relative_alt is an input feature only.\n"
        "large_jump is an audit flag, not a confirmed label error. label_at_sensor_floor marks dist=0.1 m.\n",
        encoding="utf-8",
    )
    print(f"Rows: {len(pairs)}")
    print("By flight:", pairs.groupby("flight").size().to_dict())
    print("dist=0.1 m:", int(pairs["label_at_sensor_floor"].sum()))
    print("Flagged jumps:", int(pairs["large_jump"].fillna(False).astype(bool).sum()))
    print("Images:", len(list(image_dir.iterdir())))
    print(f"Output: {out}")


if __name__ == "__main__":
    main()

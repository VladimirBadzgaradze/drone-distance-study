# Drone distance study

Preliminary analysis of nadir-camera frames, flight-controller telemetry, and lidar range measurements from two UAV recordings made on 18 September 2025.

## Key files

- [`report_for_supervisor.md`](report_for_supervisor.md) — short progress note ready to send to the research supervisor.
- [`dataset_report.md`](dataset_report.md) — fuller dataset review, statistics, caveats, and next steps.
- `scripts/` — exploratory dataset analysis and lidar/camera synchronization audits.
- `results/` — aggregate summaries and derived audit tables. These are analysis outputs, not the original image or telemetry folders.
- `figures/` — plots and a contact sheet used during visual review.

The original datasets are not included: the source `fc_data.csv`, `lidar_data.csv`, and camera images stay local. The repository contains only derived audit tables, aggregate summaries, and figures. In particular, `dist` is the target measurement and `relative_alt` is retained only as a comparison signal.

## Reproduce the analysis

Install the dependencies listed in [`requirements.txt`](requirements.txt). Place the original session folders next to this README, keeping their original names and structure. They are ignored by Git:

```text
drone-distance-study/
├── 20250918_lidar/
├── 20250918_lidar_evening/
└── scripts/
```

Run from the project root:

```powershell
python scripts/analyze_drone_dataset.py . --out results/eda
python scripts/audit_drone_target.py . --out results/target_audit
python scripts/audit_evening_focus.py
```

The focus audit expects `20250918_lidar_evening/` beside the scripts directory and writes its derived outputs to `results/target_audit/` in this project copy.

## Current limits

The work checks data structure, sampling rates, approximate timestamp alignment, and label behavior. It does not validate lidar accuracy against a surveyed ground truth, identify which physical surface produced each return, or evaluate a trained computer-vision model. Statistical jump flags indicate locations to inspect; they are not confirmed sensor errors.

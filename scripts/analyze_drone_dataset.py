#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA для датасета дрона: CSV полётного контроллера, лидар и JPEG-кадры.

Ожидаемая структура:
ROOT/
  20250918_lidar/
    fc_data.csv
    lidar_data.csv
    nadir_camera/*.jpeg
  20250918_lidar_evening/
    fc_data.csv
    lidar_data.csv
    nadir_camera/*.jpeg

Зависимости:
    pip install pandas numpy matplotlib pillow

Запуск:
    python analyze_drone_dataset.py "D:/path/to/ROOT"
или из папки, внутри которой лежат обе папки:
    python analyze_drone_dataset.py .

Результаты будут в папке dataset_analysis рядом с ROOT (либо путь можно задать
аргументом --out). Изображения не копируются и никуда не загружаются.
"""

from __future__ import annotations

import argparse
import math
import statistics
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from PIL import Image, ImageStat, ImageFilter
except ImportError as exc:
    raise SystemExit("Не найдена Pillow. Установи зависимости: pip install pandas numpy matplotlib pillow") from exc


TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d_%H-%M-%S_%f",
    "%Y-%m-%d_%H-%M-%S",
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse timestamps, tolerating fractional seconds and minor format differences."""
    return pd.to_datetime(series.astype(str).str.strip(), errors="coerce")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def write_line(handle, text=""):
    handle.write(str(text) + "\n")


def describe_intervals(times: pd.Series, source: str, session: str) -> dict:
    t = pd.to_datetime(times, errors="coerce").dropna().sort_values()
    if len(t) < 2:
        return {
            "session": session, "source": source, "n_timestamps": len(t),
            "start": t.min() if len(t) else pd.NaT,
            "end": t.max() if len(t) else pd.NaT,
            "duration_s": np.nan, "median_interval_s": np.nan,
            "mean_interval_s": np.nan, "std_interval_s": np.nan,
            "min_interval_s": np.nan, "max_interval_s": np.nan,
            "estimated_rate_hz": np.nan, "intervals_over_1_5x_median": np.nan,
            "duplicate_timestamps": int(t.duplicated().sum()),
        }
    diffs = t.diff().dt.total_seconds().dropna()
    positive = diffs[diffs > 0]
    med = float(positive.median()) if len(positive) else np.nan
    return {
        "session": session,
        "source": source,
        "n_timestamps": len(t),
        "start": t.min(),
        "end": t.max(),
        "duration_s": float((t.max() - t.min()).total_seconds()),
        "median_interval_s": med,
        "mean_interval_s": float(positive.mean()) if len(positive) else np.nan,
        "std_interval_s": float(positive.std()) if len(positive) > 1 else np.nan,
        "min_interval_s": float(positive.min()) if len(positive) else np.nan,
        "max_interval_s": float(positive.max()) if len(positive) else np.nan,
        "estimated_rate_hz": (1.0 / med) if med and med > 0 else np.nan,
        "intervals_over_1_5x_median": int((positive > 1.5 * med).sum()) if med and med > 0 else np.nan,
        "duplicate_timestamps": int(t.duplicated().sum()),
    }


def timestamp_from_filename(path: Path):
    stem = path.stem
    for fmt in TIME_FORMATS:
        try:
            return pd.Timestamp(pd.to_datetime(stem, format=fmt, errors="raise"))
        except Exception:
            pass
    # Flexible fallback for filenames that contain a timestamp with separators.
    try:
        return pd.to_datetime(stem, errors="raise")
    except Exception:
        return pd.NaT


def image_file_list(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(
        (p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name
    )


def sample_evenly(items: list, maximum: int) -> list:
    if len(items) <= maximum:
        return items
    idx = np.linspace(0, len(items) - 1, maximum).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def image_metrics(path: Path) -> dict:
    result = {
        "filename": path.name,
        "datetime": timestamp_from_filename(path),
        "width": np.nan, "height": np.nan, "mode": "",
        "file_size_bytes": np.nan, "brightness_mean": np.nan,
        "brightness_std": np.nan, "contrast_gray_std": np.nan,
        "edge_mean": np.nan, "edge_std": np.nan,
        "read_error": "",
    }
    try:
        result["file_size_bytes"] = path.stat().st_size
        with Image.open(path) as im:
            result["width"], result["height"] = im.size
            result["mode"] = im.mode
            gray = im.convert("L")
            # Resize for quick, comparable image metrics.
            gray.thumbnail((512, 512))
            arr = np.asarray(gray, dtype=np.float32)
            result["brightness_mean"] = float(arr.mean())
            result["brightness_std"] = float(arr.std())
            result["contrast_gray_std"] = float(arr.std())
            edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
            result["edge_mean"] = float(edges.mean())
            result["edge_std"] = float(edges.std())
    except Exception as exc:
        result["read_error"] = f"{type(exc).__name__}: {exc}"
    return result


def plot_timeseries(df: pd.DataFrame, time_col: str, columns: list[str], title: str, out_path: Path):
    cols = [c for c in columns if c in df.columns and df[c].notna().any()]
    if not cols or time_col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in cols:
        ax.plot(df[time_col], pd.to_numeric(df[col], errors="coerce"), label=col, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Время")
    ax.set_ylabel("Значение")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_distribution(df: pd.DataFrame, column: str, title: str, out_path: Path):
    if column not in df.columns:
        return
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=min(60, max(10, int(np.sqrt(len(values)))),), edgecolor="black", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Количество")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_intervals(times: pd.Series, title: str, out_path: Path):
    t = pd.to_datetime(times, errors="coerce").dropna().sort_values()
    if len(t) < 2:
        return
    d = t.diff().dt.total_seconds().dropna()
    d = d[d > 0]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(d, bins=50, edgecolor="black", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel("Интервал между соседними событиями, с")
    ax.set_ylabel("Количество")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def describe_dataframe(df: pd.DataFrame, session: str, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-column stats and per-column missingness/constant diagnostics."""
    rows = []
    diagnostics = []
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        row = {
            "session": session, "source": source, "column": col,
            "dtype": str(s.dtype), "rows": len(s), "missing_count": missing,
            "missing_pct": (100 * missing / len(s)) if len(s) else np.nan,
            "unique_non_null": int(s.nunique(dropna=True)),
            "constant_non_null": bool(s.nunique(dropna=True) <= 1 and s.notna().any()),
        }
        if pd.api.types.is_numeric_dtype(s):
            numeric = pd.to_numeric(s, errors="coerce").dropna()
            row.update({
                "min": numeric.min() if len(numeric) else np.nan,
                "q01": numeric.quantile(0.01) if len(numeric) else np.nan,
                "q05": numeric.quantile(0.05) if len(numeric) else np.nan,
                "median": numeric.median() if len(numeric) else np.nan,
                "mean": numeric.mean() if len(numeric) else np.nan,
                "std": numeric.std() if len(numeric) > 1 else np.nan,
                "q95": numeric.quantile(0.95) if len(numeric) else np.nan,
                "q99": numeric.quantile(0.99) if len(numeric) else np.nan,
                "max": numeric.max() if len(numeric) else np.nan,
            })
        else:
            row.update({"min": np.nan, "q01": np.nan, "q05": np.nan, "median": np.nan,
                        "mean": np.nan, "std": np.nan, "q95": np.nan, "q99": np.nan, "max": np.nan})
        rows.append(row)
        diagnostics.append({
            "session": session, "source": source, "column": col,
            "dtype": str(s.dtype), "missing_count": missing,
            "missing_pct": row["missing_pct"], "unique_non_null": row["unique_non_null"],
            "constant_non_null": row["constant_non_null"],
            "most_common_value": str(s.mode(dropna=True).iloc[0]) if s.notna().any() and not s.mode(dropna=True).empty else "",
            "most_common_count": int(s.value_counts(dropna=True).iloc[0]) if s.notna().any() else 0,
        })
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def load_csv(path: Path, time_col: str = "datetime") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if time_col in df.columns:
        df[time_col] = parse_datetime_series(df[time_col])
    for col in df.columns:
        if col != time_col:
            # Convert columns that are numeric except for a few malformed values.
            converted = pd.to_numeric(df[col], errors="coerce")
            original_non_null = df[col].notna().sum()
            if original_non_null and converted.notna().sum() >= 0.8 * original_non_null:
                df[col] = converted
    return df


def main():
    parser = argparse.ArgumentParser(description="Сбор EDA-отчёта для датасета дрона")
    parser.add_argument("root", nargs="?", default=".", help="Папка, внутри которой находятся папки записей")
    parser.add_argument("--out", default=None, help="Папка для результатов (по умолчанию ROOT/dataset_analysis)")
    parser.add_argument("--max-images", type=int, default=1000, help="Максимум изображений для расчёта метрик качества на одну запись")
    parser.add_argument("--image-step", type=int, default=0, help="Если >0, анализировать каждый N-й кадр вместо равномерной выборки")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Папка не найдена: {root}")
    out_root = Path(args.out).expanduser().resolve() if args.out else root / "dataset_analysis"
    out_root.mkdir(parents=True, exist_ok=True)
    plots_root = out_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    sessions = []
    # Find folders containing both CSVs, while avoiding the output directory.
    for fc_path in root.rglob("fc_data.csv"):
        if out_root == fc_path.parent or out_root in fc_path.parents:
            continue
        folder = fc_path.parent
        lidar_path = folder / "lidar_data.csv"
        if lidar_path.exists():
            sessions.append((folder, fc_path, lidar_path))
    sessions.sort(key=lambda x: str(x[0]))

    if not sessions:
        raise SystemExit(
            f"Не нашёл пары fc_data.csv + lidar_data.csv внутри {root}.\n"
            "Укажи корневую папку, внутри которой лежат обе папки записей."
        )

    all_stats, all_diag, all_timing, all_corr, all_sync, all_img_metrics = [], [], [], [], [], []
    summary_path = out_root / "summary.txt"

    with summary_path.open("w", encoding="utf-8") as summary:
        write_line(summary, "EDA-ОТЧЁТ ПО ДАТАСЕТУ ДРОНА")
        write_line(summary, f"Корневая папка: {root}")
        write_line(summary, f"Записей найдено: {len(sessions)}")
        write_line(summary, f"Папка результатов: {out_root}")
        write_line(summary, "")
        write_line(summary, "Примечание: метрики яркости/резкости — эвристики, не оценка пригодности кадра сами по себе.")
        write_line(summary, "Лидар dist и FC relative_alt могут измерять разные величины; корреляция не доказывает их эквивалентность.")
        write_line(summary, "")

        for folder, fc_path, lidar_path in sessions:
            session = folder.name
            session_plot_dir = plots_root / safe_name(session)
            session_plot_dir.mkdir(parents=True, exist_ok=True)
            write_line(summary, "=" * 80)
            write_line(summary, f"ЗАПИСЬ: {session}")
            write_line(summary, f"Папка: {folder}")

            try:
                fc = load_csv(fc_path)
            except Exception as exc:
                write_line(summary, f"ОШИБКА чтения {fc_path}: {exc}")
                continue
            try:
                lidar = load_csv(lidar_path)
            except Exception as exc:
                write_line(summary, f"ОШИБКА чтения {lidar_path}: {exc}")
                continue

            image_dir = folder / "nadir_camera"
            images = image_file_list(image_dir)
            image_times_all = pd.Series([timestamp_from_filename(p) for p in images], name="datetime")
            image_times_all = pd.to_datetime(image_times_all, errors="coerce")
            image_times_valid = image_times_all.dropna().sort_values()

            write_line(summary, f"FC: {len(fc):,} строк, {len(fc.columns)} столбцов")
            write_line(summary, f"Lidar: {len(lidar):,} строк, {len(lidar.columns)} столбцов")
            write_line(summary, f"Изображений найдено: {len(images):,}")
            write_line(summary, f"Изображений с распознанным временем в имени: {len(image_times_valid):,}")
            write_line(summary, "")

            # Column-level stats / missingness / constants.
            for source, df in (("fc", fc), ("lidar", lidar)):
                stats, diag = describe_dataframe(df, session, source)
                all_stats.append(stats)
                all_diag.append(diag)

            # Timestamp diagnostics.
            timing_rows = []
            if "datetime" in fc.columns:
                timing_rows.append(describe_intervals(fc["datetime"], "fc", session))
            if "datetime" in lidar.columns:
                timing_rows.append(describe_intervals(lidar["datetime"], "lidar", session))
            timing_rows.append(describe_intervals(image_times_valid, "camera", session))
            all_timing.extend(timing_rows)
            for r in timing_rows:
                write_line(summary, f"{r['source']}: n={r['n_timestamps']}, start={r['start']}, end={r['end']}, "
                                    f"duration={r['duration_s']:.2f}s, median dt={r['median_interval_s']:.5f}s, "
                                    f"rate≈{r['estimated_rate_hz']:.3f} Hz, duplicate timestamps={r['duplicate_timestamps']}")

            # Data time span overlap.
            valid_starts, valid_ends = [], []
            for ts in [fc.get("datetime", pd.Series(dtype="datetime64[ns]")),
                       lidar.get("datetime", pd.Series(dtype="datetime64[ns]")),
                       image_times_valid]:
                ts = pd.to_datetime(ts, errors="coerce").dropna()
                if len(ts):
                    valid_starts.append(ts.min())
                    valid_ends.append(ts.max())
            if valid_starts:
                overlap_start, overlap_end = max(valid_starts), min(valid_ends)
                overlap_s = max(0.0, (overlap_end - overlap_start).total_seconds())
                write_line(summary, f"Общий временной диапазон всех трёх источников: {overlap_s:.2f} с "
                                    f"(начало {overlap_start}, конец {overlap_end})")
            write_line(summary, "")

            # Numeric distributions / constants.
            for source, df in (("fc", fc), ("lidar", lidar)):
                for col in numeric_columns(df):
                    vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(vals):
                        write_line(summary, f"{source}.{col}: n={len(vals)}, min={vals.min():.6g}, "
                                            f"q01={vals.quantile(.01):.6g}, q05={vals.quantile(.05):.6g}, "
                                            f"median={vals.median():.6g}, mean={vals.mean():.6g}, "
                                            f"std={vals.std():.6g}, q95={vals.quantile(.95):.6g}, "
                                            f"q99={vals.quantile(.99):.6g}, max={vals.max():.6g}")
            write_line(summary, "")

            # Correlation matrix for numeric FC + lidar fields, and alt/dist correlation.
            merged_values = None
            if "datetime" in fc.columns and "datetime" in lidar.columns:
                fcv = fc.dropna(subset=["datetime"]).sort_values("datetime")
                liv = lidar.dropna(subset=["datetime"]).sort_values("datetime")
                if len(fcv) and len(liv):
                    # Nearest lidar observation for each FC row; tolerance intentionally generous (0.6 s).
                    joined = pd.merge_asof(
                        fcv, liv, on="datetime", direction="nearest",
                        tolerance=pd.Timedelta(seconds=0.6), suffixes=("_fc", "_lidar")
                    )
                    for col in ("relative_alt", "dist"):
                        if col not in joined.columns:
                            # Handle suffixing if a column name unexpectedly overlaps.
                            for candidate in (f"{col}_fc", f"{col}_lidar"):
                                if candidate in joined.columns:
                                    joined[col] = joined[candidate]
                                    break
                    if "relative_alt" in joined.columns and "dist" in joined.columns:
                        paired = joined[["datetime", "relative_alt", "dist"]].dropna()
                        if len(paired) >= 2:
                            corr = paired["relative_alt"].corr(paired["dist"])
                            all_corr.append({
                                "session": session, "x": "relative_alt", "y": "dist",
                                "pearson_correlation": corr, "n_pairs": len(paired),
                                "alignment": "nearest lidar to FC timestamp, tolerance 0.6 s",
                            })
                            write_line(summary, f"Корреляция relative_alt vs dist после грубого nearest-match: "
                                                f"Pearson r={corr:.5f}, пар={len(paired)}")
                            merged_values = paired
                    # Time offset from each camera frame to nearest FC and lidar timestamp.
                    if len(image_times_valid):
                        cam_df = pd.DataFrame({"camera_datetime": image_times_valid})
                        fc_times = pd.DataFrame({"fc_datetime": fcv["datetime"].dropna().sort_values()})
                        lidar_times = pd.DataFrame({"lidar_datetime": liv["datetime"].dropna().sort_values()})
                        cam_fc = pd.merge_asof(
                            cam_df.sort_values("camera_datetime"), fc_times,
                            left_on="camera_datetime", right_on="fc_datetime",
                            direction="nearest", tolerance=pd.Timedelta(seconds=0.2)
                        )
                        cam_fc["camera_to_fc_offset_s"] = (
                            cam_fc["fc_datetime"] - cam_fc["camera_datetime"]
                        ).dt.total_seconds()
                        cam_lidar = pd.merge_asof(
                            cam_df.sort_values("camera_datetime"), lidar_times,
                            left_on="camera_datetime", right_on="lidar_datetime",
                            direction="nearest", tolerance=pd.Timedelta(seconds=0.6)
                        )
                        cam_lidar["camera_to_lidar_offset_s"] = (
                            cam_lidar["lidar_datetime"] - cam_lidar["camera_datetime"]
                        ).dt.total_seconds()
                        sync = cam_df.copy()
                        sync = sync.merge(cam_fc[["camera_datetime", "fc_datetime", "camera_to_fc_offset_s"]],
                                          on="camera_datetime", how="left")
                        sync = sync.merge(cam_lidar[["camera_datetime", "lidar_datetime", "camera_to_lidar_offset_s"]],
                                          on="camera_datetime", how="left")
                        sync.insert(0, "session", session)
                        all_sync.append(sync)
                        write_line(summary, f"Синхронизация камеры → FC (допуск 0.2с): "
                                            f"совпадений {sync['fc_datetime'].notna().sum()}/{len(sync)}; "
                                            f"медианный |offset|={sync['camera_to_fc_offset_s'].abs().median():.5f}s")
                        write_line(summary, f"Синхронизация камеры → lidar (допуск 0.6с): "
                                            f"совпадений {sync['lidar_datetime'].notna().sum()}/{len(sync)}; "
                                            f"медианный |offset|={sync['camera_to_lidar_offset_s'].abs().median():.5f}s")

            # Correlations within each original CSV.
            for source, df in (("fc", fc), ("lidar", lidar)):
                nums = df.select_dtypes(include=[np.number])
                if len(nums.columns) >= 2:
                    cm = nums.corr(numeric_only=True)
                    for a in cm.columns:
                        for b in cm.columns:
                            if a < b and pd.notna(cm.loc[a, b]):
                                all_corr.append({
                                    "session": session, "x": a, "y": b,
                                    "pearson_correlation": cm.loc[a, b],
                                    "n_pairs": int(df[[a, b]].dropna().shape[0]),
                                    "alignment": f"within {source}_data.csv",
                                })

            # Plots.
            if "datetime" in fc.columns:
                plot_timeseries(fc, "datetime", ["relative_alt", "alt", "vx", "vy", "vz", "roll", "pitch"],
                                f"{session}: параметры FC", session_plot_dir / "fc_timeseries.png")
                plot_intervals(fc["datetime"], f"{session}: интервалы FC", session_plot_dir / "fc_intervals.png")
                for col in ["relative_alt", "alt", "vx", "vy", "vz", "roll", "pitch", "yaw"]:
                    if col in fc.columns and pd.to_numeric(fc[col], errors="coerce").notna().any():
                        plot_distribution(fc, col, f"{session}: распределение {col}",
                                          session_plot_dir / f"fc_distribution_{safe_name(col)}.png")
            if "datetime" in lidar.columns:
                plot_timeseries(lidar, "datetime", ["dist"], f"{session}: расстояние лидара", session_plot_dir / "lidar_timeseries.png")
                plot_intervals(lidar["datetime"], f"{session}: интервалы лидара", session_plot_dir / "lidar_intervals.png")
                plot_distribution(lidar, "dist", f"{session}: распределение dist", session_plot_dir / "lidar_dist_distribution.png")
            plot_intervals(image_times_valid, f"{session}: интервалы камеры", session_plot_dir / "camera_intervals.png")
            if merged_values is not None and len(merged_values):
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(merged_values["relative_alt"], merged_values["dist"], s=12, alpha=0.6)
                ax.set_title(f"{session}: relative_alt vs dist (nearest match)")
                ax.set_xlabel("relative_alt")
                ax.set_ylabel("dist")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(session_plot_dir / "relative_alt_vs_dist.png", dpi=140)
                plt.close(fig)

            # Sample image metrics. This does not copy images.
            if args.image_step > 0:
                sampled_images = images[::args.image_step]
            else:
                sampled_images = sample_evenly(images, max(1, args.max_images))
            img_rows = [image_metrics(p) for p in sampled_images]
            if img_rows:
                imdf = pd.DataFrame(img_rows)
                imdf.insert(0, "session", session)
                all_img_metrics.append(imdf)
                ok = imdf[imdf["read_error"] == ""]
                write_line(summary, f"Метрики изображений: проверено {len(imdf)} из {len(images)}, "
                                    f"ошибок чтения {len(imdf) - len(ok)}")
                if len(ok):
                    write_line(summary, f"Размеры кадра (уникальные width×height): "
                                        f"{sorted(set(str(int(w))+'x'+str(int(h)) for w,h in zip(ok['width'].dropna(), ok['height'].dropna())))}")
                    write_line(summary, f"Средняя яркость: median={ok['brightness_mean'].median():.2f}, "
                                        f"range=[{ok['brightness_mean'].min():.2f}, {ok['brightness_mean'].max():.2f}]")
                    write_line(summary, f"Показатель краёв (грубый proxy резкости): median={ok['edge_std'].median():.2f}, "
                                        f"range=[{ok['edge_std'].min():.2f}, {ok['edge_std'].max():.2f}]")
                    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
                    axes[0].hist(ok["brightness_mean"].dropna(), bins=30, edgecolor="black", linewidth=0.4)
                    axes[0].set_title("Яркость")
                    axes[1].hist(ok["contrast_gray_std"].dropna(), bins=30, edgecolor="black", linewidth=0.4)
                    axes[1].set_title("Контраст")
                    axes[2].hist(ok["edge_std"].dropna(), bins=30, edgecolor="black", linewidth=0.4)
                    axes[2].set_title("Показатель краёв")
                    for ax in axes:
                        ax.grid(True, axis="y", alpha=0.25)
                    fig.suptitle(f"{session}: качество выборки изображений")
                    fig.tight_layout()
                    fig.savefig(session_plot_dir / "image_quality_distributions.png", dpi=140)
                    plt.close(fig)
            else:
                write_line(summary, "Изображения не найдены или папка nadir_camera отсутствует.")
            write_line(summary, "")

    # Consolidated CSV outputs.
    if all_stats:
        pd.concat(all_stats, ignore_index=True).to_csv(out_root / "column_statistics.csv", index=False)
    if all_diag:
        pd.concat(all_diag, ignore_index=True).to_csv(out_root / "missingness_and_constants.csv", index=False)
    if all_timing:
        pd.DataFrame(all_timing).to_csv(out_root / "timestamp_analysis.csv", index=False)
    if all_corr:
        pd.DataFrame(all_corr).to_csv(out_root / "correlations.csv", index=False)
    if all_sync:
        pd.concat(all_sync, ignore_index=True).to_csv(out_root / "camera_sync.csv", index=False)
    if all_img_metrics:
        pd.concat(all_img_metrics, ignore_index=True).to_csv(out_root / "image_quality_sample.csv", index=False)

    # Small index/readme.
    readme = out_root / "README.txt"
    with readme.open("w", encoding="utf-8") as f:
        write_line(f, "Результаты analyze_drone_dataset.py")
        write_line(f, "")
        write_line(f, "summary.txt — сводный человекочитаемый отчёт.")
        write_line(f, "column_statistics.csv — распределения, квантили и пропуски по столбцам.")
        write_line(f, "missingness_and_constants.csv — пропуски, число уникальных значений и константные столбцы.")
        write_line(f, "timestamp_analysis.csv — длительность, интервалы и оценка частоты каждого источника.")
        write_line(f, "correlations.csv — корреляции между числовыми признаками.")
        write_line(f, "camera_sync.csv — ближайшие временные совпадения камеры с FC и лидаром.")
        write_line(f, "image_quality_sample.csv — метрики выборки изображений; исходные изображения не копируются.")
        write_line(f, "plots/ — графики по каждой записи.")
        write_line(f, "")
        write_line(f, "Важно: camera_sync.csv — предварительная синхронизация по ближайшей временной метке, а не гарантия")
        write_line(f, "того, что данные относятся к одному физическому моменту. Проверьте временные зоны/часы устройства,")
        write_line(f, "а также смысл relative_alt и dist перед выбором целевой переменной.")

    print(f"\nГотово. Результаты сохранены в:\n{out_root}")
    print("Для отправки мне лучше всего упаковать в ZIP всю папку dataset_analysis.")
    print("Папки с JPEG в архив добавлять не нужно.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()

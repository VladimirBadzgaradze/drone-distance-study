#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Аудит целевой переменной dist и синхронизации камеры/лидара/FC.

Структура ROOT/SESSION/{fc_data.csv,lidar_data.csv,nadir_camera/*.jpeg}
Зависимости: pip install pandas numpy matplotlib
Запуск: python audit_drone_target.py /path/to/dataset_root
Результат: target_audit/ рядом с ROOT (или --out DIR).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_ts(s):
    return pd.to_datetime(s.astype(str).str.strip(), errors='coerce')


def load_csv(path, time_col='datetime'):
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise ValueError(f'{path}: нет столбца {time_col}; есть {list(df.columns)}')
    df[time_col] = parse_ts(df[time_col])
    return df.dropna(subset=[time_col]).sort_values(time_col).drop_duplicates(time_col).reset_index(drop=True)


def nearest(source_times, target_times):
    """Return nearest source row index and signed source_time-target_time seconds."""
    a = pd.DatetimeIndex(source_times).asi8
    b = pd.DatetimeIndex(target_times).asi8
    idx = np.searchsorted(a, b)
    idx = np.clip(idx, 0, len(a)-1)
    prev = np.clip(idx-1, 0, len(a)-1)
    choose_prev = np.abs(a[prev]-b) <= np.abs(a[idx]-b)
    idx = np.where(choose_prev, prev, idx)
    delta = (a[idx]-b) / 1e9
    return idx, delta


def audit_session(folder: Path, outdir: Path):
    lidar = load_csv(folder/'lidar_data.csv')
    if 'dist' not in lidar.columns:
        raise ValueError(f'{folder}: в lidar_data.csv нет столбца dist')
    lidar['dist'] = pd.to_numeric(lidar['dist'], errors='coerce')
    lidar = lidar.dropna(subset=['dist']).sort_values('datetime').reset_index(drop=True)
    if len(lidar) < 2:
        return {'session': folder.name, 'error': 'меньше двух измерений лидара'}

    lidar['dt_s'] = lidar.datetime.diff().dt.total_seconds()
    lidar['delta_dist_m'] = lidar.dist.diff()
    lidar['abs_delta_dist_m'] = lidar.delta_dist_m.abs()
    lidar['rate_mps'] = lidar.delta_dist_m / lidar.dt_s
    # Data-driven robust jump flag; 10 m floor prevents tiny noise being overflagged.
    med_abs = float(lidar.abs_delta_dist_m.median())
    mad = float((lidar.abs_delta_dist_m - med_abs).abs().median())
    jump_threshold = max(10.0, med_abs + 6 * 1.4826 * mad)
    lidar['large_jump'] = lidar.abs_delta_dist_m >= jump_threshold
    lidar.to_csv(outdir/f'{folder.name}_lidar_quality.csv', index=False)

    fc_path = folder/'fc_data.csv'
    fc = load_csv(fc_path) if fc_path.exists() else pd.DataFrame(columns=['datetime'])
    if 'relative_alt' in fc.columns:
        fc['relative_alt'] = pd.to_numeric(fc['relative_alt'], errors='coerce')

    # One row per camera frame, based on timestamps embedded in image filenames.
    camdir = folder/'nadir_camera'
    files = sorted([p for p in camdir.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]) if camdir.exists() else []
    cam = pd.DataFrame({'image_path': [str(p.relative_to(folder)) for p in files],
                        'datetime': [pd.to_datetime(p.stem, format='%Y-%m-%d_%H-%M-%S_%f', errors='coerce') for p in files]})
    cam = cam.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    if len(cam):
        li, ldelta = nearest(lidar.datetime, cam.datetime)
        cam['lidar_nearest_datetime'] = lidar.datetime.iloc[li].to_numpy()
        cam['lidar_nearest_dist_m'] = lidar.dist.iloc[li].to_numpy()
        cam['lidar_time_offset_s'] = ldelta
        cam['lidar_nearest_abs_offset_s'] = np.abs(ldelta)
        cam['lidar_label_within_0_25s'] = cam.lidar_nearest_abs_offset_s <= 0.25
        # Linear interpolation is diagnostic only; do not treat as trusted labels across large jumps.
        l_ns = pd.DatetimeIndex(lidar.datetime).asi8.astype(np.int64)
        c_ns = pd.DatetimeIndex(cam.datetime).asi8.astype(np.int64)
        dist_interp = np.interp(c_ns.astype(float), l_ns.astype(float), lidar.dist.to_numpy(), left=np.nan, right=np.nan)
        cam['lidar_dist_linear_interp_m'] = dist_interp
        right = np.searchsorted(l_ns, c_ns, side='left')
        right = np.clip(right, 0, len(lidar)-1)
        left = np.clip(right-1, 0, len(lidar)-1)
        bracket_gap = (l_ns[right]-l_ns[left])/1e9
        bracket_jump = np.abs(lidar.dist.to_numpy()[right] - lidar.dist.to_numpy()[left])
        cam['lidar_bracket_gap_s'] = bracket_gap
        cam['lidar_bracket_change_m'] = bracket_jump
        cam['interpolation_suspect'] = (bracket_gap > 0.8) | (bracket_jump >= jump_threshold)
        if len(fc) and 'relative_alt' in fc.columns:
            fi, fdelta = nearest(fc.datetime, cam.datetime)
            cam['fc_nearest_datetime'] = fc.datetime.iloc[fi].to_numpy()
            cam['fc_nearest_relative_alt_m'] = fc.relative_alt.iloc[fi].to_numpy()
            cam['fc_time_offset_s'] = fdelta
        cam.to_csv(outdir/f'{folder.name}_camera_labels_audit.csv', index=False)

    # Compare lidar with nearest FC sample at lidar timestamps, without repeating slow lidar values across FC rows.
    if len(fc) and 'relative_alt' in fc.columns:
        fi, fdelta = nearest(fc.datetime, lidar.datetime)
        lidar['fc_nearest_datetime'] = fc.datetime.iloc[fi].to_numpy()
        lidar['fc_nearest_relative_alt_m'] = fc.relative_alt.iloc[fi].to_numpy()
        lidar['fc_time_offset_s'] = fdelta
        lidar['relative_alt_minus_dist_m'] = lidar.fc_nearest_relative_alt_m - lidar.dist
        lidar.to_csv(outdir/f'{folder.name}_lidar_fc_comparison.csv', index=False)

    # Plot distance and FC relative altitude on raw timestamps.
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(lidar.datetime, lidar.dist, label='Лидар: dist, м', linewidth=1.2)
    if len(fc) and 'relative_alt' in fc.columns:
        ax.plot(fc.datetime, fc.relative_alt, label='FC: relative_alt, м', alpha=0.65, linewidth=0.8)
    ax.set_title(f'{folder.name}: расстояние до поверхности и relative_alt')
    ax.set_xlabel('Время'); ax.set_ylabel('Метры'); ax.grid(True, alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(outdir/f'{folder.name}_dist_vs_relative_alt.png', dpi=150); plt.close(fig)

    # Plot lidar steps and mark large changes.
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(lidar.datetime, lidar.dist, marker='.', markersize=2.5, linewidth=.8, label='dist')
    flagged = lidar[lidar.large_jump]
    if len(flagged): ax.scatter(flagged.datetime, flagged.dist, marker='x', s=45, label=f'скачок ≥ {jump_threshold:.1f} м')
    ax.set_title(f'{folder.name}: скачки показаний лидара'); ax.set_xlabel('Время'); ax.set_ylabel('dist, м')
    ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(outdir/f'{folder.name}_lidar_jumps.png', dpi=150); plt.close(fig)

    valid_dist = lidar.dist
    result = {
        'session': folder.name, 'n_lidar': len(lidar), 'n_camera_frames': len(cam), 'n_fc': len(fc),
        'lidar_start': lidar.datetime.min(), 'lidar_end': lidar.datetime.max(),
        'dist_min_m': valid_dist.min(), 'dist_q01_m': valid_dist.quantile(.01),
        'dist_median_m': valid_dist.median(), 'dist_mean_m': valid_dist.mean(),
        'dist_q99_m': valid_dist.quantile(.99), 'dist_max_m': valid_dist.max(),
        'dist_nonpositive_n': int((valid_dist <= 0).sum()),
        'dist_at_or_below_0_2m_n': int((valid_dist <= .2).sum()),
        'median_lidar_interval_s': lidar.dt_s.median(), 'jump_threshold_m': jump_threshold,
        'large_jump_count': int(lidar.large_jump.sum()),
        'median_abs_lidar_camera_offset_s': float(cam.lidar_nearest_abs_offset_s.median()) if len(cam) else np.nan,
        'camera_frames_within_0_25s_pct': float(cam.lidar_label_within_0_25s.mean()*100) if len(cam) else np.nan,
        'camera_frames_interpolation_suspect_pct': float(cam.interpolation_suspect.mean()*100) if len(cam) else np.nan,
    }
    if len(fc) and 'relative_alt' in fc.columns and 'relative_alt_minus_dist_m' in lidar:
        dif = lidar.relative_alt_minus_dist_m.dropna()
        result.update({'median_relative_alt_minus_dist_m': dif.median(), 'q05_relative_alt_minus_dist_m': dif.quantile(.05),
                       'q95_relative_alt_minus_dist_m': dif.quantile(.95)})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path, help='Папка с папками записей')
    ap.add_argument('--out', type=Path, default=None, help='Папка результатов')
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    out = (args.out.expanduser().resolve() if args.out else root/'target_audit')
    out.mkdir(parents=True, exist_ok=True)
    sessions = [p for p in root.iterdir() if p.is_dir() and (p/'lidar_data.csv').exists() and (p/'nadir_camera').exists()]
    if not sessions:
        raise SystemExit(f'Не найдены папки записей с lidar_data.csv и nadir_camera внутри {root}')
    summaries = []
    for session in sorted(sessions):
        try:
            summaries.append(audit_session(session, out))
            print(f'Готово: {session.name}')
        except Exception as e:
            summaries.append({'session': session.name, 'error': str(e)})
            print(f'ОШИБКА {session.name}: {e}')
    pd.DataFrame(summaries).to_csv(out/'target_audit_summary.csv', index=False)
    (out/'README.txt').write_text(
        'Аудит целевой переменной dist.\n'
        '* *_lidar_quality.csv: интервалы, скачки и скорость изменения dist.\n'
        '* *_camera_labels_audit.csv: по одному ряду на кадр; ближайший лидар, временной сдвиг и диагностическая интерполяция.\n'
        '* *_lidar_fc_comparison.csv: сравнение relative_alt и dist на временных метках лидара.\n'
        '* *_dist_vs_relative_alt.png и *_lidar_jumps.png: графики.\n\n'
        'Важно: линейная интерполяция помечена как диагностическая, а не гарантированная истинная разметка.\n'
        'Порог 0.25 с — предварительный критерий близости к измерению лидара; окончательный выбор зависит от динамики полёта.\n'
        'Большие скачки dist отмечаются автоматически и требуют просмотра соответствующих кадров/видео, а не автоматически считаются ошибками.\n', encoding='utf-8')
    print(f'Результаты: {out}')

if __name__ == '__main__': 
    main()

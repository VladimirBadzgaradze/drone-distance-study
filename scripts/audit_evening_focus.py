#!/usr/bin/env python3
"""Create a focused, read-only audit for the 2025-09-18 04:12 camera minute."""
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "20250918_lidar_evening"
AUDIT = ROOT / "results" / "target_audit"
START = pd.Timestamp("2025-09-18 04:12:00")
END = pd.Timestamp("2025-09-18 04:13:00")
OUT_FRAMES = AUDIT / "20250918_lidar_evening_0412_0413_frame_audit.csv"
OUT_EVENTS = AUDIT / "20250918_lidar_evening_0412_0413_jump_events.csv"
OUT_SHEET = AUDIT / "20250918_lidar_evening_0412_0413_jump_contact_sheet.jpg"
OUT_REPORT = AUDIT / "20250918_lidar_evening_0412_0413_analysis.md"


def read_timed(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)


def nearest_frame(frames, timestamp):
    if frames.empty:
        return "", ""
    i = (frames["datetime"] - timestamp).abs().idxmin()
    row = frames.loc[i]
    return row["image_path"], row["datetime"].strftime("%Y-%m-%d %H:%M:%S.%f")


def main():
    frames = read_timed(AUDIT / "20250918_lidar_evening_camera_labels_audit.csv")
    frames = frames[(frames.datetime >= START) & (frames.datetime < END)].copy()
    if frames.empty:
        raise SystemExit("No camera audit rows found for the requested interval")

    frames["label_status"] = "candidate_nearest_dist_review_required"
    frames.loc[frames["lidar_nearest_abs_offset_s"] > 0.25, "label_status"] = "resync_or_exclude"
    frames.loc[frames["interpolation_suspect"].astype(str).str.lower().eq("true"), "label_status"] = "manual_review_jump_or_gap"
    frames["target_variable"] = "lidar_nearest_dist_m"
    frames["relative_alt_role"] = "comparison_only"
    frames.to_csv(OUT_FRAMES, index=False, encoding="utf-8")

    quality = read_timed(AUDIT / "20250918_lidar_evening_lidar_quality.csv")
    lidar = read_timed(DATA / "lidar_data.csv")
    jumps = quality[quality["large_jump"].astype(str).str.lower().eq("true")].copy()
    jumps = jumps[(jumps.datetime >= START) & (jumps.datetime < END)].reset_index(drop=True)

    # Adjacent flagged lidar intervals belong to one event when their end times
    # are no more than 1.1 s apart (roughly two nominal lidar periods).
    groups = []
    for _, row in jumps.iterrows():
        if not groups or (row.datetime - groups[-1][-1].datetime).total_seconds() > 1.1:
            groups.append([row])
        else:
            groups[-1].append(row)

    event_rows = []
    for n, group in enumerate(groups, 1):
        first, last = group[0], group[-1]
        t0, t1 = first.datetime, last.datetime
        before = lidar[lidar.datetime < t0].tail(1)
        after = lidar[lidar.datetime >= t1].head(1)
        pre_time = before.iloc[0].datetime if not before.empty else t0
        post_time = after.iloc[0].datetime if not after.empty else t1
        before_path, before_frame_time = nearest_frame(frames, pre_time)
        after_path, after_frame_time = nearest_frame(frames, post_time)
        local_frames = frames[(frames.datetime >= t0 - pd.Timedelta(seconds=.5)) &
                              (frames.datetime <= t1 + pd.Timedelta(seconds=.5))]
        event_rows.append({
            "event_id": f"J{n:02d}",
            "jump_interval_start": t0,
            "jump_interval_end": t1,
            "flagged_intervals_n": len(group),
            "max_abs_delta_dist_m": pd.to_numeric(pd.Series([r.abs_delta_dist_m for r in group]), errors="coerce").max(),
            "dist_before_m": before.iloc[0].dist if not before.empty else None,
            "dist_after_m": after.iloc[0].dist if not after.empty else None,
            "frames_within_event_window_n": len(local_frames),
            "representative_frame_before": before_path,
            "representative_frame_before_time": before_frame_time,
            "representative_frame_after": after_path,
            "representative_frame_after_time": after_frame_time,
            "review_status": "visual_review_required; automatic jump flag is not a confirmed sensor error",
        })
    events = pd.DataFrame(event_rows)
    events.to_csv(OUT_EVENTS, index=False, encoding="utf-8")

    # Make a compact visual index; originals remain in the dataset directory.
    thumb_w, thumb_h, label_h = 360, 203, 34
    sheet = Image.new("RGB", (thumb_w * 2, max(1, len(events)) * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in events.iterrows():
        for col, key, caption in ((0, "representative_frame_before", "before"),
                                  (1, "representative_frame_after", "after")):
            path = DATA / row[key]
            x, y = col * thumb_w, i * (thumb_h + label_h)
            value = row.dist_before_m if caption == "before" else row.dist_after_m
            draw.text((x + 4, y + 2), f"{row.event_id} {caption} | dist={value} m", fill="black")
            if path.exists():
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((thumb_w, thumb_h))
                    sheet.paste(im, (x, y + label_h))
            else:
                draw.text((x + 8, y + label_h + 8), f"Missing: {row[key]}", fill="red")
    sheet.save(OUT_SHEET, quality=88)

    counts = frames.label_status.value_counts().to_dict()
    dist = pd.to_numeric(frames.lidar_nearest_dist_m, errors="coerce")
    offsets = pd.to_numeric(frames.lidar_nearest_abs_offset_s, errors="coerce")
    suspect_mask = frames["interpolation_suspect"].astype(str).str.lower().eq("true")
    report = f"""# Фокусный аудит 18.09.2025, 04:12–04:13

## Данные и метка

В минуте найдено **{len(frames)} кадров**. Целевая величина — ближайшее измерение `dist` из лидара, расстояние лидара до поверхности под дроном. `relative_alt` из FC оставлен только для сравнения. Линейная интерполяция сохранена как диагностика и не назначается целевой меткой.

Ближайший `dist` в этом интервале лежит в диапазоне **{dist.min():.1f}–{dist.max():.1f} м**. По порогу временного рассогласования 0,25 с: **{(offsets <= .25).sum()}** кадров проходят предварительный фильтр и **{(offsets > .25).sum()}** его не проходят. У **{suspect_mask.sum()}** кадров дополнительно подозрительна интерполяция из-за скачка или длинного интервала. Это независимые признаки; статусы ниже назначены с приоритетом ручной проверки скачка/интервала.

## Результат отбора кадров

- `candidate_nearest_dist_review_required`: {counts.get('candidate_nearest_dist_review_required', 0)} — ближайшее измерение в пределах 0,25 с, пригодно лишь как кандидат.
- `manual_review_jump_or_gap`: {counts.get('manual_review_jump_or_gap', 0)} — интерполяция затрагивает большой скачок или длинный интервал; проверить кадр и соседние отсчёты.
- `resync_or_exclude`: {counts.get('resync_or_exclude', 0)} — ближайший отсчёт дальше 0,25 с; не использовать автоматически.

Статусы взаимоисключающие для удобства очереди проверки. Четыре кадра одновременно превышают 0,25 с и имеют флаг подозрительной интерполяции; они попали в `manual_review_jump_or_gap`.

## Скачки лидара

В интервале отмечено **{len(jumps)} скачков** по порогу из общего аудита, объединённых в **{len(events)} эпизодов**. Порог статистический: он выделяет места для просмотра, но сам по себе не доказывает ошибку лидара. В CSV эпизодов указаны скачки, значения до/после, количество кадров вокруг события и репрезентативные кадры для визуальной проверки.

Визуально просмотренный контактный лист показывает лесной полог и край просеки с меняющимся ракурсом. Для большинства пар нет очевидной мгновенной смены сцены, которая сама объяснила бы скачок; кадры не позволяют подтвердить или опровергнуть сбой лидара. Кроме того, в доступном CSV нет метаданных, определяющих, отражается ли `dist` от крон, земли или иной поверхности. Поэтому фактический объект измерения нужно уточнить по документации лидара, если нужна именно высота над землёй, а не расстояние до ближайшей отражающей поверхности.

## Файлы

- `{OUT_FRAMES.name}` — по одной строке на кадр с целевой колонкой, временем, сдвигом, диагностикой интерполяции, статусом отбора и сравнительным `relative_alt`.
- `{OUT_EVENTS.name}` — каталог эпизодов скачков и кадров до/после.
- `{OUT_SHEET.name}` — контактный лист пар кадров до/после скачков.
- Исходные изображения находятся в `../20250918_lidar_evening/nadir_camera/`; исходные CSV не изменялись.

## Рекомендуемое использование

Для обучения брать `lidar_nearest_dist_m` только после визуальной проверки кандидатов и пригодности качества кадра. Строки `manual_review_jump_or_gap` разбирать по соседним отсчётам и кадрам; `resync_or_exclude` не превращать в метки автоматически. Не заменять `dist` на `relative_alt` и не использовать диагностическую интерполяцию как истину около скачков.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {OUT_FRAMES}")
    print(f"Wrote {OUT_EVENTS}")
    print(f"Wrote {OUT_SHEET}")
    print(f"Wrote {OUT_REPORT}")
    print(f"Frames={len(frames)}; jumps={len(jumps)}; events={len(events)}; statuses={counts}")


if __name__ == "__main__":
    main()

"""Create validation dashboards with dataset labels and per-image comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynamic_attention import register_dynamic_attention  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IOU_MATCH_THRESHOLD = 0.50
PREDICTION_CONFIDENCE = 0.25
SEED = 42


def image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def choose_sample(paths: list[Path], count: int, seed: int = SEED) -> list[Path]:
    if len(paths) <= count:
        return paths
    return sorted(random.Random(seed).sample(paths, count))


def read_yolo_labels(path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        _, cx, cy, bw, bh = map(float, fields)
        boxes.append([
            (cx - bw / 2) * width, (cy - bh / 2) * height,
            (cx + bw / 2) * width, (cy + bh / 2) * height,
        ])
    return boxes


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_boxes(truth: list[list[float]], predictions: list[list[float]]) -> tuple[int, int, int]:
    candidates = sorted(
        ((box_iou(gt, pred), gi, pi) for gi, gt in enumerate(truth) for pi, pred in enumerate(predictions)),
        reverse=True,
    )
    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    for overlap, gi, pi in candidates:
        if overlap < IOU_MATCH_THRESHOLD:
            break
        if gi not in matched_truth and pi not in matched_predictions:
            matched_truth.add(gi)
            matched_predictions.add(pi)
    return len(matched_truth), len(predictions) - len(matched_predictions), len(truth) - len(matched_truth)


def draw_detection_dashboard(
    image: np.ndarray, truth: list[list[float]] | None, predictions: list[list[float]],
    confidences: list[float], task: str, source: Path, output: Path,
) -> tuple[int, int, int]:
    tp, fp, fn = match_boxes(truth, predictions) if truth is not None else (None, None, None)
    canvas = image.copy()
    for box in truth or []:
        x1, y1, x2, y2 = map(lambda v: int(round(v)), box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
    for box, confidence in zip(predictions, confidences):
        x1, y1, x2, y2 = map(lambda v: int(round(v)), box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(canvas, f"pred {confidence:.2f}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

    panel_width = 410
    panel = np.full((max(image.shape[0], 245), panel_width, 3), (28, 29, 34), dtype=np.uint8)
    canvas = np.hstack((cv2.copyMakeBorder(canvas, 0, panel.shape[0] - image.shape[0], 0, 0,
                                           cv2.BORDER_CONSTANT, value=(0, 0, 0)), panel))
    lines = [
        (f"VALIDASI: {task.upper()}", (0, 215, 255), 0.62),
        ("Acuan dataset = kotak hijau", (0, 240, 0), 0.48) if truth is not None else ("Label acuan tidak tersedia", (0, 180, 255), 0.48),
        ("Prediksi model = kotak merah", (0, 100, 255), 0.48),
        (f"IoU cocok >= {IOU_MATCH_THRESHOLD:.2f}", (220, 220, 220), 0.48) if truth is not None else ("Tidak dapat hitung TP/FP/FN", (220, 220, 220), 0.46),
        (f"Cocok (TP): {tp}" if tp is not None else "Acuan: tidak diketahui", (220, 220, 220), 0.52),
        (f"Prediksi tanpa pasangan (FP): {fp}" if fp is not None else f"Kotak prediksi: {len(predictions)}", (220, 220, 220), 0.46),
        (f"Acuan terlewat (FN): {fn}" if fn is not None else "Bukan berarti gambar negatif", (220, 220, 220), 0.48),
        (f"Label acuan: {len(truth)} | prediksi: {len(predictions)}" if truth is not None else f"Label acuan: tidak ada | prediksi: {len(predictions)}", (220, 220, 220), 0.43),
        ("Acuan dataset perlu audit manusia." if truth is not None else "Tanpa label bukan bukti objek tidak ada.", (0, 180, 255), 0.40),
        ("Hasil ini bukan diagnosis.", (0, 180, 255), 0.42),
    ]
    y = 30
    for text, color, scale in lines:
        cv2.putText(canvas, text, (image.shape[1] + 15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 25
    cv2.putText(canvas, source.name[:48], (image.shape[1] + 15, min(panel.shape[0] - 8, y + 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 170, 170), 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)
    return tp, fp, fn


def detection_dashboards(root: Path, task: str, count: int, output_root: Path) -> list[dict]:
    from ultralytics import YOLO

    dataset = root / "datasets" / ("fish4knowledge" if task == "ikan" else "fishdisease")
    image_dir, label_dir = dataset / "images" / "val", dataset / "labels" / "val"
    images = image_files(image_dir)
    paired = [p for p in images if (label_dir / f"{p.stem}.txt").is_file()]
    unpaired = [p for p in images if not (label_dir / f"{p.stem}.txt").is_file()]
    selected = [(p, True) for p in choose_sample(paired, count, SEED + (0 if task == "ikan" else 1))]
    # Fish validation contains thousands of images without matching labels. Show a
    # small sample, but mark their ground truth unknown instead of calling boxes FP.
    if task == "ikan":
        selected.extend((p, False) for p in choose_sample(unpaired, min(20, count), SEED + 100))
    if task == "ikan":
        runs = sorted((root / "runs/detect/Runs_DynamicAttention").glob("model_ikan*/weights/best.pt"),
                      key=lambda p: p.stat().st_mtime_ns)
        weight = runs[-1] if runs else None
    else:
        weight = root / "runs/detect/Runs_DynamicAttention/model_lesi/weights/best.pt"
    if not weight or not weight.is_file():
        raise FileNotFoundError(f"Checkpoint {task} tidak ditemukan: {weight}")

    model = YOLO(str(weight))
    rows = []
    output_dir = output_root / f"deteksi_{task}"
    for index, (image_path, has_label) in enumerate(selected, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        truth = read_yolo_labels(label_dir / f"{image_path.stem}.txt", width, height) if has_label else None
        result = model(image, conf=PREDICTION_CONFIDENCE, verbose=False)[0]
        predictions = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
        confidences = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
        output = output_dir / f"{index:03d}_{image_path.stem}_acuan_vs_prediksi.jpg"
        tp, fp, fn = draw_detection_dashboard(image, truth, predictions, confidences, task, image_path, output)
        rows.append({
            "task": f"deteksi_{task}", "image": str(image_path), "dashboard": str(output),
            "checkpoint": str(weight), "ground_truth_status": "paired_dataset_label" if has_label else "unknown_no_matching_label",
            "ground_truth_boxes": len(truth) if truth is not None else None, "predicted_boxes": len(predictions),
            "tp_iou50": tp, "fp_iou50": fp, "fn_iou50": fn,
            "iou_threshold": IOU_MATCH_THRESHOLD, "prediction_confidence_threshold": PREDICTION_CONFIDENCE,
            "ground_truth_xyxy": json.dumps(truth), "prediction_xyxy": json.dumps(predictions),
        })
        if index % 10 == 0:
            print(f"Deteksi {task}: {index}/{len(selected)}")
    return rows


def classification_dashboards(root: Path, count_per_class: int, output_root: Path) -> list[dict]:
    from ultralytics import YOLO

    weight = root / "runs/classify/Runs_DynamicAttention/model_penyakit/weights/best.pt"
    if not weight.is_file():
        raise FileNotFoundError(f"Checkpoint klasifikasi tidak ditemukan: {weight}")
    model = YOLO(str(weight))
    validation = root / "datasets/freshwater_kaggle/val"
    rows = []
    output_dir = output_root / "klasifikasi_penyakit"
    class_dirs = sorted(p for p in validation.iterdir() if p.is_dir())
    for class_index, class_dir in enumerate(class_dirs):
        for image_index, image_path in enumerate(choose_sample(image_files(class_dir), count_per_class,
                                                                SEED + class_index), start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            result = model(image, verbose=False)[0]
            predicted_index = int(result.probs.top1)
            predicted_class = str(model.names[predicted_index])
            score = float(result.probs.data[predicted_index])
            correct = predicted_class == class_dir.name

            panel_width = 410
            panel = np.full((max(image.shape[0], 250), panel_width, 3), (28, 29, 34), dtype=np.uint8)
            canvas = np.hstack((cv2.copyMakeBorder(image, 0, panel.shape[0] - image.shape[0], 0, 0,
                                                   cv2.BORDER_CONSTANT, value=(0, 0, 0)), panel))
            details = [
                "VALIDASI: KLASIFIKASI",
                f"Acuan folder: {class_dir.name}",
                f"Prediksi top-1: {predicted_class}",
                f"Confidence relatif: {score:.4f} (bukan akurasi)",
                f"Cocok dengan acuan: {'YA' if correct else 'TIDAK'}",
                "Acuan = label folder dataset.",
                "Bukan diagnosis kesehatan ikan.",
                image_path.name[:48],
            ]
            y = 30
            for line_index, text in enumerate(details):
                color = (0, 215, 255) if line_index == 0 else (0, 220, 0) if line_index == 4 and correct else (0, 100, 255) if line_index == 4 else (225, 225, 225)
                cv2.putText(canvas, text[:50], (image.shape[1] + 15, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.43 if line_index else 0.58, color, 1, cv2.LINE_AA)
                y += 27
            output = output_dir / class_dir.name / f"{image_index:03d}_{image_path.stem}_acuan_vs_prediksi.jpg"
            output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output), canvas)
            rows.append({
                "task": "klasifikasi_penyakit", "image": str(image_path), "dashboard": str(output),
                "checkpoint": str(weight), "ground_truth_class": class_dir.name,
                "predicted_class": predicted_class, "top1_score": round(score, 6), "correct": correct,
            })
        print(f"Klasifikasi: selesai kelas {class_dir.name}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--detection-samples", type=int, default=60,
                        help="Jumlah gambar berlabel per tugas deteksi (ikan dan lesi).")
    parser.add_argument("--classification-per-class", type=int, default=40,
                        help="Jumlah gambar validasi per kelas klasifikasi.")
    args = parser.parse_args()
    if args.detection_samples < 1 or args.classification_per_class < 1:
        parser.error("Jumlah sampel harus minimal 1")
    root = args.root.expanduser().resolve()
    register_dynamic_attention()
    output_root = root / "reports" / "dashboard_validasi_acuan"
    rows = []
    rows.extend(detection_dashboards(root, "ikan", args.detection_samples, output_root))
    rows.extend(detection_dashboards(root, "lesi", args.detection_samples, output_root))
    rows.extend(classification_dashboards(root, args.classification_per_class, output_root))
    summary = output_root / "perbandingan_per_gambar.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with summary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Total dashboard: {len(rows)}")
    print(f"Ringkasan per gambar: {summary}")
    print("Catatan: label validasi adalah acuan dataset dan belum tentu benar; lakukan audit anotasi.")


if __name__ == "__main__":
    main()

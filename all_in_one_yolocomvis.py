"""All-in-one pipeline for fish detection, lesion detection, and disease classification."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS) if directory.exists() else []


def dataset_summary(root: Path) -> dict:
    """Collect image, label, object, and class statistics for the final report."""
    summary = {"fish_detection": {}, "lesion_detection": {}, "disease_classification": {}}
    for key, dataset in (("fish_detection", "fish4knowledge"), ("lesion_detection", "fishdisease")):
        dataset_dir = root / "datasets" / dataset
        split_stats = {}
        for split in ("train", "val"):
            image_dir = dataset_dir / "images" / split
            label_dir = dataset_dir / "labels" / split
            images = _image_files(image_dir)
            labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
            object_count = sum(len(label.read_text(encoding="utf-8").splitlines()) for label in labels)
            image_stems = {path.stem for path in images}
            label_stems = {path.stem for path in labels}
            split_stats[split] = {
                "images": len(images), "labels": len(labels), "objects": object_count,
                "images_without_labels": len(image_stems - label_stems),
                "labels_without_images": len(label_stems - image_stems),
            }
        summary[key] = {"path": str(dataset_dir), "splits": split_stats}

    classification_dir = root / "datasets" / "freshwater_kaggle"
    class_stats = {}
    for split in ("train", "val"):
        split_dir = classification_dir / split
        class_stats[split] = {name.name: len(_image_files(name)) for name in sorted(split_dir.iterdir()) if name.is_dir()} if split_dir.exists() else {}
    summary["disease_classification"] = {"path": str(classification_dir), "classes": class_stats}
    return summary


def save_dataset_report(root: Path) -> Path:
    """Save dataset statistics as JSON, CSV, and a readable Markdown table."""
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = dataset_summary(root)
    (report_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = []
    for dataset_name, dataset_data in summary.items():
        if "splits" in dataset_data:
            for split, values in dataset_data["splits"].items():
                rows.append([dataset_name, split, values["images"], values["labels"], values["objects"]])
    with (report_dir / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "split", "images", "labels", "objects"])
        writer.writerows(rows)
    lines = ["# Ringkasan Dataset", "", "| Dataset | Split | Gambar | Label | Objek |", "|---|---:|---:|---:|---:|"]
    lines.extend(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |" for row in rows)
    lines.extend(["", "> Periksa field `images_without_labels` dan `labels_without_images` pada JSON untuk menemukan pasangan file yang tidak cocok."])
    lines.extend(["", "## Klasifikasi Penyakit", "", "```json", json.dumps(summary["disease_classification"], indent=2), "```"])
    (report_dir / "dataset_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Laporan dataset: {report_dir}")
    return report_dir


def prepare_fish_dataset(root: Path, val_ratio: float = 0.2, seed: int = 42) -> None:
    """Convert masks, collect matching images, and create a validation split."""
    dataset = root / "datasets" / "fish4knowledge"
    mask_dir = dataset / "maskikan"
    train_images = dataset / "images" / "train"
    train_labels = dataset / "labels" / "train"
    val_images = dataset / "images" / "val"
    val_labels = dataset / "labels" / "val"

    for directory in (train_images, train_labels, val_images, val_labels):
        directory.mkdir(parents=True, exist_ok=True)

    converted = 0
    copied_images = 0
    missing_images = 0
    image_index = {}
    for image_path in dataset.rglob("*"):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
            and "maskikan" not in image_path.parts
            and image_path.parent not in (train_images, val_images)
        ):
            image_index.setdefault(image_path.stem.lower(), image_path)

    mask_paths = sorted(mask_dir.rglob("*.png")) if mask_dir.exists() else []
    if not mask_paths:
        mask_paths = sorted(
            path for path in dataset.rglob("*.png")
            if path.is_file() and path.stem.lower().startswith("mask_")
        )
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        height, width = mask.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        x, y, box_width, box_height = cv2.boundingRect(max(contours, key=cv2.contourArea))
        label_name = mask_path.name.replace("mask_", "fish_").replace(".png", ".txt")
        label_path = train_labels / label_name
        mask_stem = mask_path.stem.lower()
        image_stems = [
            mask_stem,
            mask_stem.removeprefix("mask_"),
            mask_stem.removeprefix("mask_").removeprefix("fish_"),
            f"fish_{mask_stem.removeprefix('mask_')}",
        ]
        image_path = next((image_index.get(stem) for stem in image_stems if image_index.get(stem)), None)
        if image_path is None:
            missing_images += 1
            continue
        target_image = train_images / image_path.name
        if not target_image.exists():
            shutil.copy2(image_path, target_image)
            copied_images += 1
        label_path.write_text(
            f"0 {(x + box_width / 2) / width:.6f} "
            f"{(y + box_height / 2) / height:.6f} "
            f"{box_width / width:.6f} {box_height / height:.6f}\n",
            encoding="utf-8",
        )
        converted += 1

    labels = sorted(train_labels.glob("*.txt"))
    random.Random(seed).shuffle(labels)
    validation_count = int(len(labels) * val_ratio)
    moved = 0
    for label_path in labels[:validation_count]:
        image_stem = label_path.stem
        candidates = [train_images / f"{image_stem}{extension}" for extension in IMAGE_EXTENSIONS]
        image_path = next((path for path in candidates if path.exists()), None)
        target_label = val_labels / label_path.name
        if not target_label.exists():
            shutil.move(str(label_path), str(target_label))
        if image_path is not None and not (val_images / image_path.name).exists():
            shutil.move(str(image_path), str(val_images / image_path.name))
        moved += 1

    print(
        f"Fish4Knowledge: {converted} pasangan diproses, {copied_images} gambar disalin, "
        f"{missing_images} gambar tidak ditemukan, {moved} pasangan dipindahkan ke val."
    )
    if converted == 0:
        raise RuntimeError(
            "Fish4Knowledge tidak menghasilkan pasangan gambar-label. "
            "Periksa struktur arsip dan pola nama mask/gambar sebelum training."
        )


def prepare_lesion_dataset(root: Path, val_ratio: float = 0.2, seed: int = 42) -> None:
    """Convert fish-disease JSON bounding boxes into a YOLO dataset."""
    raw_dir = root / "datasets" / "fishdisease_mentah"
    output_dir = root / "datasets" / "fishdisease"
    train_images = output_dir / "images" / "train"
    val_images = output_dir / "images" / "val"
    train_labels = output_dir / "labels" / "train"
    val_labels = output_dir / "labels" / "val"
    for directory in (train_images, val_images, train_labels, val_labels):
        directory.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    counters = {"success": 0, "empty": 0, "missing_image": 0, "without_bbox": 0, "failed": 0}
    json_paths = sorted(raw_dir.rglob("*.json"))
    print(f"FishDisease: ditemukan {len(json_paths)} file JSON.")

    for json_path in json_paths:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            annotations = data.get("annotations") or []
            if not annotations:
                counters["empty"] += 1
                continue

            image_path = raw_dir / data.get("image", "")
            if not image_path.is_file():
                page_dir = raw_dir / json_path.parent.name
                image_candidates = sorted(
                    path for path in page_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
                ) if page_dir.is_dir() else []
                image_path = image_candidates[0] if image_candidates else Path()
            if not image_path.is_file():
                counters["missing_image"] += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                counters["missing_image"] += 1
                continue
            height, width = image.shape[:2]
            yolo_lines = []
            for annotation in annotations:
                bbox = annotation.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x_min, y_min, x_max, y_max = map(float, bbox)
                x_min, x_max = sorted((max(0.0, x_min), min(float(width), x_max)))
                y_min, y_max = sorted((max(0.0, y_min), min(float(height), y_max)))
                if x_max <= x_min or y_max <= y_min:
                    continue
                yolo_lines.append(
                    f"0 {((x_min + x_max) / 2) / width:.6f} "
                    f"{((y_min + y_max) / 2) / height:.6f} "
                    f"{(x_max - x_min) / width:.6f} {(y_max - y_min) / height:.6f}"
                )
            if not yolo_lines:
                counters["without_bbox"] += 1
                continue

            unique_stem = f"{image_path.parent.name}_{image_path.stem}"
            is_validation = rng.random() < val_ratio
            image_dir = val_images if is_validation else train_images
            label_dir = val_labels if is_validation else train_labels
            shutil.copy2(image_path, image_dir / f"{unique_stem}{image_path.suffix.lower()}")
            (label_dir / f"{unique_stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")
            counters["success"] += 1
        except (OSError, ValueError, json.JSONDecodeError) as error:
            counters["failed"] += 1
            print(f"Lewati {json_path.name}: {error}")

    print(f"FishDisease selesai: {counters}")


def prepare_classification_dataset(root: Path, seed: int = 42) -> None:
    """Split the raw class-folder dataset into train and val folders."""
    try:
        import splitfolders
    except ImportError as error:
        raise RuntimeError("Install split-folders terlebih dahulu: pip install split-folders") from error

    source = root / "datasets" / "freshwater_asli"
    destination = root / "datasets" / "freshwater_kaggle"
    splitfolders.ratio(str(source), output=str(destination), seed=seed, ratio=(0.8, 0.2))
    print(f"Dataset klasifikasi siap: {destination}")


def write_dataset_yaml(root: Path) -> tuple[Path, Path]:
    """Write portable YOLO YAML files instead of relying on machine-specific paths."""
    fish_dir = root / "datasets" / "fish4knowledge"
    lesion_dir = root / "datasets" / "fishdisease"
    fish_yaml = root / "data_ikan_generated.yaml"
    lesion_yaml = root / "data_lesi_generated.yaml"
    fish_yaml.write_text(
        f"path: {fish_dir.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: fish\n",
        encoding="utf-8",
    )
    lesion_yaml.write_text(
        f"path: {lesion_dir.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: lesion\n",
        encoding="utf-8",
    )
    return fish_yaml, lesion_yaml


def train_models(root: Path, epochs: int = 10, project: str = "Runs_Baseline") -> dict:
    """Train the two detectors and the disease classifier."""
    from ultralytics import YOLO

    fish_yaml, lesion_yaml = write_dataset_yaml(root)
    results = {}
    results["fish_detection"] = YOLO(str(root / "yolo11n.pt")).train(
        data=str(fish_yaml), epochs=epochs, imgsz=640, project=project, name="model_ikan"
    )
    results["lesion_detection"] = YOLO(str(root / "yolo11n.pt")).train(
        data=str(lesion_yaml), epochs=epochs, imgsz=640, project=project, name="model_lesi"
    )
    results["disease_classification"] = YOLO(str(root / "yolo11n-cls.pt")).train(
        data=str(root / "datasets" / "freshwater_kaggle"),
        epochs=epochs, imgsz=224, project=project, name="model_penyakit",
    )
    return results


def _metric_value(metrics, *names: str):
    for name in names:
        value = getattr(metrics, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def evaluate_models(root: Path, project: str = "Runs_Baseline") -> dict:
    """Run validation and export numeric metrics for all trained models."""
    from ultralytics import YOLO

    model_specs = {
        "fish_detection": (root / project / "model_ikan" / "weights" / "best.pt", root / "data_ikan_generated.yaml"),
        "lesion_detection": (root / project / "model_lesi" / "weights" / "best.pt", root / "data_lesi_generated.yaml"),
        "disease_classification": (root / project / "model_penyakit" / "weights" / "best.pt", root / "datasets" / "freshwater_kaggle"),
    }
    report = {}
    for name, (weights, data) in model_specs.items():
        if not weights.exists():
            report[name] = {"status": "checkpoint tidak ditemukan", "weights": str(weights)}
            continue
        model = YOLO(str(weights))
        metrics = model.val(data=str(data), plots=True, project=str(root / "reports"), name=f"val_{name}")
        if name == "disease_classification":
            report[name] = {
                "status": "ok", "top1": _metric_value(metrics, "top1", "accuracy_top1"),
                "top5": _metric_value(metrics, "top5", "accuracy_top5"), "weights": str(weights),
            }
        else:
            box = getattr(metrics, "box", metrics)
            report[name] = {
                "status": "ok", "precision": _metric_value(box, "mp", "precision"),
                "recall": _metric_value(box, "mr", "recall"), "map50": _metric_value(box, "map50"),
                "map50_95": _metric_value(box, "map", "map50_95"), "weights": str(weights),
            }
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "evaluation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def plot_training_curves(root: Path, project: str = "Runs_Baseline") -> list[Path]:
    """Create readable training curve images from Ultralytics results.csv files."""
    import matplotlib.pyplot as plt
    import pandas as pd

    output_paths = []
    for model_name in ("model_ikan", "model_lesi", "model_penyakit"):
        results_csv = root / project / model_name / "results.csv"
        if not results_csv.exists():
            continue
        frame = pd.read_csv(results_csv)
        frame.columns = [column.strip() for column in frame.columns]
        selected = [column for column in frame.columns if any(token in column.lower() for token in ("loss", "precision", "recall", "map50"))]
        if not selected:
            continue
        figure, axis = plt.subplots(figsize=(12, 6))
        for column in selected:
            axis.plot(frame["epoch"], frame[column], label=column)
        axis.set_title(f"Kurva Training - {model_name}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Nilai")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
        figure.tight_layout()
        output = root / "reports" / f"training_curves_{model_name}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
        plt.close(figure)
        output_paths.append(output)
    print(f"Kurva training dibuat: {len(output_paths)} file")
    return output_paths


def classification_confusion_matrix(root: Path, project: str = "Runs_Baseline") -> Path | None:
    """Build and save a confusion matrix from the classification validation split."""
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    weights = root / project / "model_penyakit" / "weights" / "best.pt"
    validation_dir = root / "datasets" / "freshwater_kaggle" / "val"
    if not weights.exists() or not validation_dir.exists():
        print("Confusion matrix dilewati: checkpoint atau folder val belum tersedia.")
        return None
    model = YOLO(str(weights))
    class_names = [model.names[index] for index in sorted(model.names)]
    name_to_index = {name: index for index, name in enumerate(class_names)}
    matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    for class_dir in sorted(path for path in validation_dir.iterdir() if path.is_dir()):
        if class_dir.name not in name_to_index:
            continue
        actual = name_to_index[class_dir.name]
        for image_path in _image_files(class_dir):
            prediction = model(str(image_path), verbose=False)[0]
            predicted = int(prediction.probs.top1)
            matrix[actual, predicted] += 1
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(class_names)), yticks=range(len(class_names)), xticklabels=class_names, yticklabels=class_names, xlabel="Prediksi", ylabel="Label Aktual", title="Confusion Matrix Klasifikasi Penyakit")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, matrix[row, column], ha="center", va="center", color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    output = root / "reports" / "confusion_matrix_classification.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    np.savetxt(root / "reports" / "confusion_matrix_classification.csv", matrix, delimiter=",", fmt="%d")
    print(f"Confusion matrix disimpan di: {output}")
    return output


def write_final_report(root: Path, dataset_info: dict | None = None, evaluation: dict | None = None) -> Path:
    """Create a single Markdown report suitable as a starting point for a thesis/report."""
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset_info = dataset_info or dataset_summary(root)
    evaluation = evaluation or {}
    lines = ["# Laporan Eksperimen YOLOComVis", "", "## Konfigurasi", "- Task: deteksi ikan, deteksi lesi, klasifikasi penyakit", "- Split deteksi: train/val 80:20", "- Seed: 42", "- Model: YOLO11n dan YOLO11n-cls", "", "## Dataset"]
    for name, value in dataset_info.items():
        lines.append(f"### {name}")
        lines.append("```json")
        lines.append(json.dumps(value, indent=2))
        lines.append("```")
    lines.extend(["", "## Hasil Evaluasi", "```json", json.dumps(evaluation, indent=2), "```", "", "## Artefak", "- `dataset_summary.json/csv/md`: statistik dataset", "- `evaluation_metrics.json`: metrik validasi", "- `training_curves_*.png`: kurva training", "- `confusion_matrix_classification.png`: confusion matrix klasifikasi", "- `val_*/`: plot validasi Ultralytics", "- `hasil_dashboard_*.jpg`: visualisasi inferensi"])
    output = report_dir / "laporan_eksperimen.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Laporan akhir: {output}")
    return output


def run_dashboard(root: Path, image_path: Path, output_path: Path | None = None) -> Path:
    """Run the three models and save a notebook-friendly dashboard image."""
    from ultralytics import YOLO

    model_dir = root / "Runs_Baseline"
    fish_model = YOLO(str(model_dir / "model_ikan" / "weights" / "best.pt"))
    lesion_model = YOLO(str(model_dir / "model_lesi" / "weights" / "best.pt"))
    classifier = YOLO(str(model_dir / "model_penyakit" / "weights" / "best.pt"))
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    result = fish_model(image, verbose=False)[0]
    canvas = image.copy()
    panel_width = 440
    panel = np.full((max(image.shape[0], 520), panel_width, 3), (30, 30, 35), dtype=np.uint8)
    canvas = np.hstack((cv2.copyMakeBorder(
        canvas, 0, panel.shape[0] - canvas.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    ), panel))
    y_text = 38
    cv2.putText(canvas, "YOLO FISH ANALYSIS", (image.shape[1] + 20, y_text), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 215, 255), 2)
    y_text += 40

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0]) * 100
        crop = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
        if crop.size == 0:
            continue
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 215, 255), 2)
        lesion_result = lesion_model(crop, verbose=False)[0]
        for lesion in lesion_result.boxes:
            lx1, ly1, lx2, ly2 = map(int, lesion.xyxy[0].tolist())
            cv2.rectangle(canvas, (x1 + lx1, y1 + ly1), (x1 + lx2, y1 + ly2), (0, 0, 255), 2)
        classification = classifier(crop, verbose=False)[0]
        top_indices = classification.probs.top5[:3]
        cv2.putText(canvas, f"Fish: {confidence:.1f}%", (image.shape[1] + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y_text += 30
        cv2.putText(canvas, f"Lesions: {len(lesion_result.boxes)}", (image.shape[1] + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        y_text += 32
        for index in top_indices:
            name = classifier.names[int(index)]
            probability = float(classification.probs.data[int(index)]) * 100
            cv2.putText(canvas, f"{name}: {probability:.1f}%", (image.shape[1] + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 220, 150), 1)
            y_text += 24
        y_text += 18

    destination = output_path or root / "hasil_dashboard_baseline.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), canvas)
    print(f"Dashboard disimpan di: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mode", choices=("prepare", "train", "evaluate", "report", "dashboard", "all"), default="prepare")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    if args.mode in ("prepare", "all"):
        prepare_fish_dataset(root, seed=args.seed)
        prepare_lesion_dataset(root, seed=args.seed)
        prepare_classification_dataset(root, seed=args.seed)
        save_dataset_report(root)
    if args.mode in ("train", "all"):
        train_models(root, epochs=args.epochs)
    if args.mode in ("evaluate", "all"):
        evaluation = evaluate_models(root)
        plot_training_curves(root)
        classification_confusion_matrix(root)
        write_final_report(root, evaluation=evaluation)
    if args.mode in ("report",):
        save_dataset_report(root)
        evaluation = evaluate_models(root)
        plot_training_curves(root)
        classification_confusion_matrix(root)
        write_final_report(root, evaluation=evaluation)
    if args.mode in ("dashboard", "all"):
        image_path = args.image or root / "ujicoba.png"
        run_dashboard(root, image_path)


if __name__ == "__main__":
    main()

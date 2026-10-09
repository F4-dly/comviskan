"""All-in-one pipeline for fish detection, lesion detection, and disease classification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
FISH_CONF_THRESHOLD = 0.50
LESION_CONF_THRESHOLD = 0.25
DEFAULT_FISH_CHECKPOINT = Path("runs") / "detect" / "Runs_DynamicAttention" / "model_ikan_mixed_v1_compact" / "weights" / "best.pt"
REVIEW_CLASSIFICATION_THRESHOLD = 0.80


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


def audit_datasets(root: Path) -> Path:
    """Audit split integrity, duplicate files, label geometry, and class balance."""
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    audit = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "datasets": {}}

    for dataset_name in ("fish4knowledge", "fishdisease"):
        dataset_dir = root / "datasets" / dataset_name
        split_hashes: dict[str, dict[str, str]] = {}
        split_stats = {}
        for split in ("train", "val"):
            image_dir = dataset_dir / "images" / split
            label_dir = dataset_dir / "labels" / split
            hashes = {}
            image_paths = _image_files(image_dir)
            image_stems = {path.stem for path in image_paths}
            label_stems = {path.stem for path in label_dir.glob("*.txt")} if label_dir.exists() else set()
            invalid_labels = 0
            for image_path in image_paths:
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                hashes[digest] = image_path.name
                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.is_file():
                    continue
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if len(fields) != 5:
                        invalid_labels += 1
                        continue
                    try:
                        class_id, cx, cy, width, height = map(float, fields)
                    except ValueError:
                        invalid_labels += 1
                        continue
                    if class_id < 0 or not all(0 <= value <= 1 for value in (cx, cy, width, height)) or width <= 0 or height <= 0:
                        invalid_labels += 1
            split_hashes[split] = hashes
            split_stats[split] = {
                "images": len(image_paths),
                "labels": len(label_stems),
                "images_without_labels": len(image_stems - label_stems),
                "labels_without_images": len(label_stems - image_stems),
                "invalid_label_lines": invalid_labels,
            }
        overlap = sorted(set(split_hashes["train"]).intersection(split_hashes["val"]))
        audit["datasets"][dataset_name] = {
            "splits": split_stats,
            "exact_duplicate_image_hashes_between_train_val": len(overlap),
            "duplicate_examples": [split_hashes["train"][digest] for digest in overlap[:10]],
            "note": "This checks exact image duplicates only; near-duplicate video frames require source metadata or perceptual hashing.",
        }

    classification_dir = root / "datasets" / "freshwater_kaggle"
    audit["datasets"]["freshwater_kaggle"] = {
        "classes": {
            split: {
                path.name: len(_image_files(path))
                for path in sorted((classification_dir / split).iterdir())
                if path.is_dir()
            } if (classification_dir / split).exists() else {}
            for split in ("train", "val")
        },
        "note": "Folder split audit cannot prove that adjacent frames or the same fish are separated.",
    }
    output = report_dir / "dataset_audit.json"
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Audit dataset: {output}")
    return output


def write_readiness_report(root: Path) -> Path:
    """Record which presentation requirements are evidenced and which remain blocked."""
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prototype_ready_with_blockers",
        "requirements": [
            {"id": 1, "name": "fish_detection", "status": "validation_complete_test_blocked",
             "evidence": "Clean-split retraining completed; validation metrics are in reports/fish_retrained_metrics.json. Independent expert-reviewed test evidence is still missing."},
            {"id": 2, "name": "lesion_detection", "status": "validation_low_retraining_required",
             "evidence": "A low-memory retraining run was completed but underperformed the retained checkpoint; reports/lesion_retrained_metrics.json records the result and limitation."},
            {"id": 3, "name": "independent_test_set", "status": "blocked_data",
             "evidence": "No locked source-aware test labels are available in the current workspace."},
            {"id": 4, "name": "leakage_and_annotation_audit", "status": "partial",
             "evidence": "Read-only hash/label audit is in reports/dataset_audit.json; expert visual annotation review remains."},
            {"id": 5, "name": "end_to_end_validation", "status": "blocked_labels",
             "evidence": "The pipeline runs, but no end-to-end ground-truth test set exists."},
            {"id": 6, "name": "checkpoint_provenance", "status": "complete",
             "evidence": "Checkpoint paths, runtime, timestamps, thresholds, and seed are recorded in evaluation_metrics.json and reports."},
            {"id": 7, "name": "safe_score_wording", "status": "complete",
             "evidence": "Dashboard labels confidence as relative and warns that it is not accuracy or diagnosis."},
            {"id": 8, "name": "per_class_and_error_analysis", "status": "partial",
             "evidence": "Classification confusion matrix and aggregate metrics exist; detector per-class/error review still requires a clean test set."},
            {"id": 9, "name": "baseline_and_ablation", "status": "partial",
             "evidence": "YOLO baseline metrics exist; a controlled same-split ablation is not valid until splits are rebuilt."},
            {"id": 10, "name": "reproducibility_and_limitations", "status": "complete",
             "evidence": "README documents commands, limitations, dataset caveats, and Windows-safe execution."},
        ],
    }
    output = root / "reports" / "presentation_readiness.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def write_checkpoint_provenance(root: Path, project: str = "Runs_DynamicAttention") -> Path:
    """Record checkpoint hashes, training epochs, and configured thresholds."""
    checkpoints = [
        ("fish_detection", root / "runs" / "detect" / project / "model_ikan_lowmem" / "weights" / "best.pt"),
        ("lesion_detection", root / "runs" / "detect" / project / "model_lesi" / "weights" / "best.pt"),
        ("disease_classification", root / "runs" / "classify" / project / "model_penyakit" / "weights" / "best.pt"),
    ]
    entries = []
    for task, checkpoint in checkpoints:
        result_csv = checkpoint.parent.parent / "results.csv"
        epoch_count = None
        if result_csv.exists():
            epoch_count = max(0, len(result_csv.read_text(encoding="utf-8").splitlines()) - 1)
        entries.append({
            "task": task,
            "checkpoint": str(checkpoint),
            "exists": checkpoint.is_file(),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else None,
            "checkpoint_modified_utc": datetime.fromtimestamp(
                checkpoint.stat().st_mtime, timezone.utc
            ).isoformat() if checkpoint.is_file() else None,
            "completed_epochs_from_results_csv": epoch_count,
        })
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "fish_detection_confidence": FISH_CONF_THRESHOLD,
            "lesion_detection_confidence": LESION_CONF_THRESHOLD,
            "classification_review_threshold": REVIEW_CLASSIFICATION_THRESHOLD,
        },
        "seed": 42,
        "entries": entries,
        "warning": "These checkpoints are historical validation artifacts, not independent-test evidence.",
    }
    output = root / "reports" / "checkpoint_provenance.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def freeze_baseline(root: Path, project: str = "Runs_DynamicAttention") -> Path:
    """Freeze hashes and metadata for the current baseline without copying datasets."""
    import subprocess

    tracked_files = [
        root / "all_in_one_yolocomvis.py",
        root / "dynamic_attention.py",
        root / "scripts" / "3_training_all.py",
        root / "scripts" / "4_tes_pipeline_final.py",
        root / "scripts" / "5_dashboard_validasi_dengan_acuan.py",
        root / "README.md",
        root / "reports" / "evaluation_metrics.json",
        root / "reports" / "dataset_audit.json",
        root / "reports" / "presentation_readiness.json",
        root / "hasil_uji_coba_sekarang.jpg",
    ]
    checkpoint_files = [
        root / "runs" / "detect" / project / "model_ikan-4" / "weights" / "best.pt",
        root / "runs" / "detect" / project / "model_lesi" / "weights" / "best.pt",
        root / "runs" / "classify" / project / "model_penyakit" / "weights" / "best.pt",
    ]

    def describe(path: Path) -> dict:
        exists = path.is_file()
        return {
            "path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else None,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if exists else None,
            "modified_utc": datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc
            ).isoformat() if exists else None,
        }

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    try:
        git_status = subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True, stderr=subprocess.STDOUT
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_status = []

    baseline = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Immutable reference for comparison; not an independent test result.",
        "git": {"commit": git_commit, "status_at_freeze": git_status},
        "files": [describe(path) for path in tracked_files],
        "checkpoints": [describe(path) for path in checkpoint_files],
        "configuration": {
            "fish_detection_confidence": FISH_CONF_THRESHOLD,
            "lesion_detection_confidence": LESION_CONF_THRESHOLD,
            "classification_review_threshold": REVIEW_CLASSIFICATION_THRESHOLD,
            "seed": 42,
        },
        "warning": "Do not use baseline validation metrics as final test metrics.",
    }
    output = root / "reports" / "baseline_manifest.json"
    output.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"Baseline dibekukan: {output}")
    return output


def write_quality_gate(root: Path) -> Path:
    """Evaluate whether the repository has evidence for a final diagnostic claim."""
    audit_path = root / "reports" / "dataset_audit.json"
    readiness_path = root / "reports" / "presentation_readiness.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    fish_audit = audit.get("datasets", {}).get("fish4knowledge", {})
    duplicate_count = fish_audit.get("exact_duplicate_image_hashes_between_train_val", None)
    lesion_metrics_path = root / "reports" / "lesion_retrained_metrics.json"
    duplicate_condition = (
        "Fish train/val exact duplicate count must be zero before final evaluation."
        if duplicate_count != 0
        else "No exact fish train/val image duplicates were found; near-duplicate/source leakage still requires review."
    )
    gate = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_allowed": False,
        "claim": "final diagnostic system",
        "blocking_conditions": [
            "Independent, source-aware, expert-reviewed test set is not present.",
            "End-to-end ground truth labels are not present.",
            duplicate_condition,
            "Lesion validation remains weak after the low-memory retraining attempt; improve annotations/modeling before a health claim.",
        ],
        "observed": {
            "fish_exact_duplicate_hashes_train_val": duplicate_count,
            "readiness_report": str(readiness_path),
            "classification_validation_only": True,
            "lesion_retraining_report": str(lesion_metrics_path),
            "lesion_retraining_report_exists": lesion_metrics_path.is_file(),
        },
        "required_before_final_claim": [
            "Create and lock test manifest before model selection.",
            "Run baseline and DynamicAttention on identical clean splits.",
            "Report per-class errors and end-to-end metrics on the locked test set.",
            "Obtain expert review/ground truth for health claims.",
        ],
    }
    output = root / "reports" / "final_quality_gate.json"
    output.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return output


def lock_test_manifest(root: Path, seed: int = 42, test_ratio: float = 0.20) -> Path:
    """Create a deterministic, hash-locked candidate holdout manifest without moving files."""
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_locked_test_not_expert_reviewed",
        "seed": seed,
        "test_ratio_from_current_validation": test_ratio,
        "warning": "This candidate is derived from the current validation folders and is not independent until source provenance and expert labels are confirmed.",
        "datasets": {},
    }
    for dataset_name in ("fish4knowledge", "fishdisease"):
        dataset_dir = root / "datasets" / dataset_name
        image_dir = dataset_dir / "images" / "val"
        label_dir = dataset_dir / "labels" / "val"
        candidates = [
            (path, _source_group(path.stem))
            for path in _image_files(image_dir)
            if (label_dir / f"{path.stem}.txt").is_file()
        ]
        groups = sorted({group for _, group in candidates})
        selected_groups = set(random.Random(seed).sample(
            groups, max(1, int(len(groups) * test_ratio))
        )) if groups else set()
        selected = []
        for image_path, group in candidates:
            if group not in selected_groups:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            selected.append({
                "image": str(image_path),
                "label": str(label_path),
                "source_group": group,
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
            })
        manifest["datasets"][dataset_name] = {
            "selected_groups": sorted(selected_groups),
            "images": selected,
            "count": len(selected),
        }

    classification_dir = root / "datasets" / "freshwater_kaggle" / "val"
    classification_images = []
    rng = random.Random(seed)
    for class_dir in sorted(classification_dir.iterdir()) if classification_dir.exists() else []:
        images = _image_files(class_dir)
        chosen = rng.sample(images, max(1, int(len(images) * test_ratio))) if images else []
        classification_images.extend({
            "image": str(path),
            "class": class_dir.name,
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        } for path in chosen)
    manifest["datasets"]["freshwater_kaggle"] = {
        "images": classification_images,
        "count": len(classification_images),
        "note": "Derived from existing validation folders; source-aware independence is unverified.",
    }
    output = root / "reports" / "locked_test_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest test kandidat dikunci: {output}")
    return output


def _source_group(stem: str) -> str:
    """Group adjacent frames from the same source sequence when names encode it."""
    parts = stem.lower().split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else stem.lower()


def _validation_groups(items: list[tuple[str, Path]], validation_ratio: float, seed: int) -> set[str]:
    groups: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for group, path in items:
        groups[group].append((group, path))
    group_names = list(groups)
    random.Random(seed).shuffle(group_names)
    target = int(len(items) * validation_ratio)
    selected: set[str] = set()
    count = 0
    for group in group_names:
        if count >= target and selected:
            break
        selected.add(group)
        count += len(groups[group])
    return selected


def prepare_fish_dataset(root: Path, val_ratio: float = 0.2, seed: int = 42) -> None:
    """Convert masks, collect matching images, and create a validation split."""
    dataset = root / "datasets" / "fish4knowledge"
    mask_dir = dataset / "maskikan"
    train_images = dataset / "images" / "train"
    train_labels = dataset / "labels" / "train"
    val_images = dataset / "images" / "val"
    val_labels = dataset / "labels" / "val"
    staging_dir = dataset / ".fish_staging"

    source_images = {}
    for image_path in dataset.rglob("*"):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
            and "maskikan" not in image_path.parts
        ):
            source_images.setdefault(image_path.stem.lower(), image_path)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_images = staging_dir / "images"
    staging_labels = staging_dir / "labels"
    staging_images.mkdir(parents=True, exist_ok=True)
    staging_labels.mkdir(parents=True, exist_ok=True)
    staged_sources = {}
    for stem, source_path in source_images.items():
        staged_path = staging_images / source_path.name
        shutil.copy2(source_path, staged_path)
        staged_sources[stem] = staged_path
    source_images = staged_sources
    for split_dir in (train_images, train_labels, val_images, val_labels):
        if split_dir.exists():
            for child in split_dir.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
        split_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    copied_images = 0
    missing_images = 0
    pairs: list[tuple[str, Path, Path]] = []
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
        valid_contours = [contour for contour in contours if cv2.contourArea(contour) >= 4.0]
        if not valid_contours:
            continue

        label_name = mask_path.name.replace("mask_", "fish_").replace(".png", ".txt")
        label_path = staging_labels / label_name
        mask_stem = mask_path.stem.lower()
        image_stems = [
            mask_stem,
            mask_stem.removeprefix("mask_"),
            mask_stem.removeprefix("mask_").removeprefix("fish_"),
            f"fish_{mask_stem.removeprefix('mask_')}",
        ]
        image_path = next((source_images.get(stem) for stem in image_stems if source_images.get(stem)), None)
        if image_path is None:
            missing_images += 1
            continue
        target_image = staging_images / f"{label_path.stem}{image_path.suffix.lower()}"
        if not target_image.exists():
            shutil.copy2(image_path, target_image)
            copied_images += 1
        label_lines = []
        for contour in valid_contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            label_lines.append(
                f"0 {(x + box_width / 2) / width:.6f} "
                f"{(y + box_height / 2) / height:.6f} "
                f"{box_width / width:.6f} {box_height / height:.6f}"
            )
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        converted += 1
        pairs.append((_source_group(label_path.stem), label_path, target_image))

    validation_groups = _validation_groups([(group, image_path) for group, _, image_path in pairs], val_ratio, seed)
    moved = 0
    for group, label_path, image_path in pairs:
        is_validation = group in validation_groups
        target_label_dir = val_labels if is_validation else train_labels
        target_image_dir = val_images if is_validation else train_images
        shutil.move(str(label_path), str(target_label_dir / label_path.name))
        shutil.move(str(image_path), str(target_image_dir / image_path.name))
        if is_validation:
            moved += 1
    shutil.rmtree(staging_dir, ignore_errors=True)

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
        if directory.exists():
            for child in directory.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
        directory.mkdir(parents=True, exist_ok=True)

    counters = {"success": 0, "empty": 0, "missing_image": 0, "without_bbox": 0, "failed": 0}
    prepared: list[tuple[str, Path, list[str]]] = []
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
                # A page can contain many photos. Never attach an annotation to
                # an arbitrary first image when its declared image path is bad.
                image_name = Path(str(data.get("image", ""))).name
                page_dir = raw_dir / json_path.parent.name
                candidates = [page_dir / image_name] if image_name else []
                candidates.extend(
                    path for path in json_path.parent.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                    and path.stem.lower() == json_path.stem.lower()
                ) if json_path.parent.is_dir() else None
                image_path = next((path for path in candidates if path.is_file()), Path())
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
            prepared.append((_source_group(unique_stem), image_path, yolo_lines))
            counters["success"] += 1
        except (OSError, ValueError, json.JSONDecodeError) as error:
            counters["failed"] += 1
            print(f"Lewati {json_path.name}: {error}")

    validation_groups = _validation_groups(
        [(group, image_path) for group, image_path, _ in prepared], val_ratio, seed
    )
    for group, image_path, yolo_lines in prepared:
        is_validation = group in validation_groups
        image_dir = val_images if is_validation else train_images
        label_dir = val_labels if is_validation else train_labels
        output_name = f"{_source_group(image_path.parent.name)}_{image_path.stem}"
        shutil.copy2(image_path, image_dir / f"{output_name}{image_path.suffix.lower()}")
        (label_dir / f"{output_name}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")
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


def _dynamic_attention_models(root: Path):
    """Register the custom layer and return the YOLO26 DynamicAttention graphs."""
    from dynamic_attention import register_dynamic_attention

    register_dynamic_attention()
    return (
        root / "models" / "yolo26n_dynamic_attention.yaml",
        root / "models" / "yolo26n_cls_dynamic_attention.yaml",
    )


def train_models(
    root: Path,
    epochs: int = 100,
    project: str = "Runs_DynamicAttention",
    seed: int = 42,
    device: str | int | None = None,
    batch: int | None = None,
    task: str = "all",
    imgsz: int | None = None,
    resume_weights: Path | None = None,
) -> dict:
    """Train YOLO26 with DynamicAttention for selected pipeline tasks."""
    import torch
    from ultralytics import YOLO

    detection_model, classification_model = _dynamic_attention_models(root)
    fish_yaml, lesion_yaml = write_dataset_yaml(root)
    training_device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
    training_batch = batch if batch is not None else (8 if torch.cuda.is_available() else 4)
    detection_imgsz = imgsz if imgsz is not None else 640
    common = {
        "seed": seed, "deterministic": True, "resume": False,
        "device": training_device, "batch": training_batch,
        "workers": 0 if os.name == "nt" else 2,
        "amp": bool(torch.cuda.is_available()), "max_det": 100, "plots": False,
        "cache": False, "mosaic": 0.0,
    }
    results = {}
    if task in ("all", "fish"):
        fish_model = YOLO(str(resume_weights)) if resume_weights else YOLO(str(detection_model)).load("yolo26n.pt")
        results["fish_detection"] = fish_model.train(
            data=str(fish_yaml), epochs=epochs, imgsz=detection_imgsz, project=project,
            name="model_ikan_lowmem" if resume_weights else "model_ikan", **common
        )
    if task in ("all", "lesion"):
        results["lesion_detection"] = YOLO(str(detection_model)).load("yolo26n.pt").train(
            data=str(lesion_yaml), epochs=epochs, imgsz=detection_imgsz, project=project, name="model_lesi", **common
        )
    if task in ("all", "classification"):
        results["disease_classification"] = YOLO(str(classification_model)).load("yolo26n-cls.pt").train(
            data=str(root / "datasets" / "freshwater_kaggle"),
            epochs=epochs, imgsz=224, project=project, name="model_penyakit", **common
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


def _classification_metrics(model, validation_dir: Path) -> dict:
    """Calculate class-averaged metrics and inference latency on a validation split."""
    class_dirs = sorted(path for path in validation_dir.iterdir() if path.is_dir())
    class_names = [str(model.names[index]) for index in sorted(model.names)]
    name_to_index = {name: index for index, name in enumerate(class_names)}
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    latencies_ms = []
    image_count = 0

    for class_dir in class_dirs:
        if class_dir.name not in name_to_index:
            continue
        actual = name_to_index[class_dir.name]
        for image_path in _image_files(class_dir):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Gambar validasi tidak dapat dibaca: {image_path}")
            started = time.perf_counter()
            prediction = model(image, verbose=False)[0]
            latencies_ms.append((time.perf_counter() - started) * 1000)
            predicted = int(prediction.probs.top1)
            confusion[actual, predicted] += 1
            image_count += 1

    true_positive = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1).astype(float)
    predicted_support = confusion.sum(axis=0).astype(float)
    precision = np.divide(true_positive, predicted_support, out=np.zeros_like(true_positive), where=predicted_support != 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support != 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(true_positive), where=(precision + recall) != 0,
    )
    total = float(confusion.sum())
    accuracy = float(true_positive.sum() / total) if total else 0.0
    weights = support / total if total else np.zeros_like(support)
    return {
        "image_count": image_count,
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()) if len(precision) else 0.0,
        "macro_recall": float(recall.mean()) if len(recall) else 0.0,
        "macro_f1": float(f1.mean()) if len(f1) else 0.0,
        "weighted_precision": float((precision * weights).sum()),
        "weighted_recall": float((recall * weights).sum()),
        "weighted_f1": float((f1 * weights).sum()),
        "latency_ms_mean": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "latency_ms_p50": float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0,
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
        "latency_note": "Per image, includes preprocessing and model inference; excludes model loading and disk I/O.",
    }


def evaluate_models(root: Path, project: str = "Runs_DynamicAttention") -> dict:
    """Run validation and export numeric metrics for all trained models."""
    from ultralytics import YOLO

    _dynamic_attention_models(root)

    detect_root = root / "runs" / "detect" / project
    fish_runs = sorted(
        detect_root.glob("model_ikan*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    fish_weights = fish_runs[-1] if fish_runs else detect_root / "model_ikan" / "weights" / "best.pt"
    model_specs = {
        "fish_detection": (fish_weights, root / "data_ikan_generated.yaml"),
        "lesion_detection": (root / "runs" / "detect" / project / "model_lesi" / "weights" / "best.pt", root / "data_lesi_generated.yaml"),
        "disease_classification": (root / "runs" / "classify" / project / "model_penyakit" / "weights" / "best.pt", root / "datasets" / "freshwater_kaggle"),
    }
    report = {
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
    }
    for name, (weights, data) in model_specs.items():
        if not weights.exists():
            report[name] = {"status": "checkpoint tidak ditemukan", "weights": str(weights)}
            continue
        model = YOLO(str(weights))
        metrics = model.val(
            data=str(data), plots=True, project=str(root / "reports"), name=f"val_{name}",
            workers=0 if os.name == "nt" else 2,
        )
        if name == "disease_classification":
            classification_metrics = _classification_metrics(model, data / "val")
            report[name] = {
                "status": "ok", "top1": _metric_value(metrics, "top1", "accuracy_top1"),
                "top5": _metric_value(metrics, "top5", "accuracy_top5"), "weights": str(weights),
                **classification_metrics,
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


def plot_training_curves(root: Path, project: str = "Runs_DynamicAttention") -> list[Path]:
    """Create readable training curve images from Ultralytics results.csv files."""
    import matplotlib.pyplot as plt
    import pandas as pd

    output_paths = []
    for model_name in ("model_ikan", "model_lesi", "model_penyakit"):
        task_root = "classify" if model_name == "model_penyakit" else "detect"
        results_csv = root / "runs" / task_root / project / model_name / "results.csv"
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


def classification_confusion_matrix(root: Path, project: str = "Runs_DynamicAttention") -> Path | None:
    """Build and save a confusion matrix from the classification validation split."""
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    _dynamic_attention_models(root)

    weights = root / "runs" / "classify" / project / "model_penyakit" / "weights" / "best.pt"
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
    lines = ["# Laporan Eksperimen YOLOComVis", "", "## Konfigurasi", "- Task: deteksi ikan, deteksi lesi, klasifikasi penyakit", "- Split deteksi: train/val 80:20", "- Seed split dan training: 42", "- Model: YOLO26n dan YOLO26n-cls", "- Attention: DynamicAttention pada fitur backbone terdalam", "", "## Dataset"]
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


def run_dashboard(
    root: Path,
    image_path: Path,
    output_path: Path | None = None,
    fish_weights: Path | None = None,
) -> Path:
    """Run the three models and save a notebook-friendly dashboard image."""
    from ultralytics import YOLO

    from dynamic_attention import register_dynamic_attention

    register_dynamic_attention()
    detect_dir = root / "runs" / "detect" / "Runs_DynamicAttention"
    classify_dir = root / "runs" / "classify" / "Runs_DynamicAttention"
    selected_fish_weights = fish_weights or root / DEFAULT_FISH_CHECKPOINT
    if not selected_fish_weights.exists():
        raise FileNotFoundError(f"Checkpoint deteksi ikan tidak ditemukan: {selected_fish_weights}")
    fish_model = YOLO(str(selected_fish_weights))
    lesion_model = YOLO(str(detect_dir / "model_lesi" / "weights" / "best.pt"))
    classifier = YOLO(str(classify_dir / "model_penyakit" / "weights" / "best.pt"))
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    result = fish_model(image, conf=FISH_CONF_THRESHOLD, verbose=False)[0]
    canvas = image.copy()
    panel_width = 440
    panel = np.full((max(image.shape[0], 520), panel_width, 3), (30, 30, 35), dtype=np.uint8)
    canvas = np.hstack((cv2.copyMakeBorder(
        canvas, 0, panel.shape[0] - canvas.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    ), panel))
    y_text = 38
    cv2.putText(canvas, "HASIL ANALISIS IKAN", (image.shape[1] + 20, y_text), cv2.FONT_HERSHEY_DUPLEX, 0.66, (0, 215, 255), 2)
    y_text += 40

    if len(result.boxes) == 0:
        cv2.putText(canvas, "Ikan: tidak terdeteksi", (image.shape[1] + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 180, 255), 1)
        y_text += 30

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0]) * 100
        crop = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
        if crop.size == 0:
            continue
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 215, 255), 2)
        lesion_result = lesion_model(crop, conf=LESION_CONF_THRESHOLD, verbose=False)[0]
        lesion_count = len(lesion_result.boxes)
        for lesion in lesion_result.boxes:
            lx1, ly1, lx2, ly2 = map(int, lesion.xyxy[0].tolist())
            lesion_confidence = float(lesion.conf[0]) * 100
            cv2.rectangle(canvas, (x1 + lx1, y1 + ly1), (x1 + lx2, y1 + ly2), (0, 0, 255), 2)
            cv2.putText(canvas, f"lesi {lesion_confidence:.0f}%", (x1 + lx1, max(12, y1 + ly1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
        classification = classifier(crop, verbose=False)[0]
        top_indices = classification.probs.top5[:3]
        cv2.putText(canvas, f"Deteksi ikan: confidence {confidence:.1f}%", (image.shape[1] + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
        y_text += 30
        cv2.putText(canvas, f"Deteksi lesi: {lesion_count} kandidat", (image.shape[1] + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        y_text += 32
        for index in top_indices:
            name = classifier.names[int(index)]
            probability = float(classification.probs.data[int(index)]) * 100
            cv2.putText(canvas, f"Deteksi penyakit: {name} ({probability:.1f}%)", (image.shape[1] + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 150), 1)
            y_text += 24
        y_text += 18

    destination = output_path or root / "hasil_dashboard_dynamic_attention.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), canvas)
    print(f"Dashboard disimpan di: {destination}")
    return destination


def run_batch_dashboards(
    root: Path,
    input_dir: Path,
    output_dir: Path | None = None,
    limit: int | None = None,
) -> Path:
    """Generate dashboards and a prediction summary for every image in a folder."""
    import csv
    from ultralytics import YOLO

    from dynamic_attention import register_dynamic_attention

    register_dynamic_attention()
    input_dir = Path(input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Folder gambar tidak ditemukan: {input_dir}")
    output_dir = (output_dir or root / "reports" / "batch_dashboards").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detect_dir = root / "runs" / "detect" / "Runs_DynamicAttention"
    classify_dir = root / "runs" / "classify" / "Runs_DynamicAttention"
    fish_weights = root / DEFAULT_FISH_CHECKPOINT
    if not fish_weights.exists():
        raise FileNotFoundError(f"Checkpoint deteksi ikan tidak ditemukan: {fish_weights}")
    fish_model = YOLO(str(fish_weights))
    lesion_model = YOLO(str(detect_dir / "model_lesi" / "weights" / "best.pt"))
    classifier = YOLO(str(classify_dir / "model_penyakit" / "weights" / "best.pt"))

    image_paths = _image_files(input_dir)
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError(f"Tidak ada gambar yang didukung di {input_dir}")

    rows = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            rows.append({"image": str(image_path), "status": "gagal_dibaca"})
            continue
        fish_result = fish_model(image, conf=FISH_CONF_THRESHOLD, verbose=False)[0]
        fish_count = len(fish_result.boxes)
        fish_confidences = [float(value) for value in fish_result.boxes.conf.tolist()]
        lesion_count = 0
        fish_predictions = []
        for box in fish_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
            if crop.size == 0:
                continue
            lesion_result = lesion_model(crop, conf=LESION_CONF_THRESHOLD, verbose=False)[0]
            lesion_count += len(lesion_result.boxes)
            classification = classifier(crop, verbose=False)[0]
            class_index = int(classification.probs.top1)
            top_probability = float(classification.probs.data[class_index])
            fish_predictions.append({
                "class": str(classifier.names[class_index]),
                "confidence": round(top_probability, 6),
                "status": "model_result" if top_probability >= REVIEW_CLASSIFICATION_THRESHOLD else "review_manual",
            })
        dashboard_path = output_dir / f"{image_path.stem}_dashboard.jpg"
        run_dashboard(root, image_path, dashboard_path)
        rows.append({
            "image": str(image_path), "dashboard": str(dashboard_path), "status": "ok",
            "fish_count": fish_count,
            "mean_fish_confidence": round(sum(fish_confidences) / len(fish_confidences), 6) if fish_confidences else None,
            "lesion_count": lesion_count, "fish_predictions": json.dumps(fish_predictions, ensure_ascii=True),
        })

    summary_path = output_dir / "batch_predictions.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Batch dashboard selesai: {len(rows)} gambar")
    print(f"Ringkasan prediksi: {summary_path}")
    return summary_path


def export_to_drive(source_dir: Path, drive_dir: Path) -> Path:
    """Copy reports, checkpoints, and dashboards to a mounted Google Drive folder."""
    source_dir = Path(source_dir)
    drive_dir = Path(drive_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Folder sumber tidak ditemukan: {source_dir}")
    drive_dir.mkdir(parents=True, exist_ok=True)
    destination = drive_dir / source_dir.name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_dir, destination)
    print(f"Artefak disalin ke Google Drive: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mode", choices=("prepare", "audit", "freeze-baseline", "lock-test", "train", "evaluate", "report", "dashboard", "batch-dashboard", "all"), default="prepare")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--task", choices=("all", "fish", "lesion", "classification"), default="all")
    parser.add_argument("--device", default=None, help="CUDA device index, for example 0; auto-detect by default")
    parser.add_argument("--batch", type=int, default=None, help="Batch size; auto-select by device memory by default")
    parser.add_argument("--imgsz", type=int, default=None, help="Resolusi training deteksi; turunkan jika RAM terbatas")
    parser.add_argument("--resume-weights", type=Path, default=None, help="Lanjutkan dari bobot model sebagai inisialisasi tanpa resume konfigurasi lama")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Path output dashboard untuk mode dashboard")
    parser.add_argument("--fish-weights", type=Path, default=None, help="Checkpoint detektor ikan alternatif untuk dashboard")
    parser.add_argument("--input-dir", type=Path, default=None, help="Folder gambar untuk mode batch-dashboard")
    parser.add_argument("--output-dir", type=Path, default=None, help="Folder output dashboard batch")
    parser.add_argument("--limit", type=int, default=None, help="Jumlah gambar maksimum pada mode batch-dashboard")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    if args.mode in ("prepare", "all"):
        prepare_fish_dataset(root, seed=args.seed)
        prepare_lesion_dataset(root, seed=args.seed)
        prepare_classification_dataset(root, seed=args.seed)
        save_dataset_report(root)
    if args.mode in ("audit", "all"):
        audit_datasets(root)
        write_readiness_report(root)
        write_checkpoint_provenance(root)
        write_quality_gate(root)
    if args.mode in ("freeze-baseline",):
        freeze_baseline(root)
    if args.mode in ("lock-test",):
        lock_test_manifest(root, seed=args.seed)
    if args.mode in ("train", "all"):
        train_models(
            root, epochs=args.epochs, seed=args.seed, device=args.device, batch=args.batch,
            task=args.task, imgsz=args.imgsz, resume_weights=args.resume_weights,
        )
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
        output_path = args.output if args.mode == "dashboard" else None
        run_dashboard(root, image_path, output_path, args.fish_weights)
    if args.mode == "batch-dashboard":
        if args.input_dir is None:
            parser.error("--input-dir wajib diisi untuk mode batch-dashboard")
        run_batch_dashboards(root, args.input_dir, args.output_dir, args.limit)


if __name__ == "__main__":
    main()

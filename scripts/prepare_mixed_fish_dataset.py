from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def copy_split(
    source_images: Path,
    source_labels: Path,
    dest_images: Path,
    dest_labels: Path,
    prefix: str,
    map_to_fish: bool,
    limit: int | None = None,
) -> int:
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    copied = 0
    candidates = sorted(source_images.glob("*"))
    if limit is not None:
        candidates = candidates[:limit]
    for image in candidates:
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label = source_labels / f"{image.stem}.txt"
        if not label.exists():
            raise FileNotFoundError(f"Missing label for {image}")
        target_stem = f"{prefix}_{image.stem}"
        shutil.copy2(image, dest_images / f"{target_stem}{image.suffix.lower()}")
        lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        normalized = []
        for line in lines:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO label in {label}: {line}")
            class_id = int(fields[0])
            if map_to_fish:
                class_id = 0
            if class_id != 0:
                raise ValueError(f"Unexpected class {class_id} in {label}")
            normalized.append(" ".join([str(class_id), *fields[1:]]))
        (dest_labels / f"{target_stem}.txt").write_text(
            "\n".join(normalized) + ("\n" if normalized else ""), encoding="utf-8"
        )
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fish4k-train-limit", type=int, default=None)
    parser.add_argument("--fish4k-val-limit", type=int, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    external = root / "input_uji_coba" / "Fish.v1-416x416.yolo26"
    fish4k = root / "datasets" / "fish4knowledge"
    output = (args.output or root / "datasets" / "fish_mixed_v1").resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")

    counts = {}
    for split, external_split, fish4k_split in (("train", "train", "train"), ("val", "valid", "val")):
        counts[f"fish4knowledge_{split}"] = copy_split(
            fish4k / "images" / fish4k_split,
            fish4k / "labels" / fish4k_split,
            output / "images" / split,
            output / "labels" / split,
            f"f4k_{split}",
            False,
            args.fish4k_train_limit if split == "train" else args.fish4k_val_limit,
        )
        counts[f"fish_v1_{split}"] = copy_split(
            external / external_split / "images",
            external / external_split / "labels",
            output / "images" / split,
            output / "labels" / split,
            f"fishv1_{split}",
            True,
        )
    counts["fish_v1_test"] = copy_split(
        external / "test" / "images",
        external / "test" / "labels",
        output / "images" / "test",
        output / "labels" / "test",
        "fishv1_test",
        True,
    )
    yaml_path = output / "data.yaml"
    yaml_path.write_text(
        f"path: {output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: fish\n",
        encoding="utf-8",
    )
    (output / "preparation_manifest.json").write_text(
        json.dumps(
            {
                "source_external": str(external),
                "source_fish4knowledge": str(fish4k),
                "output": str(output),
                "class_mapping": "all source classes mapped to class 0 fish",
                "fish4k_limits": {
                    "train": args.fish4k_train_limit,
                    "val": args.fish4k_val_limit,
                },
                "counts": counts,
                "test_policy": "Fish v1 test remains held out and is not included in train or val.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "yolo26_baseline_metrics.json"
VALIDATION_FRACTION = 1.0


def evaluate() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name, model_name, data, image_size in (
        ("fish_detection", "yolo26n.pt", ROOT / "data_ikan.yaml", 640),
        ("lesion_detection", "yolo26n.pt", ROOT / "data_lesi.yaml", 640),
        ("disease_classification", "yolo26n-cls.pt", ROOT / "datasets" / "freshwater_kaggle", 224),
    ):
        model = YOLO(model_name)
        result = model.val(
            data=str(data), imgsz=image_size, batch=32, workers=0,
            fraction=VALIDATION_FRACTION, plots=True, verbose=False,
        )
        if name == "disease_classification":
            metrics[name] = {
                "status": "zero-shot-pretrained-full-validation",
                "model": model_name, "fraction": VALIDATION_FRACTION,
                "top1": float(result.top1), "top5": float(result.top5),
            }
        else:
            box = result.box
            metrics[name] = {
                "status": "zero-shot-pretrained-full-validation",
                "model": model_name,
                "fraction": VALIDATION_FRACTION,
                "precision": float(box.mp),
                "recall": float(box.mr),
                "map50": float(box.map50),
                "map50_95": float(box.map),
            }
    OUTPUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    evaluate()
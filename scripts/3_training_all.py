from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynamic_attention import register_dynamic_attention
from ultralytics import YOLO

EPOCHS = 100
SEED = 42
PROJECT = "Runs_DynamicAttention"
DETECTION_MODEL = ROOT / "models" / "yolo26n_dynamic_attention.yaml"
CLASSIFICATION_MODEL = ROOT / "models" / "yolo26n_cls_dynamic_attention.yaml"

register_dynamic_attention()


def main() -> None:
    print("--- 1. TRAINING DETEKSI IKAN ---")
    model_ikan = YOLO(str(DETECTION_MODEL)).load("yolo26n.pt")
    model_ikan.train(
        data=str(ROOT / "data_ikan_generated.yaml"), epochs=EPOCHS, imgsz=640,
        project=PROJECT, name="model_ikan", seed=SEED, deterministic=True, resume=False, workers=0
    )

    print("--- 2. TRAINING DETEKSI LESI (LUKA) ---")
    model_lesi = YOLO(str(DETECTION_MODEL)).load("yolo26n.pt")
    model_lesi.train(
        data=str(ROOT / "data_lesi_generated.yaml"), epochs=EPOCHS, imgsz=640,
        project=PROJECT, name="model_lesi", seed=SEED, deterministic=True, resume=False, workers=0
    )

    print("--- 3. TRAINING KLASIFIKASI PENYAKIT (KAGGLE) ---")
    model_penyakit = YOLO(str(CLASSIFICATION_MODEL)).load("yolo26n-cls.pt")
    model_penyakit.train(
        data=str(ROOT / "datasets" / "freshwater_kaggle"), epochs=EPOCHS, imgsz=224,
        project=PROJECT, name="model_penyakit", seed=SEED, deterministic=True, resume=False, workers=0
    )


if __name__ == "__main__":
    main()
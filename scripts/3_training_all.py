from ultralytics import YOLO

EPOCHS = 10
SEED = 42
PROJECT = "Runs_Baseline"

print("--- 1. TRAINING DETEKSI IKAN ---")
model_ikan = YOLO("yolo26n.pt")
model_ikan.train(
    data="D:/yolocomvis/data_ikan.yaml", epochs=EPOCHS, imgsz=640,
    project=PROJECT, name="model_ikan", seed=SEED, deterministic=True, resume=False
)

print("--- 2. TRAINING DETEKSI LESI (LUKA) ---")
model_lesi = YOLO("yolo26n.pt")
model_lesi.train(
    data="D:/yolocomvis/data_lesi.yaml", epochs=EPOCHS, imgsz=640,
    project=PROJECT, name="model_lesi", seed=SEED, deterministic=True, resume=False
)

print("--- 3. TRAINING KLASIFIKASI PENYAKIT (KAGGLE) ---")
model_penyakit = YOLO("yolo26n-cls.pt")
model_penyakit.train(
    data="D:/yolocomvis/datasets/freshwater_kaggle", epochs=EPOCHS, imgsz=224,
    project=PROJECT, name="model_penyakit", seed=SEED, deterministic=True, resume=False
)
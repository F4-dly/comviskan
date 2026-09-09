from ultralytics import YOLO

print("--- 1. TRAINING DETEKSI IKAN ---")
model_ikan = YOLO("yolo11n.pt") 
model_ikan.train(
    data="D:/yolocomvis/data_ikan.yaml",
    epochs=10, imgsz=640, project="Runs_Baseline", name="model_ikan"
)

print("--- 2. TRAINING DETEKSI LESI (LUKA) ---")
model_lesi = YOLO("yolo11n.pt")
model_lesi.train(
    data="D:/yolocomvis/data_lesi.yaml",
    epochs=10, imgsz=640, project="Runs_Baseline", name="model_lesi"
)

print("--- 3. TRAINING KLASIFIKASI PENYAKIT (KAGGLE) ---")
model_penyakit = YOLO("yolo11n-cls.pt") # Perhatikan '-cls' untuk klasifikasi
model_penyakit.train(
    data="D:/yolocomvis/datasets/freshwater_kaggle", # Langsung tunjuk folder
    epochs=10, imgsz=224, project="Runs_Baseline", name="model_penyakit"
)
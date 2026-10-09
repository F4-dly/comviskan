# Laporan Eksperimen YOLOComVis

## Konfigurasi
- Task: deteksi ikan, deteksi lesi, klasifikasi penyakit
- Split deteksi: train/val 80:20
- Seed split dan training: 42
- Model: YOLO26n dan YOLO26n-cls
- Attention: DynamicAttention pada fitur backbone terdalam

## Dataset
### fish_detection
```json
{
  "path": "D:\\yolocomvis\\datasets\\fish4knowledge",
  "splits": {
    "train": {
      "images": 9666,
      "labels": 9666,
      "objects": 9666,
      "images_without_labels": 0,
      "labels_without_images": 0
    },
    "val": {
      "images": 2436,
      "labels": 2436,
      "objects": 2436,
      "images_without_labels": 0,
      "labels_without_images": 0
    }
  }
}
```
### lesion_detection
```json
{
  "path": "D:\\yolocomvis\\datasets\\fishdisease",
  "splits": {
    "train": {
      "images": 922,
      "labels": 922,
      "objects": 1766,
      "images_without_labels": 0,
      "labels_without_images": 0
    },
    "val": {
      "images": 233,
      "labels": 233,
      "objects": 380,
      "images_without_labels": 0,
      "labels_without_images": 0
    }
  }
}
```
### disease_classification
```json
{
  "path": "D:\\yolocomvis\\datasets\\freshwater_kaggle",
  "classes": {
    "train": {
      "Bacterial diseases - Aeromoniasis": 250,
      "Bacterial gill disease": 250,
      "Bacterial Red disease": 250,
      "Fungal diseases Saprolegniasis": 250,
      "Healthy Fish": 250,
      "Parasitic diseases": 250,
      "Viral diseases White tail disease": 250
    },
    "val": {
      "Bacterial diseases - Aeromoniasis": 100,
      "Bacterial gill disease": 100,
      "Bacterial Red disease": 100,
      "Fungal diseases Saprolegniasis": 100,
      "Healthy Fish": 100,
      "Parasitic diseases": 100,
      "Viral diseases White tail disease": 100
    }
  }
}
```

## Hasil Evaluasi
```json
{
  "evaluation_timestamp_utc": "2026-10-09T02:32:32.647108+00:00",
  "runtime": {
    "python": "3.14.4",
    "platform": "Windows-11-10.0.26300-SP0"
  },
  "fish_detection": {
    "status": "ok",
    "precision": 0.9989171128943028,
    "recall": 0.9983579638752053,
    "map50": 0.9949876644736843,
    "map50_95": 0.821474928434923,
    "weights": "D:\\yolocomvis\\runs\\detect\\Runs_DynamicAttention\\model_ikan_lowmem\\weights\\best.pt"
  },
  "lesion_detection": {
    "status": "ok",
    "precision": 0.6925376631398874,
    "recall": 0.45,
    "map50": 0.5445718757240362,
    "map50_95": 0.23530033246692858,
    "weights": "D:\\yolocomvis\\runs\\detect\\Runs_DynamicAttention\\model_lesi\\weights\\best.pt"
  },
  "disease_classification": {
    "status": "ok",
    "top1": 0.9942857027053833,
    "top5": 1.0,
    "weights": "D:\\yolocomvis\\runs\\classify\\Runs_DynamicAttention\\model_penyakit\\weights\\best.pt",
    "image_count": 700,
    "accuracy": 0.9942857142857143,
    "macro_precision": 0.9943422913719944,
    "macro_recall": 0.9942857142857143,
    "macro_f1": 0.9942927144607187,
    "weighted_precision": 0.9943422913719941,
    "weighted_recall": 0.9942857142857142,
    "weighted_f1": 0.9942927144607185,
    "latency_ms_mean": 6.625301285634383,
    "latency_ms_p50": 5.87305000226479,
    "latency_ms_p95": 10.645890002342632,
    "latency_note": "Per image, includes preprocessing and model inference; excludes model loading and disk I/O."
  }
}
```

## Artefak
- `dataset_summary.json/csv/md`: statistik dataset
- `evaluation_metrics.json`: metrik validasi
- `training_curves_*.png`: kurva training
- `confusion_matrix_classification.png`: confusion matrix klasifikasi
- `val_*/`: plot validasi Ultralytics
- `hasil_dashboard_*.jpg`: visualisasi inferensi
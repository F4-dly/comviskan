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
      "images": 5374,
      "labels": 9690,
      "objects": 9690,
      "images_without_labels": 0,
      "labels_without_images": 4316
    },
    "val": {
      "images": 6728,
      "labels": 2422,
      "objects": 2422,
      "images_without_labels": 4309,
      "labels_without_images": 3
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
      "images": 929,
      "labels": 929,
      "objects": 1705,
      "images_without_labels": 0,
      "labels_without_images": 0
    },
    "val": {
      "images": 226,
      "labels": 226,
      "objects": 441,
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
  "fish_detection": {
    "status": "ok",
    "precision": 0.3600949593008111,
    "recall": 0.9968193835521021,
    "map50": 0.3757105942259617,
    "map50_95": 0.34459740178248155,
    "weights": "D:\\yolocomvis\\runs\\detect\\Runs_DynamicAttention\\model_ikan-4\\weights\\best.pt"
  },
  "lesion_detection": {
    "status": "ok",
    "precision": 0.5966513727985204,
    "recall": 0.45283147900271203,
    "map50": 0.4898693439184679,
    "map50_95": 0.1913380153173588,
    "weights": "D:\\yolocomvis\\runs\\detect\\Runs_DynamicAttention\\model_lesi\\weights\\best.pt"
  },
  "disease_classification": {
    "status": "ok",
    "top1": 0.9942857027053833,
    "top5": 1.0,
    "weights": "D:\\yolocomvis\\runs\\classify\\Runs_DynamicAttention\\model_penyakit\\weights\\best.pt"
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
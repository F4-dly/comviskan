# Ringkasan Dataset

| Dataset | Split | Gambar | Label | Objek |
|---|---:|---:|---:|---:|
| fish_detection | train | 9682 | 9682 | 9682 |
| fish_detection | val | 2420 | 2420 | 2420 |
| lesion_detection | train | 929 | 929 | 1705 |
| lesion_detection | val | 226 | 226 | 441 |

> Periksa field `images_without_labels` dan `labels_without_images` pada JSON untuk menemukan pasangan file yang tidak cocok.

## Klasifikasi Penyakit

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
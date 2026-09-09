import os
import glob
import json
import cv2
import shutil
import random

# UBAH PATH INI SESUAI LOKASI FOLDER MENTAH ANDA
dir_mentah = "D:/yolocomvis/datasets/fishdisease_mentah/"

# Folder tujuan akhir untuk YOLO
dir_tujuan = "D:/yolocomvis/datasets/fishdisease/"
img_train_dir = os.path.join(dir_tujuan, "images/train")
img_val_dir = os.path.join(dir_tujuan, "images/val")
lbl_train_dir = os.path.join(dir_tujuan, "labels/train")
lbl_val_dir = os.path.join(dir_tujuan, "labels/val")

for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
    os.makedirs(d, exist_ok=True)

print("Mulai memindai dan memproses file JSON anotasi...")

semua_json = glob.glob(os.path.join(dir_mentah, "**/*.json"), recursive=True)
print(f"Ditemukan {len(semua_json)} file JSON.")

sukses = 0
skip_kosong = 0
skip_gambar = 0
skip_tanpa_bbox = 0
for path_json in semua_json:
    try:
        with open(path_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "annotations" not in data or not data["annotations"]:
            skip_kosong += 1
            continue
            
        # Field image menunjuk ke folder page di root dataset, bukan folder annos.
        path_gambar_asli = os.path.join(dir_mentah, data["image"])
        if not os.path.isfile(path_gambar_asli):
            # Fallback untuk JSON lama yang hanya memiliki nama file gambar.
            folder_page = os.path.join(dir_mentah, os.path.basename(os.path.dirname(path_json)))
            gambar_list = glob.glob(os.path.join(folder_page, "*.jp*g")) + glob.glob(os.path.join(folder_page, "*.png"))
            if not gambar_list:
                skip_gambar += 1
                continue
            path_gambar_asli = gambar_list[0]

        if not os.path.isfile(path_gambar_asli):
            skip_gambar += 1
            continue

        folder_page = os.path.dirname(path_gambar_asli)
        nama_file_asli = os.path.basename(path_gambar_asli)
        
        # Buat nama unik untuk file output agar tidak bertabrakan antar page (misal: page10_img1.txt)
        nama_page = os.path.basename(folder_page)
        nama_unik = f"{nama_page}_{os.path.splitext(nama_file_asli)[0]}"
        
        # Baca ukuran gambar asli
        img = cv2.imread(path_gambar_asli)
        if img is None:
            continue
        tinggi_img, lebar_img, _ = img.shape
        
        baris_yolo = []
        for ann in data["annotations"]:
            if "bbox" in ann:
                x_min, y_min, x_max, y_max = ann["bbox"]
                
                # Konversi ke format YOLO (x_center, y_center, w, h) dinormalisasi (0-1)
                x_center = ((x_min + x_max) / 2.0) / lebar_img
                y_center = ((y_min + y_max) / 2.0) / tinggi_img
                w = (x_max - x_min) / lebar_img
                h = (y_max - y_min) / tinggi_img
                
                # Kelas 0 untuk 'lesion'
                baris_yolo.append(f"0 {x_center} {y_center} {w} {h}")
        
        if baris_yolo:
            # Bagi data secara acak: 80% train, 20% val
            is_val = random.random() < 0.2
            tujuan_img = img_val_dir if is_val else img_train_dir
            tujuan_lbl = lbl_val_dir if is_val else lbl_train_dir
            
            # Salin gambar dan buat file teks label dengan nama unik
            shutil.copy(path_gambar_asli, os.path.join(tujuan_img, f"{nama_unik}{os.path.splitext(nama_file_asli)[1]}"))
            with open(os.path.join(tujuan_lbl, f"{nama_unik}.txt"), 'w') as f:
                f.write("\n".join(baris_yolo))
                
            sukses += 1
        else:
            skip_tanpa_bbox += 1
    except Exception as e:
        print(f"Gagal memproses {path_json}: {e}")
        continue

print(f"Selesai! Berhasil memproses {sukses} data lesi ke folder YOLO.")
print(f"Dilewati: {skip_kosong} tanpa anotasi, {skip_gambar} tanpa gambar, {skip_tanpa_bbox} tanpa bbox.")
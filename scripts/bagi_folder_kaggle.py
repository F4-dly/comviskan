import splitfolders

# Folder asli hasil download dari Kaggle yang berisi 7 folder penyakit
folder_asli_kaggle = "../datasets/freshwater_asli/" 

# Folder tujuan yang sudah dibagi untuk YOLO
folder_tujuan = "../datasets/freshwater_kaggle_yolo/"

# Membagi data: 80% untuk belajar (train), 20% untuk validasi/ujian (val)
splitfolders.ratio(folder_asli_kaggle, output=folder_tujuan, seed=42, ratio=(0.8, 0.2))

print("Selesai! Folder Kaggle siap digunakan untuk training YOLO-cls")
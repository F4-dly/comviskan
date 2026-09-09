import os
import glob
import json

dir_mentah = "D:/yolocomvis/datasets/fishdisease_mentah/"

# Cari 1 file json pertama yang ketemu di subfolder page...
semua_json = glob.glob(os.path.join(dir_mentah, "**/*.json"), recursive=True)

if semua_json:
    file_pertama = semua_json[0]
    print(f"Contoh nama file JSON: {file_pertama}")
    
    with open(file_pertama, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("Isi struktur kunci (keys) dari JSON tersebut adalah:")
    print(list(data.keys()))
else:
    print("Tidak ditemukan file JSON sama sekali di dalam folder mentah!")
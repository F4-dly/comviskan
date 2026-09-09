import os

# Folder tempat file .txt Anda berada
folder_labels = "D:/yolocomvis/datasets/fish4knowledge/labels/train/"

# Variabel untuk menghitung berapa yang berhasil diubah
jumlah_diubah = 0

print("Mulai mengubah nama file...")

# Mengecek semua file di dalam folder label
for nama_file in os.listdir(folder_labels):
    # Jika file berawalan 'mask_' dan berakhiran '.txt'
    if nama_file.startswith("mask_") and nama_file.endswith(".txt"):
        
        # Ganti kata 'mask_' menjadi 'fish_'
        nama_baru = nama_file.replace("mask_", "fish_")
        
        # Proses mengubah nama file
        path_lama = os.path.join(folder_labels, nama_file)
        path_baru = os.path.join(folder_labels, nama_baru)
        
        os.rename(path_lama, path_baru)
        jumlah_diubah += 1

print(f"Selesai! Berhasil mengubah nama {jumlah_diubah} file .txt.")
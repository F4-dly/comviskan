import cv2
import os
import glob

# PERBAIKAN: Tambahkan '/' di bagian akhir folder_mask
folder_mask = "D:/yolocomvis/datasets/fish4knowledge/maskikan/" 
folder_label_tujuan = "D:/yolocomvis/datasets/fish4knowledge/labels/train/"

os.makedirs(folder_label_tujuan, exist_ok=True)

# Mencari semua gambar mask
daftar_mask = glob.glob(folder_mask + "*.png")

# Tambahan untuk mengecek apakah gambar berhasil ditemukan
print(f"Ditemukan {len(daftar_mask)} gambar mask. Memulai konversi...")

for path_mask in daftar_mask:
    nama_file = os.path.basename(path_mask).replace(".png", ".txt")
    
    # Baca gambar mask (hitam putih)
    mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
    tinggi, lebar = mask.shape
    
    # Cari kontur (garis tepi) dari warna putih (ikan)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Ambil tepi paling luar untuk dibuatkan kotak pembatas (bounding box)
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # YOLO butuh titik tengah kotak, bukan titik sudut. Ini rumus konversinya:
        x_center = (x + (w / 2)) / lebar
        y_center = (y + (h / 2)) / tinggi
        w_norm = w / lebar
        h_norm = h / tinggi
        
        # Tulis ke file .txt (angka 0 di depan artinya kelas ke-0 yaitu 'fish')
        with open(folder_label_tujuan + nama_file, 'w') as f:
            f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}\n")
            
print("Selesai! Cek folder labels/train Anda.")
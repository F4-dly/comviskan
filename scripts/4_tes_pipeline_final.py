import cv2
import numpy as np
import os
from ultralytics import YOLO

# ==========================================
# 1. SETUP MODEL & WARNA TEMA
# ==========================================
path_ikan = "D:/yolocomvis/runs/detect/Runs_Baseline/model_ikan/weights/best.pt"
path_lesi = "D:/yolocomvis/runs/detect/Runs_Baseline/model_lesi/weights/best.pt"
path_penyakit = "D:/yolocomvis/runs/classify/Runs_Baseline/model_penyakit/weights/best.pt"

deteksi_ikan = YOLO(path_ikan)
deteksi_lesi = YOLO(path_lesi)
klasifikasi = YOLO(path_penyakit)

COLOR_BG_PANEL = (30, 30, 35)      
COLOR_TEXT_MAIN = (255, 255, 255)  
COLOR_TEXT_SUB = (180, 180, 180)   
COLOR_BOX_FISH = (0, 215, 255)     
COLOR_BOX_LESI = (0, 0, 255)       

# ==========================================
# 2. BACA GAMBAR & BUAT KANVAS DASHBOARD
# ==========================================
path_gambar_tes = "D:/yolocomvis/ujicoba10.png"
gambar = cv2.imread(path_gambar_tes)

if gambar is None:
    print(f"Error: Gambar {path_gambar_tes} tidak ditemukan.")
    exit()

img_h, img_w = gambar.shape[:2]
panel_w = 450 
canvas_h = max(img_h, 500) 

canvas = np.zeros((canvas_h, img_w + panel_w, 3), dtype=np.uint8)
canvas[:] = COLOR_BG_PANEL
canvas[0:img_h, 0:img_w] = gambar

# ==========================================
# 3. PROSES INFERENSI & MENGGAMBAR VISUAL
# ==========================================
hasil_ikan = deteksi_ikan(gambar)[0]

y_text = 40
cv2.putText(canvas, "YOLOv26 BASELINE ANALYSIS", (img_w + 20, y_text), cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_BOX_FISH, 2)
y_text += 40

if len(hasil_ikan.boxes) == 0:
    cv2.putText(canvas, "Tidak ada ikan terdeteksi.", (img_w + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_MAIN, 1)
else:
    for kotak in hasil_ikan.boxes:
        x1, y1, x2, y2 = map(int, kotak.xyxy[0])
        conf_ikan = float(kotak.conf[0]) * 100
        
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_BOX_FISH, 2)
        label_ikan = f"Fish {conf_ikan:.0f}%"
        (lw, lh), _ = cv2.getTextSize(label_ikan, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_label = y1 - 20 if y1 - 20 > 0 else y1 + 10 
        cv2.rectangle(canvas, (x1, y_label), (x1 + lw, y_label + lh + 5), COLOR_BOX_FISH, -1)
        cv2.putText(canvas, label_ikan, (x1, y_label + lh + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        potongan_ikan = gambar[y1:y2, x1:x2]

        hasil_lesi = deteksi_lesi(potongan_ikan)[0]
        jumlah_luka = len(hasil_lesi.boxes)
        
        for luka in hasil_lesi.boxes:
            lx1, ly1, lx2, ly2 = map(int, luka.xyxy[0])
            cv2.rectangle(canvas, (x1 + lx1, y1 + ly1), (x1 + lx2, y1 + ly2), COLOR_BOX_LESI, 2)

        hasil_klasifikasi = klasifikasi(potongan_ikan)[0]
        
        top5_indices = hasil_klasifikasi.probs.top5
        top_classes = [klasifikasi.names[i] for i in top5_indices]
        top_probs = [float(hasil_klasifikasi.probs.data[i]) * 100 for i in top5_indices]

        # 4. TULIS DETAIL KE PANEL INFORMASI
        cv2.putText(canvas, "[1] DETEKSI LOKASI", (img_w + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_MAIN, 1); y_text += 25
        cv2.putText(canvas, f"Status  : Ditemukan ({conf_ikan:.1f}% Akurat)", (img_w + 30, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_SUB, 1); y_text += 40

        cv2.putText(canvas, "[2] PELACAKAN LESI/LUKA", (img_w + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_MAIN, 1); y_text += 25
        status_luka = "Aman (Tidak ada lesi fisik)" if jumlah_luka == 0 else f"Terdeteksi {jumlah_luka} Area Lesi"
        cv2.putText(canvas, f"Hasil   : {status_luka}", (img_w + 30, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_SUB, 1); y_text += 40

        cv2.putText(canvas, "[3] DIAGNOSIS KONDISI (TOP 3)", (img_w + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_MAIN, 1); y_text += 25
        for i in range(3):
            bar_w = int(top_probs[i] * 2) 
            cv2.rectangle(canvas, (img_w + 30, y_text - 10), (img_w + 30 + bar_w, y_text + 4), (0, 200, 100), -1)
            cv2.putText(canvas, f"{top_classes[i]} ({top_probs[i]:.1f}%)", (img_w + 35, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0) if bar_w > 100 else COLOR_TEXT_MAIN, 1)
            y_text += 25
        y_text += 20

        cv2.putText(canvas, "[!] CATATAN SISTEM BASELINE", (img_w + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1); y_text += 20
        catatan1 = "Prediksi klasifikasi penyakit ikan berdasarkan model baseline YOLOv26"
        catatan2 = "berdasarkan fitur warna global ikan."
        catatan3 = "Karena belum memakai Dynamic Attention,"
        catatan4 = "warna alami ikan dapat memicu false positive."
        cv2.putText(canvas, catatan1, (img_w + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_SUB, 1); y_text += 20
        cv2.putText(canvas, catatan2, (img_w + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_SUB, 1); y_text += 20
        cv2.putText(canvas, catatan3, (img_w + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_SUB, 1); y_text += 20
        cv2.putText(canvas, catatan4, (img_w + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_SUB, 1)

# ==========================================
# 5. SIMPAN DAN TAMPILKAN POP-UP HASIL
# ==========================================
path_simpan = "D:/yolocomvis/hasil_dashboard_baseline.jpg"
cv2.imwrite(path_simpan, canvas)
print(f"\n📸 Dashboard visual otomatis disimpan di: {path_simpan}")

# --- BAGIAN BARU: MEMUNCULKAN POP-UP WINDOW ---
print("Tampilkan pop-up... (Tekan tombol apa saja di keyboard untuk menutup jendela)")

# Memunculkan jendela GUI dengan nama "Hasil Analisis YOLOv26"
cv2.imshow("Hasil Analisis YOLOv26", canvas)

# Menahan program agar tidak langsung keluar sampai Anda menekan tombol di keyboard
cv2.waitKey(0)

# Menutup jendela secara bersih jika tombol sudah ditekan
cv2.destroyAllWindows()
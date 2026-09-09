import cv2, os, glob, shutil, random

folder_mask = "D:/yolocomvis/datasets/fish4knowledge/maskikan/"
folder_img_train = "D:/yolocomvis/datasets/fish4knowledge/images/train/"
folder_lbl_train = "D:/yolocomvis/datasets/fish4knowledge/labels/train/"

folder_img_val = "D:/yolocomvis/datasets/fish4knowledge/images/val/"
folder_lbl_val = "D:/yolocomvis/datasets/fish4knowledge/labels/val/"

os.makedirs(folder_lbl_train, exist_ok=True)
os.makedirs(folder_img_val, exist_ok=True)
os.makedirs(folder_lbl_val, exist_ok=True)

print("1. Mengonversi Mask menjadi file .txt...")
for path_mask in glob.glob(folder_mask + "*.png"):
    # Langsung simpan dengan nama 'fish_' agar cocok dengan gambar asli
    nama_file_txt = os.path.basename(path_mask).replace("mask_", "fish_").replace(".png", ".txt")
    
    mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
    tinggi, lebar = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # Normalisasi ke format YOLO (0-1)
        x_center, y_center = (x + w/2) / lebar, (y + h/2) / tinggi
        with open(os.path.join(folder_lbl_train, nama_file_txt), 'w') as f:
            f.write(f"0 {x_center} {y_center} {w/lebar} {h/tinggi}\n")

print("2. Membagi 20% data untuk Validasi (val)...")
semua_label = [f for f in os.listdir(folder_lbl_train) if f.endswith('.txt')]
val_labels = random.sample(semua_label, int(len(semua_label) * 0.2))

for lbl_name in val_labels:
    # Pindah file .txt
    shutil.move(os.path.join(folder_lbl_train, lbl_name), os.path.join(folder_lbl_val, lbl_name))
    # Pindah file gambar
    img_name = lbl_name.replace('.txt', '.png')
    if os.path.exists(os.path.join(folder_img_train, img_name)):
        shutil.move(os.path.join(folder_img_train, img_name), os.path.join(folder_img_val, img_name))

print("Selesai! Fish4Knowledge siap ditraining.")
from __future__ import annotations

import csv
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Laporan_Baseline_YOLOComVis_revisi_YOLO26.docx"


def set_cell_shading(cell, fill):
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def set_cell_text(cell, text, bold=False, color=None, size=9):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    run = paragraph.add_run(str(text))
    run.bold = bold
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for index, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[index], header, True, (255, 255, 255), 9)
        set_cell_shading(table.rows[0].cells[index], "17365D")
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            set_cell_text(cells[index], value, size=8.5)
            if len(table.rows) % 2 == 0:
                set_cell_shading(cells[index], "EAF1F8")
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)
    doc.add_paragraph()
    return table


def add_caption(doc, text):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.italic = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)
    return paragraph


def add_image(doc, path, caption, width=6.0):
    path = Path(path)
    if not path.exists():
        return
    try:
        with Image.open(path) as image:
            ratio = image.height / image.width
        height = min(width * ratio, 7.0)
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.add_run().add_picture(str(path), width=Inches(width), height=Inches(height))
        add_caption(doc, caption)
    except (OSError, ValueError):
        pass


def add_code(doc, text):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(0.2)
    paragraph.paragraph_format.right_indent = Inches(0.2)
    paragraph.paragraph_format.space_after = Pt(6)
    for line in text.strip().splitlines():
        run = paragraph.add_run(line + "\n")
        run.font.name = "Consolas"
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(45, 45, 45)
    return paragraph


def add_bullets(doc, items):
    for item in items:
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph.add_run(item)


def add_numbered(doc, items):
    for item in items:
        paragraph = doc.add_paragraph(style="List Number")
        paragraph.add_run(item)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    run._r.append(field)


def read_csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def metric_rows(path, classification=False):
    rows = read_csv_rows(path)
    result = []
    for row in rows:
        if classification:
            result.append([row["epoch"], f"{float(row['train/loss']):.4f}", f"{float(row['val/loss']):.4f}", f"{float(row['metrics/accuracy_top1'])*100:.2f}%", f"{float(row['metrics/accuracy_top5'])*100:.2f}%"])
        else:
            result.append([row["epoch"], f"{float(row['train/box_loss']):.4f}", f"{float(row['metrics/precision(B)']):.4f}", f"{float(row['metrics/recall(B)']):.4f}", f"{float(row['metrics/mAP50(B)']):.4f}", f"{float(row['metrics/mAP50-95(B)']):.4f}"])
    return result


def style_document(doc):
    section = doc.sections[0]
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.25)
    section.right_margin = Inches(1.0)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(6)
    for name, size, color in (("Title", 20, "17365D"), ("Heading 1", 15, "17365D"), ("Heading 2", 13, "2F5597"), ("Heading 3", 12, "404040")):
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
    for section in doc.sections:
        footer = section.footer.paragraphs[0]
        add_page_number(footer)


def heading(doc, text, level=1):
    return doc.add_heading(text, level=level)


def paragraph(doc, text, bold_prefix=None):
    p = doc.add_paragraph()
    if bold_prefix and text.startswith(bold_prefix):
        p.add_run(bold_prefix).bold = True
        p.add_run(text[len(bold_prefix):])
    else:
        p.add_run(text)
    return p


def build_report():
    doc = Document()
    style_document(doc)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(85)
    run = title.add_run("LAPORAN EKSPERIMEN BASELINE\nYOLOCOMVIS")
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(21)
    run.font.color.rgb = RGBColor(23, 54, 93)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_before = Pt(25)
    run = subtitle.add_run("Deteksi Ikan, Deteksi Lesi, dan Klasifikasi Penyakit Ikan")
    run.font.name = "Times New Roman"
    run.font.size = Pt(14)
    run.italic = True
    doc.add_paragraph()
    metadata = doc.add_table(rows=4, cols=2)
    metadata.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row, values in zip(metadata.rows, [("Proyek", "YOLOComVis"), ("Eksperimen", "Baseline YOLO11n"), ("Periode artefak", "Training 10 epoch"), ("Tanggal penyusunan", "13 September 2026")]):
        set_cell_text(row.cells[0], values[0], True, size=11)
        set_cell_text(row.cells[1], values[1], size=11)
        set_cell_shading(row.cells[0], "D9EAF7")
    doc.add_page_break()

    heading(doc, "Ringkasan Eksekutif", 1)
    paragraph(doc, "Laporan ini mendokumentasikan seluruh baseline YOLOComVis dari persiapan data sampai pengujian pipeline. Sistem dibuat sebagai rangkaian tiga model: YOLO11n untuk menemukan lokasi ikan, YOLO11n untuk mendeteksi lesi atau luka pada crop ikan, dan YOLO11n-cls untuk mengklasifikasikan kondisi penyakit.")
    paragraph(doc, "Hasil aktual menunjukkan bahwa klasifikasi penyakit merupakan komponen dengan performa paling kuat pada validasi, yaitu akurasi Top-1 99,29% dan Top-5 100% pada epoch ke-10. Deteksi lesi mencapai precision 0,6733, recall 0,5646, mAP50 0,6264, dan mAP50-95 0,2968. Deteksi ikan memiliki recall 0,9996, tetapi precision 0,3592 dan mAP50 0,3618, sehingga hasilnya perlu dibaca sebagai baseline awal yang sangat sensitif namun belum presisi.")
    add_table(doc, ["Komponen", "Model", "Input", "Metrik akhir"], [
        ["Deteksi ikan", "YOLO11n", "640 x 640", "P 0,3592; R 0,9996; mAP50 0,3618"],
        ["Deteksi lesi", "YOLO11n", "640 x 640", "P 0,6733; R 0,5646; mAP50 0,6264"],
        ["Klasifikasi", "YOLO11n-cls", "224 x 224", "Top-1 99,29%; Top-5 100%"],
    ], [1.25, 1.25, 1.0, 2.7])
    paragraph(doc, "Kesimpulan praktis: baseline sudah berhasil membentuk pipeline end-to-end dan menghasilkan artefak evaluasi lengkap, tetapi penggunaan operasional harus mempertimbangkan false positive pada deteksi ikan, keterbatasan validasi, serta perbedaan antara confidence inferensi pada satu gambar dan metrik agregat seluruh validation set.")

    heading(doc, "Daftar Isi", 1)
    toc = doc.add_paragraph()
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), 'TOC \\o "1-3" \\h \\z \\u')
    toc._p.append(field)
    doc.add_page_break()

    heading(doc, "1. Latar Belakang dan Tujuan", 1)
    paragraph(doc, "YOLOComVis dirancang untuk membantu analisis citra ikan secara bertahap. Masalah visual dibagi menjadi tiga keputusan yang saling berhubungan: apakah dan di mana ikan berada, apakah terdapat lesi pada tubuh ikan, dan kondisi atau penyakit apa yang paling mungkin berdasarkan crop ikan.")
    paragraph(doc, "Pembagian tugas tersebut dipilih agar model klasifikasi tidak langsung menerima seluruh frame yang dapat berisi latar belakang, karang, air, atau objek lain. Pada tahap inferensi, detektor ikan menghasilkan bounding box, crop dari box diteruskan ke detektor lesi dan classifier, lalu seluruh informasi digabungkan menjadi dashboard.")
    heading(doc, "Tujuan eksperimen", 2)
    add_bullets(doc, ["Menyiapkan dataset dari arsip mentah menjadi struktur yang kompatibel dengan Ultralytics YOLO.", "Melatih dua model deteksi dan satu model klasifikasi dengan bobot pretrained.", "Menyimpan grafik loss, precision-recall, mAP, confusion matrix, dan contoh prediksi.", "Menghasilkan dashboard inferensi sebagai bukti pipeline dapat dijalankan dari input gambar.", "Menyediakan dokumentasi yang dapat direproduksi melalui notebook Colab dan script lokal."])

    heading(doc, "2. Arsitektur Sistem", 1)
    add_table(doc, ["Tahap", "Masukan", "Proses", "Keluaran"], [
        ["1. Deteksi ikan", "Frame atau foto", "YOLO11n detect", "Bounding box ikan + confidence"],
        ["2. Crop", "Bounding box ikan", "Pemotongan area ikan", "Citra ikan terlokalisasi"],
        ["3. Deteksi lesi", "Crop ikan", "YOLO11n detect", "Bounding box lesi/luka"],
        ["4. Klasifikasi", "Crop ikan", "YOLO11n-cls", "Top-3 kelas penyakit + probabilitas"],
        ["5. Dashboard", "Semua keluaran", "OpenCV compositing", "Visualisasi gabungan"],
    ], [1.1, 1.5, 2.0, 2.0])
    paragraph(doc, "Deteksi menggunakan format label YOLO satu kelas. Setiap baris anotasi memiliki bentuk class_id, center_x, center_y, width, height, dengan koordinat telah dinormalisasi terhadap lebar dan tinggi gambar. Klasifikasi menggunakan struktur folder kelas, sehingga nama folder menjadi label kelas.")
    add_code(doc, "fish_model = YOLO('Runs_Baseline/model_ikan/weights/best.pt')\nlesion_model = YOLO('Runs_Baseline/model_lesi/weights/best.pt')\nclassifier = YOLO('Runs_Baseline/model_penyakit/weights/best.pt')\nresult = fish_model(image)[0]\nfor box in result.boxes:\n    crop = image[y1:y2, x1:x2]\n    lesion_result = lesion_model(crop)[0]\n    classification = classifier(crop)[0]")

    heading(doc, "3. Sumber Data dan Persiapan Dataset", 1)
    paragraph(doc, "Notebook Colab menyiapkan data secara otomatis dari Fish4Knowledge, FishDisease, dan Freshwater Fish Disease Aquaculture in South Asia. Arsip dapat diambil dari URL publik atau Google Drive, diekstrak dengan pemeriksaan path aman, lalu dinormalisasi ke folder kerja runtime. Cache dipakai agar dataset tidak diunduh ulang ketika sudah tersedia.")
    add_table(doc, ["Dataset", "Tujuan", "Train", "Val", "Keterangan"], [
        ["Fish4Knowledge", "Deteksi ikan", "5.374 gambar / 9.690 objek", "6.728 gambar / 2.422 objek", "1 kelas: fish"],
        ["FishDisease", "Deteksi lesi", "929 gambar / 1.705 objek", "226 gambar / 441 objek", "1 kelas: lesion"],
        ["Freshwater Kaggle", "Klasifikasi penyakit", "1.750 gambar", "700 gambar", "7 kelas, masing-masing seimbang"],
    ], [1.35, 1.3, 1.45, 1.45, 1.55])
    heading(doc, "3.1 Konversi Fish4Knowledge", 2)
    paragraph(doc, "Fungsi prepare_fish_dataset mengosongkan folder split keluaran, membangun indeks gambar berdasarkan stem nama file, membaca mask grayscale, mencari kontur eksternal terbesar, lalu mengubah bounding rectangle kontur menjadi satu baris label YOLO kelas 0. Pasangan gambar-label diacak dengan seed 42; 20% pasangan dipindahkan ke validation.")
    add_code(doc, "contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\nx, y, box_width, box_height = cv2.boundingRect(max(contours, key=cv2.contourArea))\nlabel = f\"0 {(x + box_width / 2) / width:.6f} ...\"\nrandom.Random(seed).shuffle(pairs)")
    heading(doc, "3.2 Konversi FishDisease", 2)
    paragraph(doc, "Fungsi prepare_lesion_dataset membaca seluruh JSON anotasi, mengambil array annotations, mencari gambar berdasarkan field image atau folder halaman, memuat ukuran gambar, melakukan clipping bounding box ke batas citra, menolak box kosong, kemudian menulis koordinat ter-normalisasi ke label YOLO. Nama keluaran diberi prefix folder halaman agar tidak terjadi benturan nama file.")
    add_code(doc, "x_min, y_min, x_max, y_max = map(float, bbox)\nx_min, x_max = sorted((max(0.0, x_min), min(float(width), x_max)))\ny_min, y_max = sorted((max(0.0, y_min), min(float(height), y_max)))\nyolo_lines.append(f\"0 {((x_min + x_max) / 2) / width:.6f} ...\")")
    heading(doc, "3.3 Konversi Dataset Klasifikasi", 2)
    paragraph(doc, "Fungsi prepare_classification_dataset memakai splitfolders.ratio dengan rasio 80:20 dan seed 42. Setiap subfolder penyakit dipertahankan sebagai nama kelas. Pada artefak lokal, terdapat tujuh kelas dengan 250 citra train dan 100 citra validation per kelas.")
    heading(doc, "3.4 Validasi Struktur", 2)
    paragraph(doc, "Fungsi dataset_summary menghitung jumlah gambar, label, objek, serta pasangan yang tidak cocok. Ringkasan yang sama dapat diekspor ke JSON, CSV, dan Markdown melalui save_dataset_report. Pemeriksaan ini penting karena gambar tanpa label atau label tanpa gambar akan menghasilkan data training yang tidak sesuai harapan.")

    heading(doc, "4. Implementasi Kode", 1)
    paragraph(doc, "Implementasi utama berada pada all_in_one_yolocomvis.py. Script tersebut memusatkan persiapan data, pembuatan YAML, training, evaluasi, pembuatan plot, dashboard, dan ekspor artefak. Script training sederhana pada scripts/3_training_all.py menunjukkan tiga pemanggilan model.train yang digunakan pada baseline awal.")
    heading(doc, "4.1 Fungsi utilitas", 2)
    add_bullets(doc, ["_image_files: mengembalikan file gambar dengan ekstensi yang didukung.", "dataset_summary: menghitung statistik split dan kecocokan gambar-label.", "save_dataset_report: menulis dataset_summary.json, dataset_summary.csv, dan dataset_summary.md.", "write_dataset_yaml: membuat YAML portable untuk deteksi ikan dan lesi.", "_metric_value: mengambil metrik dari object hasil Ultralytics dengan beberapa nama atribut alternatif."])
    heading(doc, "4.2 Training", 2)
    paragraph(doc, "train_models membuat tiga model pretrained. Dua detector memakai yolo11n.pt, imgsz 640, dan 10 epoch. Classifier memakai yolo11n-cls.pt, imgsz 224, dan 10 epoch. Semua hasil diarahkan ke project Runs_Baseline dengan nama model_ikan, model_lesi, dan model_penyakit.")
    add_code(doc, "YOLO('yolo11n.pt').train(data=fish_yaml, epochs=10, imgsz=640, project='Runs_Baseline', name='model_ikan')\nYOLO('yolo11n.pt').train(data=lesion_yaml, epochs=10, imgsz=640, project='Runs_Baseline', name='model_lesi')\nYOLO('yolo11n-cls.pt').train(data=classification_dir, epochs=10, imgsz=224, project='Runs_Baseline', name='model_penyakit')")
    heading(doc, "4.3 Evaluasi dan ekspor metrik", 2)
    paragraph(doc, "evaluate_models memuat best.pt, menjalankan model.val, dan menyimpan precision, recall, mAP50, mAP50-95 untuk deteksi atau Top-1 dan Top-5 untuk klasifikasi. plot_training_curves membaca results.csv dan membangun grafik gabungan loss serta metrik. classification_confusion_matrix melakukan prediksi satu per satu pada folder validation lalu mengisi matriks aktual versus prediksi.")
    heading(doc, "4.4 Dashboard", 2)
    paragraph(doc, "run_dashboard membaca satu gambar, menjalankan detektor ikan, memotong setiap ikan, menjalankan detector lesi pada crop, lalu menjalankan classifier. Bounding box ikan diberi warna kuning, lesi merah, sedangkan panel kanan berisi confidence ikan, jumlah lesi, dan tiga prediksi penyakit teratas.")

    heading(doc, "4.5 Penjelasan Script Tahap demi Tahap", 2)
    paragraph(doc, "Selain pipeline terpadu, proyek menyediakan empat script eksplisit yang memperlihatkan proses eksperimen secara berurutan. Bagian ini menjelaskan kode asli pada folder scripts, bukan hanya versi fungsi yang sudah dirapikan di all_in_one_yolocomvis.py.")

    heading(doc, "4.5.1 scripts/1_proses_fish4knowledge.py", 3)
    paragraph(doc, "Script ini menyiapkan dataset deteksi ikan dari mask Fish4Knowledge. Mask adalah citra grayscale yang menunjukkan area ikan. Script mengubah area tersebut menjadi bounding box dan label YOLO, kemudian memindahkan 20% pasangan gambar-label ke folder validation.")
    paragraph(doc, "Bagian import memanggil OpenCV untuk membaca mask, os dan glob untuk mengelola path serta mencari file, shutil untuk memindahkan file, dan random untuk mengambil sampel validation. Enam variabel folder memisahkan sumber mask, gambar train, label train, gambar val, dan label val. os.makedirs(..., exist_ok=True) memastikan folder tujuan tersedia sebelum proses dimulai.")
    add_code(doc, "mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)\ntinggi, lebar = mask.shape\ncontours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\nc = max(contours, key=cv2.contourArea)\nx, y, w, h = cv2.boundingRect(c)")
    paragraph(doc, "Loop pertama membaca setiap PNG pada folder mask. cv2.IMREAD_GRAYSCALE menghasilkan mask satu kanal. cv2.findContours mencari batas objek; RETR_EXTERNAL hanya mengambil kontur luar, sehingga bagian internal tidak dianggap sebagai objek terpisah. max(..., key=cv2.contourArea) memilih kontur terbesar, lalu cv2.boundingRect mengubah kontur menjadi koordinat x, y, lebar, dan tinggi bounding box.")
    add_code(doc, "x_center, y_center = (x + w/2) / lebar, (y + h/2) / tinggi\nf.write(f\"0 {x_center} {y_center} {w/lebar} {h/tinggi}\\n\")")
    paragraph(doc, "Empat nilai koordinat dibagi ukuran gambar agar berubah dari piksel menjadi rentang 0 sampai 1 sesuai format YOLO: titik tengah x, titik tengah y, lebar box, dan tinggi box. Angka 0 di awal baris adalah class_id karena dataset ini hanya memiliki satu kelas, yaitu fish. Nama mask_ diubah menjadi fish_ agar nama label dapat dipasangkan dengan gambar.")
    add_code(doc, "semua_label = [f for f in os.listdir(folder_lbl_train) if f.endswith('.txt')]\nval_labels = random.sample(semua_label, int(len(semua_label) * 0.2))\nshutil.move(label_train, label_val)\nshutil.move(image_train, image_val)")
    paragraph(doc, "Bagian kedua mengambil sampel acak 20% dari label train. Setiap label dipindahkan ke labels/val dan gambar dengan stem yang sama dipindahkan ke images/val. Dengan demikian, label dan gambar tetap berpasangan. Script ini belum menetapkan random seed, sehingga pembagian dapat berbeda setiap kali dijalankan; pipeline terpadu memperbaikinya dengan seed 42.")
    paragraph(doc, "Output script adalah file .txt berformat YOLO, gambar pada images/train dan images/val, serta ringkasan terminal bahwa Fish4Knowledge siap digunakan untuk training. Risiko utama versi script ini adalah asumsi ekstensi gambar .png dan kecocokan langsung nama fish_*.txt dengan fish_*.png.")

    heading(doc, "4.5.2 scripts/2_proses_fishdisease.py", 3)
    paragraph(doc, "Script ini mengubah anotasi mentah FishDisease berbentuk JSON menjadi dataset deteksi lesi berformat YOLO. Setiap JSON dapat memiliki beberapa anotasi bbox, sehingga satu file label dapat berisi beberapa baris objek lesi.")
    paragraph(doc, "Pada awal script, dir_mentah menunjuk folder sumber, sedangkan empat variabel tujuan memisahkan images/train, images/val, labels/train, dan labels/val. glob(..., recursive=True) memindai JSON pada seluruh subfolder page. Counter sukses, skip_kosong, skip_gambar, dan skip_tanpa_bbox dipakai untuk merangkum kualitas proses.")
    add_code(doc, "data = json.load(f)\nif \"annotations\" not in data or not data[\"annotations\"]:\n    skip_kosong += 1\n    continue\npath_gambar_asli = os.path.join(dir_mentah, data[\"image\"])")
    paragraph(doc, "Setiap JSON dibaca menggunakan json.load. JSON tanpa annotations dilewati karena tidak memiliki target training. Field image dicoba terlebih dahulu untuk menemukan gambar asli. Jika path tersebut tidak cocok dengan struktur arsip, script menggunakan fallback: mengambil folder page dari lokasi JSON lalu mencari file JPG atau PNG pertama pada folder tersebut.")
    add_code(doc, "tinggi_img, lebar_img, _ = img.shape\nfor ann in data[\"annotations\"]:\n    x_min, y_min, x_max, y_max = ann[\"bbox\"]\n    x_center = ((x_min + x_max) / 2.0) / lebar_img\n    y_center = ((y_min + y_max) / 2.0) / tinggi_img\n    w = (x_max - x_min) / lebar_img\n    h = (y_max - y_min) / tinggi_img\n    baris_yolo.append(f\"0 {x_center} {y_center} {w} {h}\")")
    paragraph(doc, "Ukuran gambar diambil dari img.shape. Setiap bbox mentah diasumsikan berbentuk x_min, y_min, x_max, y_max, lalu diubah ke format YOLO center-x, center-y, width, height dan dinormalisasi terhadap lebar serta tinggi gambar. Class_id selalu 0 karena semua anotasi diperlakukan sebagai kelas lesion. Berbeda dari script pertama, satu gambar dapat menghasilkan banyak baris label.")
    add_code(doc, "nama_unik = f\"{nama_page}_{os.path.splitext(nama_file_asli)[0]}\"\nis_val = random.random() < 0.2\nshutil.copy(path_gambar_asli, os.path.join(tujuan_img, ...))\nwith open(os.path.join(tujuan_lbl, f\"{nama_unik}.txt\"), 'w') as f:\n    f.write(\"\\n\".join(baris_yolo))")
    paragraph(doc, "Nama unik menggabungkan nama page dan nama gambar untuk mencegah tabrakan file seperti page10_img1 dan page11_img1. random.random() < 0.2 memilih validation dengan peluang 20%. Gambar disalin, sedangkan label baru ditulis pada folder yang sesuai. try-except menjaga agar satu JSON yang rusak tidak menghentikan pemrosesan seluruh dataset.")
    paragraph(doc, "Output script adalah pasangan gambar-label FishDisease, serta jumlah data berhasil dan jumlah data yang dilewati. Pipeline terpadu menambahkan clipping bbox ke batas citra dan penggunaan Random(seed) agar pembagian lebih reproducible.")

    heading(doc, "4.5.3 scripts/3_training_all.py", 3)
    paragraph(doc, "Script ini adalah entry point training baseline. Tidak ada preprocessing di dalamnya; script mengasumsikan data sudah disiapkan dan YAML atau folder klasifikasi sudah tersedia. Tiga blok training dijalankan berurutan.")
    add_code(doc, "from ultralytics import YOLO\nmodel_ikan = YOLO(\"yolo11n.pt\")\nmodel_ikan.train(data=\"D:/yolocomvis/data_ikan.yaml\", epochs=10, imgsz=640, project=\"Runs_Baseline\", name=\"model_ikan\")")
    paragraph(doc, "YOLO(\"yolo11n.pt\") memuat bobot pretrained YOLO11n untuk task deteksi. Pemanggilan train menerima YAML yang mendefinisikan path dataset, melatih selama 10 epoch dengan ukuran input 640 x 640, dan menyimpan hasil ke Runs_Baseline/model_ikan. Ultralytics otomatis menyimpan best.pt, last.pt, results.csv, args.yaml, grafik, confusion matrix, dan contoh batch.")
    add_code(doc, "model_lesi = YOLO(\"yolo11n.pt\")\nmodel_lesi.train(data=\"D:/yolocomvis/data_lesi.yaml\", epochs=10, imgsz=640, project=\"Runs_Baseline\", name=\"model_lesi\")\nmodel_penyakit = YOLO(\"yolo11n-cls.pt\")\nmodel_penyakit.train(data=\"D:/yolocomvis/datasets/freshwater_kaggle\", epochs=10, imgsz=224, project=\"Runs_Baseline\", name=\"model_penyakit\")")
    paragraph(doc, "Blok kedua menggunakan arsitektur deteksi yang sama untuk lesi, tetapi membaca data_lesi.yaml dan menyimpan ke model_lesi. Blok ketiga menggunakan yolo11n-cls.pt; akhiran -cls penting karena task-nya klasifikasi, bukan bounding-box detection. Parameter data langsung menunjuk folder freshwater_kaggle karena Ultralytics membaca subfolder kelas train dan val. Ukuran input classifier 224 x 224 berbeda dari detector karena mengikuti konfigurasi klasifikasi.")
    paragraph(doc, "Output script berupa tiga folder hasil eksperimen dan checkpoint model. results.csv menjadi sumber tabel metrik per epoch pada laporan ini. Perlu diperhatikan bahwa path pada script bersifat absolut Windows, sehingga perlu disesuaikan ketika dijalankan di komputer atau Colab lain. Pada artefak model_lesi, args.yaml menunjukkan resume dari last.pt; fakta tersebut dicatat sebagai bagian dari riwayat eksperimen.")

    heading(doc, "4.5.4 scripts/4_tes_pipeline_final.py", 3)
    paragraph(doc, "Script ini menguji pipeline end-to-end menggunakan tiga best checkpoint. Berbeda dari training, script ini tidak mengubah bobot; tugasnya adalah membaca satu gambar, menemukan ikan, menjalankan analisis lesi dan klasifikasi pada setiap crop ikan, lalu menyusun dashboard visual.")
    add_code(doc, "deteksi_ikan = YOLO(path_ikan)\ndeteksi_lesi = YOLO(path_lesi)\nklasifikasi = YOLO(path_penyakit)\ngambar = cv2.imread(path_gambar_tes)\nimg_h, img_w = gambar.shape[:2]\ncanvas = np.zeros((canvas_h, img_w + panel_w, 3), dtype=np.uint8)\ncanvas[0:img_h, 0:img_w] = gambar")
    paragraph(doc, "Tiga model dimuat dari best.pt. cv2.imread membaca gambar uji; apabila hasilnya None, script menghentikan proses dengan pesan error. Canvas baru dibuat lebih lebar dari gambar asli agar memiliki panel informasi di sisi kanan. Warna panel dan box ditetapkan dalam format BGR OpenCV: ikan cyan/kuning, lesi merah, dan teks putih atau abu-abu.")
    add_code(doc, "hasil_ikan = deteksi_ikan(gambar)[0]\nfor kotak in hasil_ikan.boxes:\n    x1, y1, x2, y2 = map(int, kotak.xyxy[0])\n    conf_ikan = float(kotak.conf[0]) * 100\n    potongan_ikan = gambar[y1:y2, x1:x2]\n    hasil_lesi = deteksi_lesi(potongan_ikan)[0]\n    hasil_klasifikasi = klasifikasi(potongan_ikan)[0]")
    paragraph(doc, "Prediksi deteksi ikan diambil dari hasil_ikan.boxes. Koordinat xyxy diubah menjadi integer untuk menggambar rectangle dan melakukan slicing array. Confidence dikalikan 100 agar tampil sebagai persen. Crop ikan diproses ulang oleh detector lesi dan classifier, sehingga lesi dan penyakit dinilai pada area ikan, bukan seluruh frame.")
    add_code(doc, "for luka in hasil_lesi.boxes:\n    lx1, ly1, lx2, ly2 = map(int, luka.xyxy[0])\n    cv2.rectangle(canvas, (x1 + lx1, y1 + ly1), (x1 + lx2, y1 + ly2), COLOR_BOX_LESI, 2)\ntop5_indices = hasil_klasifikasi.probs.top5\ntop_probs = [float(hasil_klasifikasi.probs.data[i]) * 100 for i in top5_indices]")
    paragraph(doc, "Koordinat lesi awalnya relatif terhadap crop. Penambahan x1 dan y1 mengembalikannya ke koordinat canvas global. Classifier menyediakan indeks Top-5 dan probabilitas; dashboard hanya menampilkan tiga teratas melalui range(3), lengkap dengan progress bar yang panjangnya sebanding dengan probabilitas.")
    add_code(doc, "cv2.imwrite(path_simpan, canvas)\ncv2.imshow(\"Hasil Analisis YOLO26\", canvas)\ncv2.waitKey(0)\ncv2.destroyAllWindows()")
    paragraph(doc, "Canvas disimpan sebagai hasil_dashboard_baseline.jpg, kemudian ditampilkan pada jendela OpenCV. cv2.waitKey(0) menahan program sampai tombol ditekan dan cv2.destroyAllWindows menutup jendela dengan bersih. Karena script ini menggunakan GUI, cv2.imshow dapat gagal pada runtime headless seperti Colab; untuk lingkungan tersebut, bagian simpan dengan cv2.imwrite tetap dapat dipakai tanpa pop-up.")
    paragraph(doc, "Alur uji pipeline dapat diringkas sebagai: gambar penuh -> deteksi ikan -> crop tiap ikan -> deteksi lesi dan klasifikasi -> gambar box serta panel informasi -> simpan dashboard. Catatan pada panel mengenai false positive warna merupakan interpretasi keterbatasan baseline, bukan hasil metrik tambahan.")

    heading(doc, "5. Konfigurasi Eksperimen", 1)
    add_table(doc, ["Parameter", "Deteksi ikan", "Deteksi lesi", "Klasifikasi"], [
        ["Model", "yolo11n.pt", "yolo11n.pt", "yolo11n-cls.pt"],
        ["Epoch", "10", "10", "10"],
        ["Image size", "640", "640", "224"],
        ["Batch", "16", "16", "16"],
        ["Optimizer", "auto", "auto", "auto"],
        ["Learning rate awal", "0,01", "0,01", "0,01"],
        ["Weight decay", "0,0005", "0,0005", "0,0005"],
        ["Augmentasi utama", "mosaic, flip LR, randaugment", "mosaic, flip LR, randaugment", "flip LR, randaugment, erasing"],
    ], [1.65, 1.55, 1.55, 1.55])
    paragraph(doc, "args.yaml aktual menunjukkan seed Ultralytics bernilai 0 dan deterministic true, sedangkan pembagian dataset menggunakan seed 42 pada pipeline persiapan. Untuk model_lesi, args.yaml menyatakan resume dari last.pt; hal ini perlu dicatat ketika membandingkan percobaan atau mengulang training secara eksak.")

    heading(doc, "6. Hasil Training dan Evaluasi", 1)
    paragraph(doc, "Catatan versi: angka pada bab ini adalah hasil artefak Tugas 2 sebelum revisi, yaitu training YOLO11. Angka ini dipertahankan agar dokumen tetap sama dengan laporan asal dan tidak boleh dianggap sebagai hasil YOLO26.")
    heading(doc, "6.1 Deteksi ikan", 2)
    paragraph(doc, "Loss box turun dari 0,8066 menjadi 0,4723 dan loss klasifikasi turun dari 1,3337 menjadi 0,2206. Recall validation sangat tinggi sejak awal dan mencapai 0,9996 pada epoch ke-10, tetapi precision berada sekitar 0,36. mAP50 terbaik pada epoch ke-3 sebesar 0,3733, sedangkan nilai akhir mAP50-95 sebesar 0,3379. Ini menandakan model cenderung menangkap objek ikan, namun kualitas ketepatan box dan seleksi prediksi masih terbatas.")
    add_table(doc, ["Epoch", "Box loss", "Precision", "Recall", "mAP50", "mAP50-95"], metric_rows(ROOT / "runs/detect/Runs_Baseline/model_ikan/results.csv"), [0.55, 1.0, 1.0, 0.9, 0.95, 1.1])
    add_image(doc, ROOT / "runs/detect/Runs_Baseline/model_ikan/results.png", "Gambar 1. Kurva training dan metrik validasi deteksi ikan.")
    add_image(doc, ROOT / "runs/detect/Runs_Baseline/model_ikan/confusion_matrix_normalized.png", "Gambar 2. Confusion matrix ternormalisasi deteksi ikan.", 5.7)
    add_image(doc, ROOT / "runs/detect/Runs_Baseline/model_ikan/val_batch0_pred.jpg", "Gambar 3. Contoh prediksi deteksi ikan pada validation batch.", 5.8)
    heading(doc, "6.2 Deteksi lesi", 2)
    paragraph(doc, "Model lesi mengalami peningkatan bertahap: precision dari 0,0036 menjadi 0,6733, mAP50 dari 0,0201 menjadi 0,6264, dan mAP50-95 dari 0,0076 menjadi 0,2968. Recall akhir 0,5646 menunjukkan sebagian lesi masih terlewat. Jarak antara mAP50 dan mAP50-95 memperlihatkan bahwa box sering cukup benar pada threshold IoU 0,50 tetapi belum konsisten ketika syarat IoU diperketat.")
    add_table(doc, ["Epoch", "Box loss", "Precision", "Recall", "mAP50", "mAP50-95"], metric_rows(ROOT / "runs/detect/Runs_Baseline/model_lesi/results.csv"), [0.55, 1.0, 1.0, 0.9, 0.95, 1.1])
    add_image(doc, ROOT / "runs/detect/Runs_Baseline/model_lesi/results.png", "Gambar 4. Kurva training dan metrik validasi deteksi lesi.")
    add_image(doc, ROOT / "runs/detect/Runs_Baseline/model_lesi/BoxPR_curve.png", "Gambar 5. Kurva precision-recall deteksi lesi.", 5.8)
    add_image(doc, ROOT / "runs/detect/Runs_Baseline/model_lesi/val_batch0_pred.jpg", "Gambar 6. Contoh prediksi lesi pada validation batch.", 5.8)
    heading(doc, "6.3 Klasifikasi penyakit", 2)
    paragraph(doc, "Loss train turun dari 1,6408 menjadi 0,1996 dan loss validation turun dari 0,8945 menjadi 0,0285. Top-1 meningkat dari 71,00% menjadi 99,29%, sedangkan Top-5 mencapai 100% mulai epoch ke-4. Confusion matrix menunjukkan nilai diagonal sekitar 0,98 sampai 1,00, dengan kesalahan kecil antar beberapa kelas penyakit bakteri dan penyakit lainnya.")
    add_table(doc, ["Epoch", "Train loss", "Val loss", "Top-1", "Top-5"], metric_rows(ROOT / "runs/classify/Runs_Baseline/model_penyakit/results.csv", True), [0.6, 1.2, 1.2, 1.0, 1.0])
    add_image(doc, ROOT / "runs/classify/Runs_Baseline/model_penyakit/results.png", "Gambar 7. Kurva loss dan akurasi klasifikasi penyakit.")
    add_image(doc, ROOT / "runs/classify/Runs_Baseline/model_penyakit/confusion_matrix_normalized.png", "Gambar 8. Confusion matrix ternormalisasi klasifikasi penyakit.", 5.7)
    add_image(doc, ROOT / "runs/classify/Runs_Baseline/model_penyakit/val_batch0_pred.jpg", "Gambar 9. Contoh prediksi kelas penyakit pada validation batch.", 5.8)

    heading(doc, "7. Dokumentasi Inferensi dan Uji Coba", 1)
    paragraph(doc, "Selain grafik training, repository menyimpan sebelas gambar ujicoba dan satu dashboard baseline. Gambar-gambar tersebut berfungsi sebagai dokumentasi kualitatif: memperlihatkan citra masukan, bounding box ikan, hasil deteksi lesi, serta ranking kelas penyakit. Evaluasi kualitatif tidak menggantikan metrik validation, tetapi membantu menemukan perilaku yang mudah terlewat pada angka agregat.")
    add_image(doc, ROOT / "hasil_dashboard_baseline.jpg", "Gambar 10. Dashboard pipeline baseline dengan deteksi ikan, jumlah lesi, dan Top-3 klasifikasi.", 6.2)
    add_image(doc, ROOT / "ujicoba.png", "Gambar 11. Contoh citra masukan uji coba.", 4.8)
    add_image(doc, ROOT / "ujicoba2.png", "Gambar 12. Contoh dokumentasi uji coba tambahan.", 4.8)
    add_image(doc, ROOT / "ujicoba3.png", "Gambar 13. Contoh dokumentasi uji coba tambahan.", 4.8)
    add_image(doc, ROOT / "ujicoba4.png", "Gambar 14. Contoh dokumentasi uji coba tambahan.", 4.8)
    add_image(doc, ROOT / "ujicoba5.png", "Gambar 15. Contoh dokumentasi uji coba tambahan.", 4.8)
    add_image(doc, ROOT / "ujicoba6.png", "Gambar 16. Contoh dokumentasi uji coba tambahan.", 4.8)
    add_image(doc, ROOT / "ujicoba7.png", "Gambar 17. Contoh dokumentasi uji coba tambahan.", 4.8)
    add_image(doc, ROOT / "ujicoba8.png", "Gambar 18. Contoh dokumentasi uji coba tambahan.", 4.8)
    add_image(doc, ROOT / "ujicoba9.png", "Gambar 19. Contoh dokumentasi uji coba tambahan.", 4.8)
    add_image(doc, ROOT / "ujicoba10.png", "Gambar 20. Contoh dokumentasi uji coba tambahan.", 4.8)
    add_image(doc, ROOT / "ujicoba11.jpg", "Gambar 21. Contoh dokumentasi uji coba tambahan.", 4.8)
    paragraph(doc, "Pada dashboard, kotak kuning merepresentasikan hasil detektor ikan. Crop ikan kemudian dianalisis oleh dua model berikutnya. Panel kanan menampilkan confidence deteksi, jumlah lesi yang ditemukan, dan tiga kelas penyakit teratas. Confidence satu gambar tidak boleh langsung disamakan dengan Top-1 validation karena keduanya dihitung pada konteks data dan prosedur yang berbeda.")

    heading(doc, "8. Analisis Kritis dan Keterbatasan", 1)
    add_bullets(doc, ["Deteksi ikan sangat sensitif tetapi precision rendah. Banyak prediksi positif dapat muncul pada latar atau objek yang mirip ikan.", "Deteksi lesi memiliki mAP50 lebih baik daripada deteksi ikan, namun recall 0,5646 berarti lesi kecil, samar, atau tidak representatif masih mungkin terlewat.", "Akurasi klasifikasi sangat tinggi pada validation set yang seimbang, tetapi belum membuktikan generalisasi ke kamera, pencahayaan, spesies, atau kondisi air yang berbeda.", "Dataset deteksi ikan memiliki jumlah train/val yang terlihat tidak sama dengan split 80:20 sederhana karena statistik berasal dari artefak final dan proses konversi mask menghasilkan pasangan tertentu.", "Notebook Colab dan script lokal menyimpan artefak pada lokasi berbeda. Jalur absolute pada YAML lokal perlu diubah menjadi jalur portable ketika dipindahkan ke mesin lain.", "Folder Google Drive yang diberikan belum dapat diambil otomatis dari sesi ini karena tautan folder memerlukan akses autentikasi. Laporan ini menggunakan semua artefak lokal yang tersedia; artefak Drive dapat ditambahkan pada revisi berikutnya tanpa mengubah struktur laporan."])
    heading(doc, "Rekomendasi", 2)
    add_numbered(doc, ["Lakukan inspeksi false positive detektor ikan dan tambahkan hard negative dari latar air/karang.", "Perbesar dan bersihkan anotasi lesi kecil; evaluasi dengan threshold confidence dan IoU yang dilaporkan eksplisit.", "Gunakan split berbasis video atau sumber individu agar tidak terjadi kebocoran visual antar train dan validation.", "Tambahkan test set eksternal dan laporkan macro-F1, balanced accuracy, serta confidence calibration untuk klasifikasi.", "Bandingkan training 10 epoch dengan epoch lebih panjang dan early stopping yang dipantau pada validation.", "Simpan environment, versi Ultralytics, GPU, dan hash dataset sebagai metadata reproduksibilitas."])

    heading(doc, "9. Kesimpulan", 1)
    paragraph(doc, "Baseline YOLOComVis telah berhasil dibangun sebagai pipeline computer vision bertingkat. Seluruh tahapan utama tersedia: pengunduhan dan normalisasi dataset pada notebook, konversi anotasi ke format YOLO, pembagian train-validation, training tiga model pretrained, penyimpanan best dan last checkpoint, evaluasi numerik, pembuatan grafik dan confusion matrix, serta inferensi terintegrasi berbentuk dashboard.")
    paragraph(doc, "Secara kinerja, classifier menjadi komponen paling matang pada eksperimen ini. Detector lesi sudah menunjukkan sinyal belajar yang jelas dan mAP50 yang cukup berarti untuk baseline. Detector ikan berhasil mempertahankan recall hampir sempurna, tetapi precision dan mAP masih menjadi fokus perbaikan utama. Karena itu, baseline layak dipakai sebagai titik pembanding eksperimen berikutnya, bukan sebagai sistem diagnosis otomatis tanpa verifikasi manusia.")

    heading(doc, "10. Revisi Baseline Sesuai Dasar Teori", 1)
    paragraph(doc, "Bab ini mendokumentasikan revisi kode setelah pencocokan dengan laporan dasar teori YOLO26 + Dynamic Attention. Revisi menerapkan baseline YOLO26 resmi tanpa modul tambahan Dynamic Attention. Fitur native YOLO26 mengikuti checkpoint dan implementasi resmi Ultralytics, sedangkan DAM, scale attention, channel attention, spatial attention tambahan, dan eksperimen pembanding dengan DAM sengaja belum digunakan.")
    heading(doc, "10.1 Arsitektur baseline revisi", 2)
    add_table(doc, ["Komponen", "Model revisi", "Peran"], [
        ["Deteksi ikan", "YOLO26n", "Mendeteksi bounding box ikan pada frame"],
        ["Deteksi lesi", "YOLO26n", "Mendeteksi bounding box lesi pada crop ikan"],
        ["Klasifikasi", "YOLO26n-cls", "Mengklasifikasikan kondisi penyakit pada crop ikan"],
        ["Dynamic Attention", "Tidak digunakan", "Disiapkan sebagai eksperimen lanjutan, bukan bagian baseline"],
    ], [1.55, 1.55, 3.0])
    paragraph(doc, "Dengan susunan ini, baseline telah sesuai dengan bagian teori yang menyatakan YOLO26 sebagai model dasar. Komponen native YOLO26, seperti arsitektur head dan loss yang disediakan checkpoint resmi, tidak diganti secara manual; yang belum ada hanyalah DAM sebagai modul usulan. Alur inferensi tetap berupa gambar penuh, deteksi ikan, crop, deteksi lesi, klasifikasi, dan dashboard.")
    heading(doc, "10.2 Perubahan kode", 2)
    add_bullets(doc, [
        "scripts/3_training_all.py menggunakan yolo26n.pt untuk dua detector dan yolo26n-cls.pt untuk classifier.",
        "all_in_one_yolocomvis.py menggunakan checkpoint YOLO26 yang sama pada mode train, evaluate, dan dashboard.",
        "Training menetapkan seed 42, deterministic=True, dan resume=False agar baseline tidak melanjutkan checkpoint lama secara diam-diam.",
        "Path evaluasi dan dashboard diselaraskan dengan struktur Ultralytics: runs/detect/Runs_Baseline dan runs/classify/Runs_Baseline.",
        "scripts/4_tes_pipeline_final.py menggunakan label YOLO26 dan menyatakan secara eksplisit bahwa baseline belum memakai Dynamic Attention.",
    ])
    heading(doc, "10.3 Status hasil revisi", 2)
    paragraph(doc, "Angka metrik pada Bab 6 berasal dari artefak training sebelumnya yang menggunakan YOLO11, sehingga tidak boleh dibaca sebagai hasil YOLO26. Kode revisi sudah siap menjalankan training baseline YOLO26, tetapi hasil baru hanya dapat dilaporkan setelah scripts/3_training_all.py atau all_in_one_yolocomvis.py dijalankan ulang dengan dataset dan konfigurasi revisi.")
    add_code(doc, "python scripts/3_training_all.py\npython all_in_one_yolocomvis.py --mode evaluate\npython all_in_one_yolocomvis.py --mode dashboard --image ujicoba.png")
    paragraph(doc, "Setelah training ulang, hasil evaluasi YOLO26 perlu menggantikan tabel lama dan diberi metadata versi Ultralytics, perangkat, seed, serta checkpoint yang digunakan. Perbandingan dengan Dynamic Attention merupakan pekerjaan lanjutan dan tidak termasuk baseline revisi ini.")

    heading(doc, "Lampiran A. Struktur Artefak", 1)
    add_table(doc, ["Artefak", "Fungsi"], [
        ["all_in_one_yolocomvis.py", "Pipeline persiapan, training, evaluasi, dashboard, dan ekspor."],
        ["scripts/3_training_all.py", "Versi ringkas tiga pemanggilan training baseline."],
        ["data_ikan.yaml / data_lesi.yaml", "Konfigurasi path dan nama kelas deteksi."],
        ["runs/detect/Runs_Baseline/model_ikan", "Checkpoint, results.csv, args.yaml, grafik, dan batch deteksi ikan."],
        ["runs/detect/Runs_Baseline/model_lesi", "Checkpoint, results.csv, args.yaml, grafik, dan batch deteksi lesi."],
        ["runs/classify/Runs_Baseline/model_penyakit", "Checkpoint, results.csv, args.yaml, grafik, dan batch klasifikasi."],
        ["hasil_dashboard_baseline.jpg", "Bukti visual pipeline terintegrasi."],
        ["ujicoba*.png/jpg", "Dokumentasi input dan hasil pengujian kualitatif."],
    ], [2.8, 3.8])
    heading(doc, "Lampiran B. Sumber dan Reproduksi", 1)
    paragraph(doc, "Notebook notebooks/yolocomvis_colab.ipynb memuat alur Colab mulai dari instalasi library, konfigurasi runtime, download arsip, ekstraksi, pemeriksaan layout, persiapan dataset, training, evaluasi, dashboard, dan ekspor ke Google Drive. Untuk reproduksi lokal, jalankan script dengan mode prepare, train, evaluate, atau dashboard sesuai urutan. Bobot pretrained YOLO26 diunduh otomatis oleh Ultralytics ketika belum tersedia.")
    add_code(doc, "python all_in_one_yolocomvis.py --mode prepare --seed 42\npython all_in_one_yolocomvis.py --mode train --epochs 10\npython all_in_one_yolocomvis.py --mode evaluate\npython all_in_one_yolocomvis.py --mode dashboard --image ujicoba.png")
    paragraph(doc, "Dokumen ini disusun dari source code, notebook, konfigurasi YAML, results.csv, args.yaml, checkpoint directory, dan gambar yang ada di workspace d:\\yolocomvis pada saat pembuatan laporan.")

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_report()
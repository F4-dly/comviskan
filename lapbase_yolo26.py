from __future__ import annotations

import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Laporan_Baseline_YOLOComVis_YOLO26_revisi_dengan_lampiran_kode.docx"


def paragraph(doc, text, bold_prefix=None):
    item = doc.add_paragraph()
    if bold_prefix and text.startswith(bold_prefix):
        item.add_run(bold_prefix).bold = True
        item.add_run(text[len(bold_prefix):])
    else:
        item.add_run(text)
    return item


def heading(doc, text, level=1):
    return doc.add_heading(text, level=level)


def table(doc, headers, rows):
    result = doc.add_table(rows=1, cols=len(headers))
    result.style = "Table Grid"
    result.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, value in zip(result.rows[0].cells, headers):
        cell.text = str(value)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for row in rows:
        cells = result.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
    doc.add_paragraph()
    return result


def bullets(doc, values):
    for value in values:
        doc.add_paragraph(value, style="List Bullet")


def code(doc, value):
    item = doc.add_paragraph()
    item.paragraph_format.left_indent = Inches(0.2)
    run = item.add_run(value)
    run.font.name = "Consolas"
    run.font.size = Pt(8)


def code_file(doc, path, max_lines=None):
    text = path.read_text(encoding="utf-8")
    if max_lines is not None:
        lines = text.splitlines()
        text = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            text += "\n... [bagian berikutnya dijelaskan pada lampiran berikut]"
    item = doc.add_paragraph()
    item.paragraph_format.left_indent = Inches(0.12)
    item.paragraph_format.right_indent = Inches(0.12)
    item.paragraph_format.space_after = Pt(8)
    run = item.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(6.5)
    run.font.color.rgb = RGBColor(45, 45, 45)


def page_break(doc):
    doc.add_page_break()


def add_image(doc, path, caption, width=5.8):
    if not path.exists():
        return
    item = doc.add_paragraph()
    item.alignment = WD_ALIGN_PARAGRAPH.CENTER
    item.add_run().add_picture(str(path), width=Inches(width))
    caption_item = doc.add_paragraph(caption)
    caption_item.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_item.runs[0].italic = True


def image_count(path):
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def load_metrics():
    path = ROOT / "reports" / "yolo26_baseline_metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def metric_text(metrics, name):
    item = metrics.get(name, {})
    if not item:
        return "belum tersedia"
    if name == "disease_classification":
        return f"Top-1 {item['top1'] * 100:.2f}%; Top-5 {item['top5'] * 100:.2f}%"
    return f"P {item['precision']:.4f}; R {item['recall']:.4f}; mAP50 {item['map50']:.4f}; mAP50-95 {item['map50_95']:.4f}"


def build_report():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1.25)
    section.right_margin = Inches(1)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.15

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("LAPORAN REVISI BASELINE\nYOLO26 YOLOCOMVIS")
    run.bold = True
    run.font.size = Pt(21)
    run.font.color.rgb = RGBColor(23, 54, 93)
    subtitle = doc.add_paragraph("Deteksi Ikan, Deteksi Lesi, dan Klasifikasi Penyakit Ikan")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].italic = True
    table(doc, ["Item", "Keterangan"], [
        ["Eksperimen", "Baseline YOLO26 tanpa Dynamic Attention"],
        ["Model", "YOLO26n dan YOLO26n-cls"],
        ["Training target", "10 epoch, seed 42"],
        ["Status metrik", "Metrik YOLO26 pretrained/zero-shot pada seluruh validation set"],
    ])
    doc.add_page_break()

    metrics = load_metrics()
    heading(doc, "Ringkasan Eksekutif", 1)
    paragraph(doc, "Laporan ini merupakan revisi lengkap terhadap Tugas 2. Revisi menyelaraskan baseline dengan dasar teori yang menjadikan YOLO26 sebagai model dasar, tetapi tidak menerapkan Dynamic Attention Module. Sistem tetap terdiri atas detektor ikan, crop ikan, detektor lesi, classifier penyakit, dan dashboard.")
    paragraph(doc, "Metrik pada laporan ini diambil dari evaluasi checkpoint resmi YOLO26 pretrained pada seluruh validation set lokal. Karena training fine-tuning 10 epoch tidak selesai pada CPU yang tersedia, metrik ini diberi label zero-shot/pretrained dan tidak disamakan dengan hasil training YOLO11 pada laporan lama.")
    table(doc, ["Komponen", "Model", "Metrik YOLO26 terbaru"], [
        ["Deteksi ikan", "YOLO26n", metric_text(metrics, "fish_detection")],
        ["Deteksi lesi", "YOLO26n", metric_text(metrics, "lesion_detection")],
        ["Klasifikasi", "YOLO26n-cls", metric_text(metrics, "disease_classification")],
    ])

    heading(doc, "1. Latar Belakang dan Tujuan Revisi", 1)
    paragraph(doc, "Laporan Tugas 2 sebelumnya menggunakan YOLO11n dan YOLO11n-cls, walaupun judul dan dasar teori membahas YOLO26 serta Dynamic Attention. Revisi ini memperbaiki ketidaksesuaian tersebut pada level baseline saja.")
    bullets(doc, ["Mengganti detector YOLO11n menjadi YOLO26n.", "Mengganti classifier YOLO11n-cls menjadi YOLO26n-cls.", "Mempertahankan pipeline deteksi ikan, crop, deteksi lesi, dan klasifikasi.", "Menetapkan seed split dan training ke 42 serta resume=False.", "Menyediakan metrik dan artefak baru yang diberi status sumber secara eksplisit.", "Menunda Dynamic Attention untuk eksperimen lanjutan."])

    heading(doc, "2. Arsitektur Baseline Revisi", 1)
    table(doc, ["Tahap", "Masukan", "Proses", "Keluaran"], [
        ["Deteksi ikan", "Frame/foto", "YOLO26n detect", "Bounding box dan confidence"],
        ["Crop", "Bounding box ikan", "Pemotongan citra", "Citra ikan terlokalisasi"],
        ["Deteksi lesi", "Crop ikan", "YOLO26n detect", "Bounding box lesi"],
        ["Klasifikasi", "Crop ikan", "YOLO26n-cls", "Top-3 kelas dan probabilitas"],
        ["Dashboard", "Semua keluaran", "OpenCV compositing", "Visualisasi gabungan"],
    ])
    paragraph(doc, "Baseline ini memakai kemampuan native dari checkpoint YOLO26 resmi. Tidak ada kode DAM tambahan, tidak ada perubahan backbone/neck manual, dan tidak ada klaim bahwa baseline sudah memperoleh peningkatan attention.")
    code(doc, "YOLO('yolo26n.pt')        # deteksi ikan dan lesi\nYOLO('yolo26n-cls.pt')    # klasifikasi penyakit\n# DAM sengaja belum digunakan")

    heading(doc, "3. Dataset dan Persiapan Data", 1)
    counts = [
        ["Fish4Knowledge", "Deteksi ikan", image_count(ROOT / "datasets/fish4knowledge/images/train"), image_count(ROOT / "datasets/fish4knowledge/images/val"), "fish"],
        ["FishDisease", "Deteksi lesi", image_count(ROOT / "datasets/fishdisease/images/train"), image_count(ROOT / "datasets/fishdisease/images/val"), "lesion"],
        ["Freshwater Kaggle", "Klasifikasi", image_count(ROOT / "datasets/freshwater_kaggle/train"), image_count(ROOT / "datasets/freshwater_kaggle/val"), "7 kelas"],
    ]
    table(doc, ["Dataset", "Tujuan", "Train", "Val", "Label"], counts)
    paragraph(doc, "Konversi Fish4Knowledge memakai mask grayscale dan bounding rectangle kontur terbesar. FishDisease dikonversi dari JSON bounding box ke format YOLO satu kelas. Dataset Kaggle mempertahankan nama folder sebagai label klasifikasi.")
    paragraph(doc, "Split disiapkan dengan seed 42 pada pipeline terpadu. Keterbatasan metodologis tetap dicatat: pembagian saat ini belum sepenuhnya berbasis trajectory, individu, atau sumber video, sehingga evaluasi independen tetap diperlukan.")

    heading(doc, "4. Revisi Kode", 1)
    table(doc, ["File", "Revisi"], [
        ["scripts/3_training_all.py", "YOLO26n/YOLO26n-cls, seed 42, deterministic=True, resume=False"],
        ["all_in_one_yolocomvis.py", "Training, evaluasi, dan dashboard memakai checkpoint YOLO26"],
        ["scripts/4_tes_pipeline_final.py", "Label dashboard diperbaiki menjadi YOLO26; DAM dinyatakan belum digunakan"],
        ["scripts/evaluate_yolo26_baseline.py", "Evaluasi checkpoint pretrained dan ekspor metrik JSON"],
        ["lapbase_yolo26.py", "Generator laporan revisi berbasis artefak terbaru"],
    ])
    code(doc, "model_ikan = YOLO('yolo26n.pt')\nmodel_lesi = YOLO('yolo26n.pt')\nmodel_penyakit = YOLO('yolo26n-cls.pt')\nmodel_ikan.train(..., seed=42, deterministic=True, resume=False)")

    heading(doc, "5. Konfigurasi Eksperimen", 1)
    table(doc, ["Parameter", "Deteksi ikan", "Deteksi lesi", "Klasifikasi"], [
        ["Model", "yolo26n.pt", "yolo26n.pt", "yolo26n-cls.pt"],
        ["Epoch target", "10", "10", "10"],
        ["Image size", "640", "640", "224"],
        ["Seed", "42", "42", "42"],
        ["Deterministic", "True", "True", "True"],
        ["Resume", "False", "False", "False"],
        ["Dynamic Attention", "Tidak", "Tidak", "Tidak"],
    ])
    paragraph(doc, "Ultralytics memilih optimizer secara otomatis melalui optimizer=auto. CPU tersedia pada environment ini, tanpa CUDA. Training fine-tuning penuh 10 epoch dicoba tetapi dihentikan karena durasi CPU terlalu panjang; evaluasi yang selesai pada laporan ini adalah zero-shot checkpoint pretrained.")

    heading(doc, "6. Metrik YOLO26 Terbaru", 1)
    paragraph(doc, "Metrik berikut berasal dari checkpoint resmi YOLO26 yang dievaluasi pada seluruh validation set proyek. Angka ini merupakan baseline awal sebelum fine-tuning pada dataset lokal.")
    table(doc, ["Task", "Precision/Top-1", "Recall/Top-5", "mAP50", "mAP50-95"], [
        ["Deteksi ikan", metrics.get("fish_detection", {}).get("precision", "N/A"), metrics.get("fish_detection", {}).get("recall", "N/A"), metrics.get("fish_detection", {}).get("map50", "N/A"), metrics.get("fish_detection", {}).get("map50_95", "N/A")],
        ["Deteksi lesi", metrics.get("lesion_detection", {}).get("precision", "N/A"), metrics.get("lesion_detection", {}).get("recall", "N/A"), metrics.get("lesion_detection", {}).get("map50", "N/A"), metrics.get("lesion_detection", {}).get("map50_95", "N/A")],
        ["Klasifikasi", metrics.get("disease_classification", {}).get("top1", "N/A"), metrics.get("disease_classification", {}).get("top5", "N/A"), "-", "-"],
    ])
    paragraph(doc, "Metrik training YOLO11 pada laporan Tugas 2 tidak dihapus dari sejarah proyek, tetapi tidak digunakan sebagai hasil revisi. Setelah fine-tuning YOLO26 selesai, tabel ini harus diganti atau dilengkapi dengan hasil checkpoint best.pt yang dilatih pada dataset lokal dan dievaluasi pada seluruh validation set.")

    heading(doc, "7. Inferensi dan Dashboard", 1)
    paragraph(doc, "Inferensi mempertahankan alur end-to-end: gambar penuh diproses oleh detektor ikan, setiap bounding box dipotong, crop diproses oleh detektor lesi dan classifier, kemudian hasilnya digambar pada dashboard. Dashboard menampilkan bounding box ikan, bounding box lesi, confidence, jumlah lesi, dan Top-3 kelas penyakit.")
    add_image(doc, ROOT / "hasil_dashboard_baseline.jpg", "Gambar 1. Dashboard baseline pipeline.")

    heading(doc, "8. Perbandingan Sebelum dan Sesudah Revisi", 1)
    table(doc, ["Aspek", "Sebelum", "Sesudah"], [
        ["Detector", "YOLO11n", "YOLO26n"],
        ["Classifier", "YOLO11n-cls", "YOLO26n-cls"],
        ["Seed training", "0 pada artefak lama", "42"],
        ["Resume", "Model lesi melanjutkan last.pt", "resume=False"],
        ["Dashboard", "Label YOLOv26 tidak konsisten", "Label YOLO26 baseline"],
        ["Attention", "Belum ada", "Tetap belum ada; sengaja ditunda"],
        ["Metrik", "Hasil fine-tuning YOLO11", "Evaluasi YOLO26 pretrained/zero-shot"],
    ])

    heading(doc, "9. Keterbatasan dan Rencana Berikutnya", 1)
    bullets(doc, [
        "Metrik baru belum merupakan hasil fine-tuning karena training 10 epoch pada CPU tidak selesai dalam sesi ini.",
        "Validation set belum sepenuhnya independen terhadap trajectory, individu, atau duplikasi sumber.",
        "Belum ada test set eksternal dan belum ada pengukuran FPS end-to-end.",
        "Dynamic Attention belum diimplementasikan dan tidak boleh diklaim berkontribusi pada metrik baseline.",
        "Langkah berikutnya adalah menjalankan training YOLO26 10 epoch pada GPU/Colab, lalu mengganti metrik zero-shot dengan metrik best.pt hasil fine-tuning.",
        "Setelah baseline YOLO26 stabil, barulah eksperimen YOLO26 + DAM dilakukan dengan seed, data, dan anggaran training yang sama.",
    ])

    heading(doc, "10. Kesimpulan", 1)
    paragraph(doc, "Revisi telah menyelaraskan kode baseline dengan dasar teori pada pemilihan model: detector sekarang menggunakan YOLO26n dan classifier menggunakan YOLO26n-cls. Pipeline tiga tahap, format dataset, ukuran input, evaluasi, dan dashboard dipertahankan. Dynamic Attention belum digunakan sehingga baseline dapat menjadi pembanding yang bersih untuk eksperimen DAM di tahap berikutnya.")
    paragraph(doc, "Metrik yang tercantum dalam laporan ini adalah metrik YOLO26 pretrained/zero-shot pada seluruh validation set lokal. Hasil tersebut merupakan pemeriksaan awal kompatibilitas model dan dataset, bukan klaim performa fine-tuned. Pelaporan final harus diperbarui setelah training YOLO26 selesai pada perangkat yang memadai.")

    heading(doc, "11. Proses Lengkap dari Awal sampai Akhir", 1)
    paragraph(doc, "Bagian ini menjelaskan urutan kerja versi terbaru agar proses eksperimen dapat diikuti oleh pembaca yang tidak melihat kode secara langsung.")
    table(doc, ["Urutan", "Tahap", "Input", "Output"], [
        ["1", "Menyiapkan environment", "Python, Ultralytics, OpenCV, split-folders", "Environment siap"],
        ["2", "Menyiapkan data ikan", "Gambar dan mask Fish4Knowledge", "Gambar dan label YOLO fish"],
        ["3", "Menyiapkan data lesi", "Gambar dan JSON FishDisease", "Gambar dan label YOLO lesion"],
        ["4", "Menyiapkan klasifikasi", "Folder kelas Freshwater Kaggle", "Train/val tujuh kelas"],
        ["5", "Validasi struktur", "Folder train/val", "Ringkasan JSON/CSV/Markdown"],
        ["6", "Training baseline", "YAML dan folder dataset", "Checkpoint YOLO26"],
        ["7", "Evaluasi", "Checkpoint dan validation set", "Precision, recall, mAP, Top-1, Top-5"],
        ["8", "Inferensi", "Satu gambar", "Dashboard ikan, lesi, dan klasifikasi"],
        ["9", "Dokumentasi", "Kode dan artefak", "DOCX laporan revisi"],
    ])
    paragraph(doc, "Perbedaan penting versi terbaru adalah model pada tahap training dan evaluasi kini menunjuk YOLO26. Dynamic Attention tidak masuk ke urutan ini sehingga hasil baseline tetap menjadi pembanding sebelum eksperimen DAM.")
    code(doc, "python all_in_one_yolocomvis.py --mode prepare --seed 42\npython scripts/3_training_all.py\npython scripts/evaluate_yolo26_baseline.py\npython all_in_one_yolocomvis.py --mode dashboard --image ujicoba.png\npython lapbase_yolo26.py")

    page_break(doc)
    heading(doc, "Lampiran A. Daftar Revisi Kode", 1)
    paragraph(doc, "Lampiran ini memetakan perubahan dari kode baseline lama ke versi terbaru. Setiap perubahan memiliki dampak langsung terhadap kesesuaian dengan dasar teori dan reproducibility.")
    table(doc, ["Lokasi", "Sebelum", "Sesudah", "Alasan"], [
        ["scripts/3_training_all.py", "yolo11n.pt", "yolo26n.pt", "Menyamakan model baseline dengan teori YOLO26"],
        ["scripts/3_training_all.py", "yolo11n-cls.pt", "yolo26n-cls.pt", "Classifier harus berasal dari keluarga model yang sama"],
        ["Training", "Seed default/artefak 0", "seed=42", "Split dan training lebih mudah direproduksi"],
        ["Training", "resume tidak eksplisit", "resume=False", "Mencegah model melanjutkan checkpoint lama tanpa sengaja"],
        ["Evaluasi", "Checkpoint lama YOLO11", "Checkpoint YOLO26 atau pretrained resmi", "Metrik harus diberi sumber yang jelas"],
        ["Dashboard", "Teks YOLOv26/YOLO11 tidak konsisten", "YOLO26 baseline", "Menghindari klaim nama model yang salah"],
        ["Attention", "Belum ada", "Tetap belum ada", "DAM ditunda agar baseline bersih"],
        ["Laporan", "Metrik lama bercampur dengan revisi", "Status zero-shot/fine-tuning dibedakan", "Mencegah interpretasi hasil yang keliru"],
    ])

    page_break(doc)
    heading(doc, "Lampiran B. Kode Training Baseline", 1)
    paragraph(doc, "File: scripts/3_training_all.py. Script ini adalah entry point paling ringkas untuk melatih tiga model. Dua model deteksi memakai checkpoint YOLO26n, sedangkan classifier memakai YOLO26n-cls.")
    heading(doc, "B.1 Penjelasan", 2)
    bullets(doc, [
        "EPOCHS=10 menjaga anggaran pelatihan sesuai laporan Tugas 2.",
        "SEED=42 menyamakan konfigurasi dengan pembagian data pada pipeline terpadu.",
        "YOLO26n dipakai untuk fish detection dan lesion detection karena keduanya adalah task deteksi bounding box.",
        "YOLO26n-cls dipakai untuk folder klasifikasi karena task-nya image classification.",
        "deterministic=True membuat proses lebih terkontrol, sedangkan resume=False mencegah melanjutkan run lama.",
        "Dynamic Attention tidak muncul pada script ini; baseline hanya memakai checkpoint resmi YOLO26.",
    ])
    code_file(doc, ROOT / "scripts" / "3_training_all.py")

    page_break(doc)
    heading(doc, "Lampiran C. Kode Evaluasi Metrik", 1)
    paragraph(doc, "File: scripts/evaluate_yolo26_baseline.py. Script ini memuat checkpoint pretrained YOLO26 dan menjalankan model.val pada tiga validation set. Hasilnya disimpan sebagai reports/yolo26_baseline_metrics.json.")
    heading(doc, "C.1 Penjelasan", 2)
    bullets(doc, [
        "ROOT membuat lokasi dataset dan output tidak bergantung pada current working directory.",
        "Tiga tuple konfigurasi memisahkan nama task, checkpoint, data, dan image size.",
        "imgsz=640 digunakan untuk detector, sedangkan classifier memakai imgsz=224.",
        "fraction=1.0 berarti seluruh validation set dipakai pada evaluasi yang menghasilkan angka laporan ini.",
        "Detector membaca result.box untuk precision, recall, mAP50, dan mAP50-95.",
        "Classifier membaca top1 dan top5.",
        "Status zero-shot-pretrained-full-validation menandakan checkpoint belum fine-tuning pada dataset lokal.",
    ])
    code_file(doc, ROOT / "scripts" / "evaluate_yolo26_baseline.py")

    page_break(doc)
    heading(doc, "Lampiran D. Kode Preprocessing dan Pipeline Utama", 1)
    paragraph(doc, "File: all_in_one_yolocomvis.py. File ini menggabungkan persiapan dataset, ringkasan data, pembuatan YAML, training, evaluasi, pembuatan grafik, confusion matrix, dashboard, dan ekspor laporan.")
    heading(doc, "D.1 Bagian yang direvisi", 2)
    bullets(doc, [
        "train_models sekarang memanggil YOLO('yolo26n.pt') dan YOLO('yolo26n-cls.pt').",
        "train_models meneruskan seed, deterministic, dan resume=False.",
        "evaluate_models mencari checkpoint pada runs/detect dan runs/classify agar sesuai struktur Ultralytics.",
        "write_final_report menyebut YOLO26n dan menyatakan Dynamic Attention tidak digunakan.",
        "run_dashboard memuat checkpoint pada lokasi hasil training deteksi dan klasifikasi yang benar.",
    ])
    paragraph(doc, "Karena file ini panjang, lampiran menampilkan blok paling penting yang mengendalikan perubahan model dan evaluasi. Fungsi preprocessing tetap dipertahankan karena format label dan alur dataset tidak berubah.")
    code_file(doc, ROOT / "all_in_one_yolocomvis.py")

    page_break(doc)
    heading(doc, "Lampiran E. Kode Inferensi Dashboard", 1)
    paragraph(doc, "File: scripts/4_tes_pipeline_final.py. Script ini tidak melatih model. Ia memuat tiga best checkpoint, mendeteksi ikan pada gambar penuh, memotong setiap ikan, lalu meneruskan crop ke detector lesi dan classifier.")
    heading(doc, "E.1 Penjelasan alur", 2)
    bullets(doc, [
        "path_ikan dan path_lesi menunjuk checkpoint detector YOLO26.",
        "path_penyakit menunjuk checkpoint classifier YOLO26-cls.",
        "hasil_ikan.boxes menjadi sumber koordinat crop ikan.",
        "Koordinat lesi ditambah offset x1 dan y1 agar kembali ke canvas gambar penuh.",
        "top5 classifier dipakai untuk menampilkan tiga kelas teratas.",
        "Teks dashboard menyatakan baseline belum memakai Dynamic Attention.",
        "cv2.imwrite menyimpan hasil, sedangkan cv2.imshow hanya untuk tampilan interaktif.",
    ])
    code_file(doc, ROOT / "scripts" / "4_tes_pipeline_final.py")

    page_break(doc)
    heading(doc, "Lampiran F. Kode Preprocessing Dataset", 1)
    paragraph(doc, "Versi terpadu pada all_in_one_yolocomvis.py menjadi preprocessing yang direkomendasikan karena memiliki seed dan clipping bounding box. Script lama tetap disimpan sebagai referensi sejarah, tetapi memiliki random split tanpa seed.")
    heading(doc, "F.1 Fish4Knowledge", 2)
    paragraph(doc, "Mask dibaca sebagai grayscale. Kontur terbesar dipakai sebagai representasi satu objek ikan, lalu bounding rectangle dinormalisasi ke format class_id, center_x, center_y, width, height. Pasangan gambar-label diacak memakai Random(seed) dan 20% dipindahkan ke validation.")
    code_file(doc, ROOT / "scripts" / "1_proses_fish4knowledge.py")
    heading(doc, "F.2 FishDisease", 2)
    paragraph(doc, "JSON dibaca satu per satu. Field annotations dipakai untuk mengambil bbox. Versi pipeline terpadu menambahkan clipping ke batas gambar, menolak bbox kosong, dan menggunakan Random(seed) untuk pembagian data.")
    code_file(doc, ROOT / "scripts" / "2_proses_fishdisease.py")

    page_break(doc)
    heading(doc, "Lampiran G. Konfigurasi Dataset dan Artefak", 1)
    paragraph(doc, "YAML berikut mendefinisikan dataset deteksi dan nama kelas. File ini tidak berisi Dynamic Attention; YAML hanya memberi informasi data kepada Ultralytics.")
    heading(doc, "G.1 data_ikan.yaml", 2)
    code_file(doc, ROOT / "data_ikan.yaml")
    heading(doc, "G.2 data_lesi.yaml", 2)
    code_file(doc, ROOT / "data_lesi.yaml")
    table(doc, ["Artefak", "Kegunaan"], [
        ["reports/yolo26_baseline_metrics.json", "Metrik YOLO26 pretrained pada validation set"],
        ["runs/detect/val-2 dan val-3", "Grafik dan contoh prediksi evaluasi detector"],
        ["runs/classify/val", "Confusion matrix dan contoh prediksi classifier"],
        ["Laporan_Baseline...docx", "Laporan versi terbaru dengan lampiran kode"],
        ["yolo26n.pt dan yolo26n-cls.pt", "Checkpoint pretrained baseline YOLO26"],
    ])

    page_break(doc)
    heading(doc, "Lampiran H. Cara Membaca Metrik dan Batas Klaim", 1)
    paragraph(doc, "Precision mengukur proporsi prediksi positif yang benar. Recall mengukur proporsi objek ground truth yang berhasil ditemukan. mAP50 adalah mean average precision pada IoU 0,50, sedangkan mAP50-95 merata-ratakan AP pada rentang IoU 0,50 sampai 0,95. Top-1 menunjukkan kelas dengan probabilitas tertinggi benar, sedangkan Top-5 menunjukkan label benar berada di antara lima prediksi teratas.")
    paragraph(doc, "Metrik YOLO26 pada laporan ini adalah zero-shot/pretrained: bobot umum YOLO26 diuji pada data ikan dan penyakit lokal tanpa proses fine-tuning yang selesai. Karena itu, angka rendah pada deteksi lesi dan klasifikasi tidak boleh langsung dipakai untuk menyimpulkan bahwa arsitektur YOLO26 gagal. Angka tersebut menunjukkan domain gap dan menjadi baseline awal sebelum training lokal.")
    paragraph(doc, "Sebaliknya, angka pada laporan lama berasal dari fine-tuning YOLO11 dan tidak boleh dibandingkan sebagai eksperimen terkontrol sempurna dengan angka YOLO26 zero-shot. Perbandingan yang valid membutuhkan data split, seed, epoch, perangkat, dan status fine-tuning yang sama.")

    heading(doc, "Lampiran I. Checklist Reproduksi", 1)
    bullets(doc, [
        "Pastikan Python environment aktif dan paket ultralytics, opencv-python, numpy, python-docx, Pillow, split-folders, pandas tersedia.",
        "Pastikan data_ikan.yaml, data_lesi.yaml, dan folder freshwater_kaggle dapat dibaca.",
        "Jalankan mode prepare bila dataset dibangun dari data mentah.",
        "Jalankan scripts/3_training_all.py pada GPU/Colab untuk menyelesaikan fine-tuning 10 epoch.",
        "Jalankan scripts/evaluate_yolo26_baseline.py atau all_in_one_yolocomvis.py --mode evaluate.",
        "Jalankan scripts/4_tes_pipeline_final.py atau mode dashboard untuk uji satu gambar.",
        "Periksa args.yaml, results.csv, best.pt, confusion matrix, dan contoh prediksi sebelum menulis metrik final.",
        "Jangan menyebut DAM digunakan sebelum ada implementasi dan eksperimen pembanding yang terpisah.",
    ])

    heading(doc, "Lampiran. Perintah Reproduksi", 1)
    code(doc, "python scripts/3_training_all.py\npython scripts/evaluate_yolo26_baseline.py\npython all_in_one_yolocomvis.py --mode evaluate\npython all_in_one_yolocomvis.py --mode dashboard --image ujicoba.png\npython lapbase_yolo26.py")
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_report()
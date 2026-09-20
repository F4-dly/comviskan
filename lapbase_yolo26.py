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
OUTPUT = ROOT / "Laporan_Baseline_YOLOComVis_YOLO26_revisi_lengkap.docx"


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

    heading(doc, "Lampiran. Perintah Reproduksi", 1)
    code(doc, "python scripts/3_training_all.py\npython scripts/evaluate_yolo26_baseline.py\npython all_in_one_yolocomvis.py --mode evaluate\npython all_in_one_yolocomvis.py --mode dashboard --image ujicoba.png\npython lapbase_yolo26.py")
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_report()
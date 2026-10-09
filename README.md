# YOLOComVis: cara membaca dan menjalankan

## Status hasil saat ini

Dashboard adalah alat bantu eksperimen. Ia **belum layak dipakai untuk menyatakan ikan sehat/sakit atau memastikan ada/tidak ada lesi**. Model dilatih pada sumber gambar yang berbeda: deteksi ikan dari Fish4Knowledge, deteksi lesi dari FishDisease, dan klasifikasi dari dataset freshwater Kaggle. Rangkaian ketiganya pada foto pengguna belum divalidasi sebagai satu sistem.

Evaluasi terbaru yang tersimpan di `reports/evaluation_metrics.json` mencatat:

| Tugas | Metrik validasi tersimpan | Arti praktis |
|---|---:|---|
| Deteksi ikan | precision 0,999; recall 0,998; mAP50 0,995; mAP50-95 0,821 | Hasil validation split hasil rebuild; bukan jaminan pada foto baru atau test independen. |
| Deteksi lesi | precision 0,693; recall 0,450; mAP50 0,545; mAP50-95 0,235 | Sekitar 55% lesi acuan masih terlewat pada validasi; hasil kosong tidak membuktikan ikan bebas lesi. |
| Klasifikasi gambar | top-1 0,994; top-5 1,000; macro/weighted F1 sekitar 0,994 | Mengukur kelas gambar pada split klasifikasi, bukan diagnosis dari crop hasil detektor. |

Metrik tersebut berasal dari validasi yang tersimpan; bukan pengujian independen. Jumlah validasi klasifikasi pada ringkasan dataset adalah 700 (100 per kelas). Split klasifikasi yang tersedia belum memiliki sumber mentah untuk dibangun ulang. Karena itu, angka klasifikasi perlu diuji ulang dengan pemisahan berdasarkan ikan/sumber/pengambilan gambar dan test set yang tidak disentuh.

Dataset deteksi ikan dan lesi sudah dibangun ulang dengan pasangan image-label lengkap dan 0 exact duplicate train-val pada audit terbaru. Near-duplicate frame dan provenance sumber tetap perlu ditinjau.

## Menjalankan dashboard

Untuk uji satu foto dengan perintah paling singkat dari folder proyek:

```powershell
.\uji_satu.ps1 .\input_uji_coba\ikan_01.jpg
```

Hasil otomatis disimpan ke `reports/uji_satu/`. Jika ingin menentukan nama output:

```powershell
.\uji_satu.ps1 .\input_uji_coba\ikan_01.jpg .\reports\ikan_01_dashboard.jpg
```

Checkpoint default sekarang adalah model mixed compact karena tujuan utama uji coba
adalah foto pengguna/domain baru. Jadi perintah biasa otomatis memakai satu model
yang sama dan tidak perlu pindah-pindah checkpoint:

```powershell
.\uji_satu.ps1 .\input_uji_coba\ikan1.jpg
```

Checkpoint lama tetap disimpan sebagai pembanding historis. Jika ingin menjalankan
perbandingan khusus, checkpoint dapat dipilih eksplisit:

```powershell
.\uji_satu.ps1 .\input_uji_coba\ikan1.jpg "" .\runs\detect\Runs_DynamicAttention\model_ikan_mixed_v1_compact\weights\best.pt
```

Argumen checkpoint alternatif hanya untuk membandingkan eksperimen, bukan untuk
penggunaan normal.

`uji_satu.ps1` hanya launcher sederhana. Pipeline utama tetap `all_in_one_yolocomvis.py`; Anda tidak perlu menjalankan training atau script lain untuk uji satu foto. Jika PowerShell memblokir script lokal, jalankan:

```powershell
powershell -ExecutionPolicy Bypass -File .\uji_satu.ps1 .\input_uji_coba\ikan_01.jpg
```

Pastikan environment Python proyek aktif, dependensi `ultralytics`, `opencv-python`, `numpy`, dan PyTorch tersedia, lalu dari PowerShell di folder proyek:

```powershell
python scripts/4_tes_pipeline_final.py --image .\ujicoba.png
```

Untuk gambar lain:

```powershell
python scripts/4_tes_pipeline_final.py --image "D:\foto\ikan.jpg" --output .\hasil_dashboard_dynamic_attention.jpg
```

Untuk membuat dashboard validasi yang menyertakan acuan dataset dan prediksi per gambar:

```powershell
python scripts/5_dashboard_validasi_dengan_acuan.py
```

Hasilnya ada di `reports/dashboard_validasi_acuan/` dan ringkasan semua gambar ada di `perbandingan_per_gambar.csv`. Default menghasilkan 60 gambar berlabel untuk tiap tugas deteksi, 20 gambar ikan tambahan tanpa label pasangan (ditandai unknown), dan 40 gambar per kelas klasifikasi. Ubah jumlahnya dengan `--detection-samples` dan `--classification-per-class`.

Dashboard membaca checkpoint `best.pt` di `runs/detect/Runs_DynamicAttention/model_ikan*/weights`, `model_lesi/weights`, dan `runs/classify/Runs_DynamicAttention/model_penyakit/weights`. Pakai checkpoint yang sudah dievaluasi; jangan memilih file berbeda tanpa memperbarui evaluasi. `best.pt` berarti checkpoint terbaik menurut validasi training, bukan model yang pasti benar.

## Cara membaca dashboard

- Kotak dan skor ikan adalah keluaran detektor. Skor confidence bukan persentase akurasi.
- Kotak lesi adalah kandidat yang melewati ambang inferensi 0,25. Tidak ada kotak berarti model tidak menemukan kandidat pada ambang itu, bukan bukti tidak ada lesi. Objek kecil, buram, tertutup, atau di luar domain data bisa terlewat.
- Kelas penyakit dan skornya adalah prediksi klasifikasi atas crop ikan. Skor softmax bukan probabilitas klinis yang terkalibrasi dan bukan diagnosis laboratorium/veteriner.
- Pada dashboard validasi, kotak hijau adalah label acuan dataset dan kotak merah adalah prediksi. TP/FP/FN dicocokkan dengan IoU minimal 0,50; acuan dataset sendiri belum tentu benar. Gambar tanpa label pasangan ditandai unknown dan tidak dihitung sebagai FP/FN.
- “Perlu verifikasi manusia” berlaku untuk semua prediksi, termasuk skor tinggi. Gunakan pemeriksaan ahli dan metode pemeriksaan yang sesuai sebelum menyampaikan status kesehatan atau tindakan.
- Pastikan foto menampilkan ikan utuh/jelas, fokus, pencahayaan memadai, tidak terhalang, dan spesies serta kondisi pengambilannya sesuai dengan data validasi. Masukan di luar kondisi itu harus dianggap tidak tervalidasi.

### Ciri gambar yang paling cocok untuk uji coba

Untuk membantu ketiga tahap berjalan lebih konsisten, gunakan:

- satu ikan yang terlihat utuh dan tidak terpotong;
- posisi ikan menyamping atau diagonal ringan, dengan bagian kepala dan ekor terlihat;
- ikan memenuhi kira-kira 40--80% area gambar;
- fokus tajam, resolusi cukup, dan pencahayaan merata tanpa pantulan kuat;
- latar belakang sederhana dengan kontras yang cukup terhadap tubuh ikan;
- lesi terlihat jelas, tidak tertutup, dan tidak terlalu kecil atau buram;
- sudut kamera, spesies, air, dan pencahayaan yang mendekati data pelatihan.

Hindari gambar terlalu gelap, blur, ikan sangat kecil, ikan tertutup tanaman/tangan, banyak ikan bertumpuk, atau lesi yang hanya tampak sebagai titik beberapa piksel. Kondisi tersebut dapat membuat detektor ikan gagal menghasilkan crop sehingga lesi dan penyakit juga tidak dapat dianalisis.

### Sumber foto ikan yang paling kompatibel

Untuk uji coba tanpa melatih ulang, mulai dari sumber yang gaya gambarnya mendekati
data deteksi ikan saat ini:

1. [Fish4Knowledge Sample Dataset](https://groups.inf.ed.ac.uk/vision/DATASETS/FISH4KNOWLEDGE/WEBSITE/F4KDATASAMPLES/INTERFACE/DATASAMPLES/search.php)
   adalah pilihan utama. Pilih beberapa kamera, lokasi, dan hari, lalu ambil sampel
   kecil dari video bawah air. Sumber ini paling dekat dengan data Fish4Knowledge
   yang dipakai model.
2. [LILA Community Fish Detection Dataset](https://lila.science/datasets/community-fish-detection-dataset/)
   menyediakan gambar dan bounding box dalam format COCO dari banyak sumber. Untuk
   percobaan awal, pilih bagian `F4K Detection and Tracking`, `Brackish Dataset`,
   atau `Tropical freshwater fish in Northern Australia`; jangan mengunduh seluruh
   koleksi jutaan gambar.
3. [Tropical freshwater fish dataset](https://zenodo.org/records/7250921) dapat
   dipakai bila ingin tampilan ikan air tawar yang lebih relevan. Pilih gambar
   dengan ikan di lingkungan air, bukan foto katalog berlatar polos.
4. [Roboflow Fish Object Detection](https://public.roboflow.com/object-detection/fish/1)
   berguna untuk melihat contoh dan anotasi dengan cepat. Periksa lisensi sumber
   sebelum memakai gambarnya untuk training atau publikasi.

Ambil 10--30 gambar dulu dan simpan ke `input_uji_coba`. Nama file boleh dibuat
seperti `f4k_01.jpg`, `brackish_01.jpg`, dan `freshwater_01.jpg`, lalu jalankan:

```powershell
.\uji_satu.ps1 .\input_uji_coba\f4k_01.jpg
```

Foto yang paling aman untuk uji coba adalah satu ikan utuh, tidak terpotong, terlihat
20--80% dari lebar gambar, fokus, cukup terang, dan berada di dalam air. Hindari
foto studio, ikan di atas meja, close-up bagian tubuh, gambar AI, serta gambar yang
terlalu berbeda dari kamera bawah air. Lisensi setiap dataset harus tetap diperiksa
di halaman sumbernya; metadata dan sumber gambar perlu disimpan bila data dipakai
untuk training atau presentasi.

Dashboard utama telah dirapikan agar panel visual hanya menampilkan ikan, confidence deteksi ikan, kandidat lesi, dan prediksi penyakit. Confidence tetap merupakan skor model, bukan akurasi atau diagnosis; batasan lengkapnya tetap dicatat dalam laporan evaluasi dan quality gate.

### Dataset `Fish.v1-416x416.yolo26`

Dataset yang diletakkan di `input_uji_coba/Fish.v1-416x416.yolo26` sudah diaudit
secara lokal dan ringkasannya ada di `reports/fish_v1_dataset_audit.json`. Isinya
1.350 gambar berlabel dengan pembagian train/valid/test, tetapi memakai **26 kelas
spesies**. Model deteksi ikan ComVis saat ini memakai **satu kelas generik
`fish`**, sehingga dataset ini tidak dapat langsung dipakai untuk training tanpa
perubahan label.

Secara visual dataset ini cukup menarik untuk uji coba karena berisi adegan ikan
bawah air/terumbu dan lebih dekat ke domain Fish4Knowledge daripada foto ikan di
meja. Namun, jangan langsung mencampurnya dengan Fish4Knowledge. Gunakan dulu
sebagai external trial set pada 10--30 gambar. Jika hasilnya ingin dipakai untuk
training detektor ikan generik, semua kelas ikan harus dipetakan ke class `0`,
label perlu diaudit, dan evaluasi harus memakai pemisahan berdasarkan sumber
dataset/video agar tidak terjadi kebocoran.

Dataset ini **tidak memiliki label lesi atau penyakit**, sehingga menambahkannya
hanya berpotensi membantu tahap deteksi ikan. Ia tidak otomatis meningkatkan
deteksi lesi maupun klasifikasi penyakit. Rencana yang paling aman adalah:

1. uji gambar dataset ini dengan checkpoint sekarang;
2. cek apakah box ikan benar secara visual;
3. bila banyak yang cocok, buat eksperimen training baru dengan gabungan
   Fish4Knowledge + dataset ini, bukan mengganti checkpoint produksi sekarang;
4. bandingkan hasil pada test target-domain yang tidak dipakai training.

Eksperimen gabungan compact sudah dijalankan 5 epoch. Hasil perbandingan tersimpan
di `reports/fish_detector_comparison_mixed_v1.json`. Model eksperimen meningkatkan
hasil pada test Fish v1 dan mendeteksi lebih banyak foto pengguna, tetapi menurunkan
hasil pada validasi Fish4Knowledge. Karena prioritas penggunaan Anda adalah foto
pengguna/domain baru, model mixed compact sekarang dijadikan checkpoint default.
Model ini tetap belum berarti sempurna: pada batch `ikan1--10` sebelumnya ia
mendeteksi 4/10 gambar pada confidence 0,25, sehingga hasil kosong tetap tidak
membuktikan ikan tidak ada.

## Syarat sebelum publikasi

### Status audit terbaru

Audit read-only terbaru tersimpan di `reports/dataset_audit.json`. Audit menemukan 71
hash gambar ikan yang identik muncul di train dan val pada snapshot saat ini, sehingga
skor validasi ikan dan klasifikasi/deteksi tidak boleh diperlakukan sebagai skor test
independen. Generator dataset sekarang memakai split berbasis grup nama sumber untuk
mencegah frame dari satu urutan masuk ke kedua split pada pembuatan ulang berikutnya.
Jalankan mode `audit` sebelum dan sesudah menyiapkan dataset. Jangan menjalankan mode
`prepare` tanpa membuat salinan dataset terlebih dahulu karena mode tersebut membangun
ulang folder train/val.

```powershell
python all_in_one_yolocomvis.py --mode audit
```

Jangan menerbitkan keluaran sebagai fakta kesehatan, alat diagnosis, klaim deteksi akurat, atau saran pengobatan. Untuk publikasi ilmiah, dokumentasikan sumber dan izin dataset, definisi label lesi/penyakit, versi checkpoint dan kode, ambang confidence, ukuran serta cara pemisahan data, dan metrik per kelas pada test set independen. Audit anotasi oleh ahli; cegah kebocoran antar-split dengan pemisahan per individu ikan/video/sesi/lokasi; uji pada perangkat/kamera dan kondisi lapangan sasaran; laporkan precision, recall, mAP per kelas untuk deteksi serta confusion matrix, sensitivity/recall, specificity, dan interval ketidakpastian untuk klasifikasi. Sertakan contoh salah positif dan salah negatif serta batasan domain. Klaim kesehatan memerlukan validasi ahli dan standar pembanding yang relevan.

Untuk mengukur performa model baru, jalankan `python all_in_one_yolocomvis.py --mode evaluate` setelah memastikan YAML menunjuk dataset yang benar dan checkpoint yang hendak dilaporkan. Mode evaluasi tidak memperbaiki data yang bocor atau label yang salah; periksa sampel prediksi terhadap anotasi secara manual.

## Status sepuluh tahap perbaikan

1. Baseline dibekukan.
2. Dataset ikan dan lesi dibackup, dibangun ulang, dan diaudit.
3. Manifest kandidat test dikunci dengan hash.
4. Status test dicatat sebagai kandidat; belum independen atau ditinjau ahli.
5. Detektor ikan dilatih ulang dan divalidasi; detektor lesi dicoba ulang tetapi hasilnya lebih buruk sehingga checkpoint lama dipertahankan.
6. Metrik classifier, confusion matrix, dan latency tersedia pada validation split.
7. Smoke test end-to-end 10 gambar berhasil; metrik end-to-end diblokir tanpa ground truth gabungan.
8. Triage error smoke test tersedia di `reports/error_analysis_smoke.json`; ini bukan TP/FP/FN final.
9. Baseline tersedia, tetapi ablation belum controlled pada split identik; status ada di `reports/baseline_ablation_status.json`.
10. Reproducibility, provenance, dan batasan terdokumentasi.

Kesimpulan presentasi yang aman: sistem dapat dipresentasikan sebagai **prototipe/alat bantu eksperimen**, bukan sistem diagnosis final. Quality gate di `reports/final_quality_gate.json` sengaja tetap menolak klaim diagnosis sampai test independen, label ahli, dan evaluasi end-to-end tersedia.

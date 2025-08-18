# Sistem Penghitungan Kendaraan Bermotor dengan YOLOv8 dan PaddleOCR

Sistem penghitungan kendaraan bermotor real-time yang menggunakan deteksi objek YOLOv8 (PaddlePaddle), tracking ByteTrack, dan pembacaan plat nomor dengan PaddleOCR. Sistem ini dapat menghitung kendaraan yang melintasi area yang telah ditentukan dan membaca plat nomor secara otomatis.

## Fitur Utama

- **Deteksi Objek Real-time**: Menggunakan YOLOv8 dengan backend PaddlePaddle untuk deteksi kendaraan bermotor
- **Multi-Object Tracking**: Algoritma ByteTrack dengan Kalman Filter adaptif untuk tracking konsisten
- **Pembacaan Plat Nomor**: Integrasi PaddleOCR untuk membaca teks plat nomor otomatis
- **Penghitungan Bidirectional**: Counter terpisah untuk kendaraan masuk dan keluar
- **Anti-Double Counting**: Mencegah objek yang sama dihitung berulang kali
- **Visualisasi Real-time**: Tampilan bounding box, trail tracking, dan panel OCR
- **ROI Detection**: Area deteksi yang dapat dikonfigurasi
- **Performance Monitoring**: Monitoring FPS dan optimasi performa

## Arsitektur Sistem

Sistem ini terdiri dari tiga komponen utama:

### 1. **main_detector.py** - Modul Deteksi YOLOv8
- **Kelas YOLOv8_Paddle_Detector**: Wrapper untuk model YOLOv8 dengan backend PaddlePaddle
- **Preprocessing**: Normalisasi gambar dan resize sesuai input model
- **Inference**: Deteksi objek dengan confidence dan NMS threshold
- **Postprocessing**: Filtering kelas dan scaling bounding box
- **Fungsi Utilitas**: `detect_and_crop_objects`, `draw_detections`, `draw_tracks`

### 2. **bytetrack.py** - Algoritma Multi-Object Tracking
- **ByteTrack**: Implementasi algoritma tracking state-of-the-art
- **AdaptiveKalmanXYSR**: Kalman Filter dengan noise adaptif untuk prediksi posisi
- **EnhancedTrack**: Manajemen status track (tentative, confirmed, deleted)
- **Trail Management**: Penyimpanan jejak pergerakan objek untuk visualisasi
- **Association**: Hungarian algorithm untuk mencocokkan deteksi dengan track

### 3. **main.py** - Aplikasi Utama
- **TrailCrossCounter**: Logika penghitungan berdasarkan crossing ROI
- **PaddleOCR Integration**: Pembacaan teks plat nomor
- **Visualization**: Rendering real-time dengan panel OCR
- **Performance Monitoring**: FPS tracking dan optimasi frame skipping

## Requirements

### System Requirements
- Python 3.8 atau lebih tinggi
- Windows 10/11 (tested)
- GPU direkomendasikan untuk performa optimal
- RAM minimal 8GB

### Dependencies
```
opencv-python>=4.5.0
paddlepaddle>=2.4.0
paddleocr>=2.6.0
numpy>=1.21.0
scipy>=1.7.0
```

## Instalasi

1. **Clone atau download repository ini**
   ```bash
   git clone <repository-url>
   cd moto_counting
   ```

2. **Buat virtual environment (direkomendasikan)**
   ```bash
   python -m venv env
   env\Scripts\activate  # Pada Windows
   ```

3. **Install dependencies**
   ```bash
   pip install paddlepaddle paddleocr opencv-python numpy scipy
   ```

4. **Persiapkan model dan data**
   - Siapkan model YOLOv8 PaddlePaddle (format .pdmodel, .pdiparams)
   - Siapkan file class names (.txt)
   - Siapkan model PaddleOCR untuk deteksi dan recognition teks
   - Siapkan video input atau gunakan webcam (camera index 0)

## Konfigurasi

Edit bagian konfigurasi di `main.py` untuk menyesuaikan sistem:

```python
# Konfigurasi Model YOLO
YOLO_MODEL_DIR = "models/yolo_model"  # Direktori model YOLOv8 PaddlePaddle
CLASS_NAMES_FILE = "models/class_names.txt"  # File nama kelas

# Konfigurasi Model OCR
OCR_DET_MODEL_DIR = "models/ocr_det"  # Model deteksi teks PaddleOCR
OCR_REC_MODEL_DIR = "models/ocr_rec"  # Model recognition teks PaddleOCR

# Konfigurasi Video
VIDEO_PATH = "input_video.mp4"  # Path video atau "0" untuk webcam
RESIZE_WIDTH = 1280   # Lebar frame (None untuk ukuran asli)
RESIZE_HEIGHT = 720   # Tinggi frame (None untuk ukuran asli)

# Konfigurasi Target
TARGET_CLASS = ["motorcycle"]  # Kelas objek yang akan dideteksi

# ROI (Region of Interest) untuk deteksi
DETECT_ROI = (100, 200, 800, 600)  # (x1, y1, x2, y2)

# Parameter Performa
FRAME_SKIP_DET = 2    # Skip frame untuk deteksi (0 = setiap frame)
FRAME_SKIP_OCR = 5    # Skip frame untuk OCR
MAX_MS_PER_FRAME_WARN = 100  # Warning jika processing > threshold

# Visualisasi
DEBUG_VIS = True      # Tampilkan visualisasi debug
DRAW_TRAILS = True    # Gambar jejak pergerakan objek
TRAIL_THICKNESS = 2   # Ketebalan garis jejak
```

### Pengaturan ROI (Region of Interest)

1. **Tentukan Koordinat ROI**: Gunakan image viewer untuk mengidentifikasi koordinat pixel
2. **Format ROI**: Didefinisikan sebagai `(x1, y1, x2, y2)` dimana (x1,y1) adalah titik kiri atas dan (x2,y2) adalah titik kanan bawah
3. **Area Deteksi**: Hanya objek dalam ROI yang akan dihitung
4. **Crossing Logic**: Sistem menghitung objek yang melintasi batas ROI

## Penggunaan

### Menjalankan Aplikasi

```bash
python main.py
```

### Kontrol
- **'q'**: Keluar dari aplikasi
- **ESC**: Metode alternatif untuk keluar

### Output Sistem

Sistem menyediakan:
- **Tampilan video real-time** dengan bounding box, ROI, dan trail tracking
- **Panel OCR** menampilkan hasil pembacaan plat nomor terbaru
- **Statistik counting** (masuk/keluar) di layar
- **Console output** dengan event detection dan OCR
- **Monitoring FPS** untuk performa sistem

### Contoh Output Console
```
Deteksi Baru (ID: 15, Tipe: enter): B1234XYZ
Deteksi Baru (ID: 23, Tipe: exit): D5678ABC
Deteksi Baru (ID: 31, Tipe: enter): F9012DEF

Masuk : 2
Keluar: 1
FPS: 25.3
```

## Cara Kerja Sistem

### Pipeline Pemrosesan Per Frame

1. **Pra-pemrosesan Frame**
   - Resize frame sesuai konfigurasi
   - Frame skipping untuk optimasi performa
   - ROI extraction untuk area deteksi

2. **Deteksi Objek (YOLOv8)**
   - Input frame ke model YOLOv8 PaddlePaddle
   - Normalisasi dan preprocessing gambar
   - Inference dengan confidence threshold
   - Non-Maximum Suppression (NMS)
   - Filtering berdasarkan kelas target

3. **Multi-Object Tracking (ByteTrack)**
   - **Prediksi**: Kalman Filter memprediksi posisi objek
   - **Asosiasi Pertama**: Matching deteksi confidence tinggi dengan track confirmed
   - **Asosiasi Kedua**: Matching sisa deteksi dengan track tentative
   - **Inisialisasi Track Baru**: Dari deteksi confidence tinggi yang tidak ter-match
   - **Update Track**: Update status dan posisi track yang ter-match
   - **Manajemen Track**: Hapus track yang sudah terlalu lama tidak ter-update

4. **Counting Logic**
   - **Trail Analysis**: Analisis jejak pergerakan setiap track
   - **ROI Crossing Detection**: Deteksi objek yang melintasi batas ROI
   - **Direction Classification**: Klasifikasi arah (masuk/keluar) berdasarkan crossing
   - **Anti-Double Counting**: Setiap track hanya dihitung sekali

5. **OCR Processing**
   - **Crop Extraction**: Ekstrak area objek yang terdeteksi
   - **Image Enhancement**: Preprocessing untuk meningkatkan kualitas OCR
   - **PaddleOCR Inference**: Deteksi dan recognition teks plat nomor
   - **Result Caching**: Simpan hasil OCR terbaru untuk visualisasi

6. **Visualisasi dan Output**
   - **Bounding Box**: Gambar kotak deteksi dengan ID track
   - **Trail Rendering**: Visualisasi jejak pergerakan objek
   - **ROI Overlay**: Tampilkan area deteksi
   - **OCR Panel**: Panel khusus untuk menampilkan hasil OCR
   - **Statistics**: Counter masuk/keluar dan FPS

### Algoritma Tracking ByteTrack

**ByteTrack** menggunakan pendekatan dua tahap:
1. **High Confidence Association**: Mencocokkan deteksi confidence tinggi dengan track yang sudah confirmed
2. **Low Confidence Recovery**: Menggunakan deteksi confidence rendah untuk memulihkan track yang hilang

**Kalman Filter Adaptif**:
- Prediksi posisi berdasarkan model motion constant velocity
- Noise adaptif berdasarkan kualitas tracking
- State vector: [center_x, center_y, scale, ratio, dx, dy, ds, dr]

### Logika Counting

Sistem menggunakan **TrailCrossCounter** yang:
1. **Menganalisis Trail**: Memeriksa jejak pergerakan setiap objek
2. **Deteksi Crossing**: Mengidentifikasi kapan objek melintasi batas ROI
3. **Klasifikasi Arah**: Menentukan arah berdasarkan posisi relatif terhadap ROI
4. **Event Generation**: Menghasilkan event 'enter' atau 'exit' untuk setiap crossing

## Troubleshooting

### Masalah Umum

1. **Model tidak ditemukan**
   - Pastikan direktori model YOLOv8 PaddlePaddle ada dan berisi file .pdmodel, .pdiparams
   - Periksa konfigurasi `YOLO_MODEL_DIR` dan `CLASS_NAMES_FILE`
   - Pastikan file class names berformat .txt dengan satu kelas per baris

2. **Video tidak dapat dimuat**
   - Verifikasi path video dan format file
   - Gunakan absolute path jika diperlukan
   - Pastikan codec video didukung OpenCV
   - Untuk webcam, gunakan `VIDEO_PATH = "0"`

3. **Akurasi deteksi rendah**
   - Sesuaikan `conf_threshold` di inisialisasi detector (default 0.4)
   - Periksa kualitas pencahayaan dan resolusi video
   - Pastikan model telah dilatih dengan data yang sesuai
   - Sesuaikan `nms_threshold` untuk mengurangi duplikasi deteksi

4. **Counting tidak akurat**
   - Periksa koordinat ROI dan sesuaikan dengan area yang diinginkan
   - Pastikan ROI tidak terlalu kecil atau terlalu besar
   - Periksa parameter tracking (iou_threshold, max_age, min_hits)
   - Sesuaikan `FRAME_SKIP_DET` untuk balance antara akurasi dan performa

5. **OCR tidak akurat**
   - Periksa kualitas crop gambar plat nomor
   - Sesuaikan preprocessing image enhancement
   - Pastikan model PaddleOCR sesuai dengan bahasa/format plat nomor
   - Periksa pencahayaan dan kontras gambar

6. **Performa lambat**
   - Tingkatkan `FRAME_SKIP_DET` dan `FRAME_SKIP_OCR`
   - Kurangi resolusi video (`RESIZE_WIDTH`, `RESIZE_HEIGHT`)
   - Gunakan GPU jika tersedia (`use_gpu=True`)
   - Sesuaikan `trail_maxlen` untuk mengurangi memory usage

### Optimasi Performa

- **GPU Acceleration**: Pastikan PaddlePaddle dengan GPU support terinstall
- **Model Size**: Gunakan model YOLOv8 yang lebih kecil untuk processing lebih cepat
- **Resolution**: Kurangi resolusi video untuk real-time processing
- **Frame Skipping**: Sesuaikan parameter skip frame berdasarkan kebutuhan akurasi vs speed
- **ROI Optimization**: Gunakan ROI yang optimal untuk mengurangi area processing

## Struktur File

```
moto_counting/
├── main.py                    # Aplikasi utama
├── main_detector.py           # Modul deteksi YOLOv8 PaddlePaddle
├── bytetrack.py              # Implementasi algoritma ByteTrack
├── models/
│   ├── yolo_model/           # Model YOLOv8 PaddlePaddle
│   │   ├── model.pdmodel     # Model architecture
│   │   ├── model.pdiparams   # Model weights
│   │   └── infer_cfg.yml     # Inference config
│   ├── class_names.txt       # File nama kelas
│   ├── ocr_det/             # Model deteksi teks PaddleOCR
│   └── ocr_rec/             # Model recognition teks PaddleOCR
├── input_video.mp4           # Video input (opsional)
├── requirements.txt          # Dependencies Python
├── README.md                # Dokumentasi ini
└── env/                     # Virtual environment (opsional)
```

## Parameter Konfigurasi Lanjutan

### YOLOv8 Detector Parameters
```python
detector = YOLOv8_Paddle_Detector(
    model_dir=YOLO_MODEL_DIR,
    class_names_file=CLASS_NAMES_FILE,
    use_gpu=True,              # Gunakan GPU jika tersedia
    device_id=0,               # ID GPU device
    conf_threshold=0.4,        # Confidence threshold
    nms_threshold=0.5          # NMS threshold
)
```

### ByteTrack Parameters
```python
tracker = ByteTrack(
    high_conf_thresh=0.6,      # Threshold confidence tinggi
    low_conf_thresh=0.1,       # Threshold confidence rendah
    iou_threshold=0.3,         # IoU threshold untuk association
    max_age=30,                # Max frame tanpa update sebelum dihapus
    min_hits=3,                # Min hits sebelum track confirmed
    use_giou=True,             # Gunakan GIoU instead of IoU
    adaptive_noise=True,       # Adaptive Kalman filter noise
    trail_maxlen=30            # Panjang maksimal trail
)
```

### PaddleOCR Parameters
```python
ocr = PaddleOCR(
    text_detection_model_dir=OCR_DET_MODEL_DIR,
    text_recognition_model_dir=OCR_REC_MODEL_DIR,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)
```

## Pengembangan Lanjutan

### Menambah Kelas Deteksi Baru
1. Update file `class_names.txt` dengan kelas baru
2. Retrain model YOLOv8 dengan dataset yang mencakup kelas baru
3. Update `TARGET_CLASS` di konfigurasi

### Meningkatkan Akurasi OCR
1. Implementasikan preprocessing yang lebih baik di fungsi `enhance_plate`
2. Fine-tune model PaddleOCR dengan dataset plat nomor lokal
3. Tambahkan post-processing untuk koreksi hasil OCR

### Optimasi Real-time
1. Implementasikan multi-threading untuk deteksi dan OCR
2. Gunakan model quantization untuk inference lebih cepat
3. Implementasikan adaptive frame skipping berdasarkan load CPU/GPU

## Lisensi

Proyek ini dilisensikan di bawah MIT License - lihat file LICENSE untuk detail.

## Acknowledgments

- **PaddlePaddle**: Framework deep learning yang powerful
- **PaddleOCR**: Toolkit OCR yang excellent
- **ByteTrack**: Algoritma tracking state-of-the-art
- **OpenCV**: Library computer vision yang fundamental

---

**Catatan**: Sistem ini dirancang untuk penghitungan kendaraan bermotor dengan pembacaan plat nomor, namun dapat diadaptasi untuk jenis kendaraan lain dengan melatih ulang model YOLOv8 menggunakan dataset yang sesuai.
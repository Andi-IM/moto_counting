# Sistem Penghitungan Kendaraan Bermotor dengan Deteksi Plat Nomor

## Gambaran Umum Proyek

Proyek ini adalah sistem **real-time** untuk menghitung kendaraan bermotor yang masuk dan keluar dari suatu area dengan menggunakan:
- **YOLOv8** (format PaddlePaddle) untuk deteksi plat nomor kendaraan
- **PaddleOCR** untuk membaca teks pada plat nomor
- **ByteTrack** untuk pelacakan objek (tracking)
- **Computer Vision** untuk analisis pergerakan kendaraan

### Tujuan Sistem
1. **Mendeteksi** plat nomor kendaraan dalam video real-time
2. **Membaca** teks plat nomor menggunakan OCR
3. **Menghitung** jumlah kendaraan yang masuk dan keluar
4. **Melacak** pergerakan kendaraan untuk mencegah double counting

---

## Arsitektur Sistem

```
[Video Input] → [Frame Processing] → [YOLO Detection] → [OCR Reading] → [Tracking] → [Counting]
      ↓              ↓                    ↓               ↓            ↓          ↓
   Webcam/File    Preprocessing      Plat Detection   Text Extract  ByteTrack  Motor Counter
```

### Komponen Utama

#### 1. **main_detector.py** - Modul Deteksi Utama
- **Kelas `YOLOv8_Paddle_Detector`**: Wrapper untuk model YOLOv8 dalam format PaddlePaddle
- **Fungsi deteksi**: Mendeteksi objek (plat nomor) dalam gambar
- **Fungsi cropping**: Memotong area plat yang terdeteksi
- **Fungsi tracking**: Integrasi dengan ByteTrack untuk pelacakan objek

#### 2. **main.py** - Pipeline Utama
- **Kelas `MotorCounter`**: Logika penghitungan kendaraan masuk/keluar
- **Fungsi `enhance_plate`**: Preprocessing gambar plat untuk OCR yang lebih akurat
- **Pipeline utama**: Orchestrasi seluruh proses dari video input hingga output

#### 3. **bytetrack.py** - Sistem Tracking
- **ByteTrack algorithm**: Pelacakan objek untuk mencegah double counting
- **Kalman Filter**: Prediksi posisi objek di frame berikutnya
- **Data association**: Mencocokkan deteksi baru dengan track yang sudah ada

---

## Cara Kerja Sistem

### 1. **Inisialisasi Model**
```python
# Load YOLOv8 model untuk deteksi plat
detector = YOLOv8_Paddle_Detector(
    model_dir="model/yolo/yolov8n-fold5/best_paddle_model/inference_model",
    class_names_file="label.names"
)

# Load PaddleOCR untuk membaca teks plat
ocr = PaddleOCR(
    text_recognition_model_dir="model/ocr/PP-OCRv5_server_rec_infer",
    text_recognition_batch_size=8
)
```

### 2. **Pipeline Per Frame**

#### A. **Frame Preprocessing**
- Resize frame untuk optimasi kecepatan
- Definisi ROI (Region of Interest) untuk area deteksi
- Frame skipping untuk optimasi performa

#### B. **Deteksi Plat Nomor (YOLOv8)**
```python
# Deteksi plat di seluruh frame
all_cropped_plates, all_valid_boxes = detector.detect_and_crop_objects(
    allowed_classes=["license-plates"], 
    image_source=frame
)
```

**Proses internal YOLOv8:**
1. **Preprocessing**: Resize gambar ke 640x640, normalisasi pixel
2. **Inference**: Forward pass melalui neural network
3. **Postprocessing**: Non-Maximum Suppression (NMS), filtering confidence
4. **Output**: Bounding boxes koordinat plat yang terdeteksi

#### C. **Filtering ROI**
```python
# Filter hanya plat yang berada di dalam ROI
for i, box in enumerate(all_valid_boxes):
    x1, y1, x2, y2 = box
    if not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2):
        roi_intersecting_boxes.append(box)
        roi_cropped_plates.append(all_cropped_plates[i])
```

#### D. **Enhancement Gambar Plat**
```python
def enhance_plate(img_bgr):
    # 1. Resize untuk ukuran minimal
    # 2. Unsharp masking untuk ketajaman
    # 3. CLAHE untuk kontras adaptif
    # 4. Denoising (opsional)
    # 5. Border padding
```

**Teknik Enhancement:**
- **Unsharp Masking**: Meningkatkan ketajaman tepi karakter
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Denoising**: Mengurangi noise untuk OCR yang lebih akurat

#### E. **OCR Reading (PaddleOCR)**
```python
# Batch OCR untuk efisiensi
if enhanced_plates:
    ocr_results = ocr.predict(enhanced_plates)
    for result in ocr_results:
        # Parse hasil OCR
        plate_text = parse_ocr_item(result)
```

**Proses OCR:**
1. **Text Detection**: Mendeteksi area teks dalam gambar plat
2. **Text Recognition**: Mengkonversi pixel menjadi karakter
3. **Confidence Scoring**: Memberikan skor kepercayaan hasil

#### F. **Tracking dan Counting**
```python
# Update tracking dengan ByteTrack
motor_counter.update_tracking(roi_intersecting_boxes, roi_bounds)

# Logika counting berdasarkan pergerakan
class MotorCounter:
    def update_tracking(self, current_boxes, roi_bounds):
        # 1. Hitung centroid setiap bounding box
        # 2. Tracking pergerakan centroid
        # 3. Deteksi crossing line (masuk/keluar)
        # 4. Update counter
```

---

## Konfigurasi dan Parameter

### **Parameter Deteksi**
```python
FRAME_SKIP_DET = 0      # Skip frame untuk deteksi (0 = setiap frame)
FRAME_SKIP_OCR = 0      # Skip frame untuk OCR
TARGET_CLASS = ["license-plates"]  # Kelas yang dideteksi
```

### **ROI (Region of Interest)**
```python
DETECT_ROI = (514, 368, 1218, 837)  # (x1, y1, x2, y2) untuk video 1080p
```

### **Parameter Enhancement**
```python
target_min_side = 128    # Ukuran minimal sisi plat
max_scale = 2.0         # Maksimal scaling
do_unsharp = True       # Aktifkan unsharp masking
do_clahe = True         # Aktifkan CLAHE
```

---

## Optimasi Performa

### **1. Frame Skipping**
- Deteksi tidak perlu dilakukan setiap frame
- OCR bisa di-skip lebih sering karena lebih lambat

### **2. Batch Processing**
- OCR dilakukan dalam batch untuk efisiensi GPU
- Multiple plat diproses sekaligus

### **3. ROI Filtering**
- Hanya proses plat yang berada di area yang relevan
- Mengurangi false positive dari area luar

### **4. Model Optimization**
- YOLOv8n (nano) untuk kecepatan
- PaddlePaddle inference engine untuk efisiensi
- GPU acceleration jika tersedia

---

## Struktur File

```
moto_counting/
├── main.py                 # Pipeline utama
├── main_detector.py        # Modul deteksi YOLOv8
├── bytetrack.py           # Sistem tracking
├── label.names            # Nama kelas untuk YOLOv8
├── requirements.txt       # Dependencies
├── model/
│   ├── yolo/             # Model YOLOv8
│   │   └── yolov8n-fold5/
│   └── ocr/              # Model PaddleOCR
│       ├── PP-OCRv5_server_det_infer/
│       └── PP-OCRv5_server_rec_infer/
└── sample/               # Gambar contoh
```

---

## Cara Menjalankan

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Jalankan Sistem**
```bash
# Untuk video file
python main.py

# Untuk webcam (ubah VIDEO_PATH = 0 di main.py)
python main.py
```

### **3. Konfigurasi**
- Edit `VIDEO_PATH` untuk sumber video
- Sesuaikan `DETECT_ROI` dengan area yang diinginkan
- Atur `TARGET_CLASS` sesuai objek yang ingin dideteksi

---

## Output Sistem

### **Visual Output**
- **Bounding boxes** di sekitar plat yang terdeteksi
- **Teks OCR** hasil pembacaan plat
- **Counter** jumlah kendaraan masuk/keluar
- **ROI visualization** area deteksi

### **Console Output**
```
Frame 150: Det=45.2ms, Enh=12.3ms, OCR=89.1ms, Total=156.7ms
Plat terdeteksi: ['B1234XYZ', 'D5678ABC']
Motor masuk: 15, Motor keluar: 12
```

### **Performance Metrics**
- **Detection time**: Waktu deteksi per frame
- **OCR time**: Waktu OCR per batch
- **Total processing time**: Waktu total per frame
- **FPS**: Frame per second yang dicapai

---

## Troubleshooting

### **Masalah Umum**

1. **Model tidak ditemukan**
   - Pastikan path model benar
   - Download model YOLOv8 dan PaddleOCR

2. **Performa lambat**
   - Tingkatkan `FRAME_SKIP_DET` dan `FRAME_SKIP_OCR`
   - Gunakan GPU jika tersedia
   - Kurangi resolusi video

3. **OCR tidak akurat**
   - Sesuaikan parameter enhancement
   - Periksa kualitas gambar input
   - Atur ROI dengan lebih presisi, tips: akses [link berikut](https://g.co/gemini/share/c377971ac90c)

4. **Double counting**
   - Sesuaikan parameter ByteTrack
   - Periksa logika crossing line
   - Atur threshold tracking

---

## Pengembangan Lanjutan

### **Possible Improvements**
1. **Deep SORT** sebagai alternatif ByteTrack
2. **Multi-class detection** (mobil, motor, truk)
3. **Database integration** untuk logging
4. **Web interface** untuk monitoring
5. **Real-time alerts** untuk anomali
6. **Speed estimation** berdasarkan tracking
7. **License plate validation** dengan database

### **Model Upgrades**
1. **YOLOv8s/m/l** untuk akurasi lebih tinggi
2. **Custom training** dengan dataset lokal
3. **TensorRT optimization** untuk inference lebih cepat
4. **Quantization** untuk deployment edge device

Sistem ini dirancang untuk fleksibilitas dan dapat diadaptasi untuk berbagai skenario monitoring kendaraan dengan modifikasi parameter dan konfigurasi yang sesuai.
import cv2
import numpy as np
import os
import time
from paddleocr import PaddleOCR

from yolo_paddle import YOLOv8_Paddle_Detector

# ---------------- CONFIG ----------------
YOLO_MODEL_DIR = "../model/yolo/best_paddle_model/inference_model"
CLASS_NAMES_FILE = "../label.names.example"
OCR_DET_MODEL_DIR = "../model/ocr/PP-OCRv5_server_det_infer"  # tetap ditetapkan, tapi tak dipakai di runtime
OCR_REC_MODEL_DIR = "../model/ocr/PP-OCRv5_server_rec_infer"
INPUT_IMAGE_PATH = "../frame_22.jpg"
TARGET_CLASS = ["license-plates"]

DEBUG_VIS = False  # matikan semua imshow saat benchmarking

# ---------------- ENHANCE ----------------
# Cache CLAHE sekali
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance_plate(
        img_bgr: np.ndarray,
        target_min_side: int = 128,  # ↓ dari 160
        max_scale: float = 2.0,  # ↓ batasi scale
        do_unsharp: bool = True,
        do_clahe: bool = True,
        do_denoise: bool = False,  # OFF: NLM mahal
        border: int = 2
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if min(h, w) <= 0:
        return img_bgr

    # Jika sudah cukup besar, skip enhancement mahal
    if min(h, w) >= target_min_side:
        enhanced = img_bgr.copy()
    else:
        # Upscale adaptif
        scale = max(1.0, target_min_side / float(min(h, w)))
        scale = min(scale, max_scale)
        if abs(scale - 1.0) > 1e-3:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            enhanced = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            enhanced = img_bgr.copy()

    # Unsharp ringan
    if do_unsharp:
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0, sigmaY=1.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    # CLAHE (pakai objek cache)
    if do_clahe:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _CLAHE.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Denoise OFF (opsional)
    if do_denoise:
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 15)

    if border > 0:
        enhanced = cv2.copyMakeBorder(enhanced, border, border, border, border, cv2.BORDER_REPLICATE)

    return enhanced


# ---------------- INIT ----------------
print("Memuat model deteksi plat (YOLOv8)...")
t0 = time.perf_counter()
detector = YOLOv8_Paddle_Detector(
    YOLO_MODEL_DIR, CLASS_NAMES_FILE, use_gpu=True,
    # kalau wrapper mendukung, set parameter cepat:
    # conf=0.4, iou=0.5, max_det=5, img_size=640, fp16=True
)
t1 = time.perf_counter()
print(f"Model YOLOv8 dimuat: {(t1 - t0) * 1000:.1f} ms")

print("Memuat model OCR (recognition-only)...")
t2 = time.perf_counter()
ocr = PaddleOCR(
    # kita tetap sediakan dir, tapi runtime nanti det=False sehingga model det tidak dipakai
    text_detection_model_dir=OCR_DET_MODEL_DIR,
    text_recognition_model_dir=OCR_REC_MODEL_DIR,
    # percepat
    text_recognition_batch_size=8,  # naikkan jika VRAM cukup
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    # Jika pakai TensorRT nanti:
    # use_tensorrt=True, precision='fp16'
)
t3 = time.perf_counter()
print(f"Model OCR dimuat: {(t3 - t2) * 1000:.1f} ms")

# ---------------- RUN ----------------
if not os.path.exists(INPUT_IMAGE_PATH):
    print(f"Error: tidak ada file {INPUT_IMAGE_PATH}")
else:
    img = cv2.imread(INPUT_IMAGE_PATH)  # sekali baca
    if img is None:
        print("Gagal membaca gambar.")
    else:
        print(f"\nMemproses: {INPUT_IMAGE_PATH}")
        overall_start = time.perf_counter()

        # 1) Deteksi plat
        det_start = time.perf_counter()
        # Jika wrapper mendukung array:
        # cropped_plates = detector.detect_and_crop_objects(image=img, allowed_classes=TARGET_CLASS)
        cropped_plates = detector.detect_and_crop_objects(image_path=INPUT_IMAGE_PATH, allowed_classes=TARGET_CLASS)
        det_end = time.perf_counter()

        if not cropped_plates:
            print(f"Tidak ada plat. (Deteksi+Crop: {(det_end - det_start) * 1000:.1f} ms)")
        else:
            print(f"Ditemukan {len(cropped_plates)} plat. (Deteksi+Crop: {(det_end - det_start) * 1000:.1f} ms)")

            # 2) Enhancement (lebih ringan) + kumpulkan batch untuk OCR
            enh_times = []
            plates_enh = []
            for i, plate_img in enumerate(cropped_plates):
                if DEBUG_VIS: cv2.imshow(f"Crop #{i + 1}", plate_img)
                t_enh0 = time.perf_counter()
                enhanced = enhance_plate(plate_img)  # default ringan
                t_enh1 = time.perf_counter()
                enh_times.append((t_enh1 - t_enh0) * 1000.0)
                plates_enh.append(enhanced)
                if DEBUG_VIS: cv2.imshow(f"Enhanced #{i + 1}", enhanced)

            # 3) OCR **batch**, **tanpa deteksi teks** (det=False)
            t_ocr0 = time.perf_counter()
            # PaddleOCR menerima list of np.ndarray
            ocr_res = ocr.predict(plates_enh)
            t_ocr1 = time.perf_counter()
            ocr_ms = (t_ocr1 - t_ocr0) * 1000.0

            # 4) Tampilkan hasil ringkas
            print(f"OCR (batch, det=False): {ocr_ms:.1f} ms")
            for i, item in enumerate(ocr_res):
                # format output: list per gambar; tiap item biasanya [(text, score)] atau list kosong
                if item and len(item) > 0:
                    # Untuk rec-only, biasanya list of [text, score]
                    try:
                        txt, score = item[0]
                    except Exception:
                        txt = item.get('rec_texts', [""])[0]
                        score = item.get('rec_scores', [0.0])[0]
                    print(f'  Plat #{i + 1}: "{txt}" (conf={score:.4f})')
                else:
                    print(f"  Plat #{i + 1}: (kosong)")


            # Ringkasan waktu
            def _avg(x):
                return sum(x) / len(x) if x else 0.0


            print("\n--- Rangkuman Waktu ---")
            print(f"Deteksi+Crop : {(det_end - det_start) * 1000:.1f} ms")
            print(
                f"Enhancement  : rata2 {_avg(enh_times):.1f} ms (min {min(enh_times):.1f} / max {max(enh_times):.1f})" if enh_times else "Enhancement  : -")
            print(f"OCR (batch)  : {ocr_ms:.1f} ms")

            overall_end = time.perf_counter()
            print(f"\nWaktu total pipeline (tanpa load): {(overall_end - overall_start) * 1000:.1f} ms")

        if DEBUG_VIS:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

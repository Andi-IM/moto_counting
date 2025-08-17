#!/usr/bin/env python3
"""
Sistem Penghitungan Lalu Lintas Sepeda Motor dengan OCR Plat Nomor menggunakan YOLOv8 dan PaddleOCR

Versi 4.1 (PARALEL OCR + Robust Names + ROI Normalized):
- ROI normalized (0..1) agar selalu terlihat di berbagai resolusi.
- OCR plat nomor paralel (ThreadPoolExecutor), PaddleOCR lazy-per-thread.
- Pemetaan nama kelas YOLO robust (dict/list), OCR auto-off jika label plat tidak ada.
"""

import cv2
from ultralytics import YOLO
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# KONFIGURASI
# ============================================================================

# YOLO
MODEL_PATH = "../runs/detect/yolov8_roboflow_custom/weights/best_ncnn_model"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# Video
VIDEO_PATH = "../from_camera_480.mp4"

# ROI (pakai normalized coords 0..1 agar fleksibel)
# COUNTING_ROI_NORM = [(0.60, 0.50), (0.95, 0.95)]   # (x1,y1),(x2,y2) dalam rasio frame
COUNTING_ROI_NORM = [
    (0.43125, 0.28333),   # kiri-atas (x1, y1) dalam rasio
    (0.79844, 0.70625)    # kanan-bawah (x2, y2) dalam rasio
]
ROI_COLOR = (255, 255, 0)  # BGR: cyan

# Visual
LINE_THICKNESS = 2
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.8
TEXT_THICKNESS = 2
PLATE_TEXT_COLOR = (0, 255, 255)  # Kuning

# OCR parallelism
NUM_OCR_WORKERS = 2
OCR_MIN_INTERVAL_SEC = 1.5
MAX_CROP_SIDE = 512

# ============================================================================
# UTILITAS
# ============================================================================

def norm_roi_to_pixels(frame, roi_norm):
    h, w = frame.shape[:2]
    (nx1, ny1), (nx2, ny2) = roi_norm
    x1 = int(np.clip(nx1, 0, 1) * w)
    x2 = int(np.clip(nx2, 0, 1) * w)
    y1 = int(np.clip(ny1, 0, 1) * h)
    y2 = int(np.clip(ny2, 0, 1) * h)
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        # fallback kecil jika ROI invalid
        x1, y1, x2, y2 = 5, 5, min(100, w - 1), min(60, h - 1)
    return (x1, y1), (x2, y2)

def draw_roi(frame, roi_norm, color, thickness=LINE_THICKNESS):
    (x1, y1), (x2, y2) = norm_roi_to_pixels(frame, roi_norm)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, "ZONA HITUNG", (x1, max(0, y1 - 10)),
                TEXT_FONT, 0.7, color, thickness)

def is_inside_roi(point, roi_norm, frame):
    (x1, y1), (x2, y2) = norm_roi_to_pixels(frame, roi_norm)
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2

def get_center_point(box):
    x1, y1, x2, y2 = map(int, box)
    return (x1 + x2) // 2, (y1 + y2) // 2

def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    scale = min(MAX_CROP_SIDE / max(ch, cw), 1.0)
    if scale < 1.0:
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_LINEAR)
    return crop

# ============================================================================
# INISIALISASI MODEL
# ============================================================================

try:
    model = YOLO(MODEL_PATH)
    CLASS_NAMES = model.names  # bisa dict {id:name} atau list [name,...]
    print("Model YOLO berhasil dimuat.")
except Exception as e:
    print(f"Error memuat model YOLO: {e}")
    raise SystemExit(1)

def _get_class_id(names, targets):
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v).strip().lower() in targets:
                return int(k)
    else:
        for i, v in enumerate(names):
            if str(v).strip().lower() in targets:
                return int(i)
    return None

def _get_class_name(names, cid):
    if isinstance(names, dict):
        return str(names.get(int(cid), "")).strip()
    else:
        return str(names[int(cid)]).strip() if 0 <= int(cid) < len(names) else ""

PLATE_LABEL_CANDIDATES = {
    "license-plate", "licence-plate", "number-plate", "plate", "license_plate", "lp"
}

LICENSE_PLATE_CLASS_ID = _get_class_id(CLASS_NAMES, PLATE_LABEL_CANDIDATES)
OCR_ENABLED = LICENSE_PLATE_CLASS_ID is not None
if not OCR_ENABLED:
    print("[PERINGATAN] Kelas plat nomor tidak ditemukan pada model. OCR dinonaktifkan.")

cap = cv2.VideoCapture(VIDEO_PATH)

# ============================================================================
# STATE PENGHITUNGAN & OCR
# ============================================================================

motor_masuk = 0
motor_keluar = 0
counted_objects = set()

plate_texts = {}
last_ocr_time = {}
ocr_futures = {}

state_lock = threading.Lock()

# ============================================================================
# WORKER OCR (PARALEL) — LAZY IMPORT PADDLEOCR
# ============================================================================

_thread_local = threading.local()
OCR_HARD_DISABLED = False  # jadi True jika init PaddleOCR gagal

def _get_ocr_instance():
    global OCR_HARD_DISABLED
    if OCR_HARD_DISABLED:
        return None
    if not hasattr(_thread_local, "ocr"):
        try:
            # Import di sini untuk menghindari circular/partial init
            from paddleocr import PaddleOCR
            _thread_local.ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )
        except Exception as e:
            print(f"[INIT OCR] Gagal inisialisasi PaddleOCR di thread: {e}")
            print("-> OCR dimatikan. Pastikan tidak ada file bernama 'paddle.py' "
                  "di project dan paket 'paddlepaddle' terpasang kompatibel.")
            OCR_HARD_DISABLED = True
            return None
    return _thread_local.ocr

def run_ocr_task(track_id, crop_bgr):
    try:
        if crop_bgr is None or crop_bgr.size == 0:
            return track_id, None
        ocr = _get_ocr_instance()
        if ocr is None:
            return track_id, None
        res = ocr.predict(input=crop_bgr)
        text = None
        if res and res[0]:
            best = max(res[0], key=lambda it: it[1][1])
            text = best[1][0]
        return track_id, text
    except Exception as e:
        print(f"[OCR Worker] Error pada track {track_id}: {e}")
        return track_id, None

executor = ThreadPoolExecutor(max_workers=NUM_OCR_WORKERS)

# ============================================================================
# LOOP UTAMA
# ============================================================================

print("\nMemulai sistem (q untuk keluar)…\n")

try:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Akhir video atau gagal membaca frame.")
            break

        # Deteksi + tracking
        results = model.track(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, persist=True)
        annotated_frame = results[0].plot()

        # ROI: gambar dan pakai untuk cek posisi
        draw_roi(annotated_frame, COUNTING_ROI_NORM, ROI_COLOR)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            # Index plat nomor di frame ini (jika OCR aktif)
            current_frame_plates = {}
            if OCR_ENABLED:
                current_frame_plates = {
                    tid: box
                    for box, tid, cid in zip(boxes, track_ids, class_ids)
                    if int(cid) == int(LICENSE_PLATE_CLASS_ID)
                }

            # Hitung & antrikan OCR
            for box, tid, cid in zip(boxes, track_ids, class_ids):
                center = get_center_point(box)
                class_name = _get_class_name(CLASS_NAMES, cid).lower()

                # Penghitungan dua arah di dalam ROI (sekali per objek)
                if is_inside_roi(center, COUNTING_ROI_NORM, annotated_frame) and tid not in counted_objects:
                    if class_name in ['lamp-front', 'lamp-rear']:
                        if class_name == 'lamp-front':
                            motor_masuk += 1
                            print(f"🟢 MASUK (ID {tid}), total: {motor_masuk}")
                        else:
                            motor_keluar += 1
                            print(f"🔴 KELUAR (ID {tid}), total: {motor_keluar}")
                        counted_objects.add(tid)

                # Antrikan OCR jika aktif & ada bbox plat untuk track ini
                if OCR_ENABLED and (tid in current_frame_plates):
                    now = time.time()
                    last_t = last_ocr_time.get(tid, 0.0)
                    has_running = tid in ocr_futures and not ocr_futures[tid].done()
                    if (not has_running) and (now - last_t >= OCR_MIN_INTERVAL_SEC):
                        x1, y1, x2, y2 = map(int, current_frame_plates[tid])
                        crop = safe_crop(frame, x1, y1, x2, y2)
                        if crop is not None:
                            future = executor.submit(run_ocr_task, tid, crop.copy())
                            ocr_futures[tid] = future
                            last_ocr_time[tid] = now

            # Ambil hasil OCR yang selesai (non-blocking)
            done_tids = []
            for tid, fut in list(ocr_futures.items()):
                if fut.done():
                    t_id, text = fut.result()
                    with state_lock:
                        if text:
                            plate_texts[t_id] = text
                            print(f"   -> OCR OK (ID {t_id}): {text}")
                    done_tids.append(tid)
            for tid in done_tids:
                ocr_futures.pop(tid, None)

            # Gambar teks OCR di atas bbox objek (kalau sudah ada hasilnya)
            for box, tid in zip(boxes, track_ids):
                if tid in plate_texts:
                    x1, y1, _, _ = map(int, box)
                    cv2.putText(annotated_frame, plate_texts[tid],
                                (x1, y1 - 10), TEXT_FONT, TEXT_SCALE, PLATE_TEXT_COLOR, TEXT_THICKNESS)

        # Overlay statistik
        cv2.putText(annotated_frame, f"Motor Masuk: {motor_masuk}", (20, 40),
                    TEXT_FONT, 1.2, (0, 255, 0), TEXT_THICKNESS)
        cv2.putText(annotated_frame, f"Motor Keluar: {motor_keluar}", (20, 85),
                    TEXT_FONT, 1.2, (0, 0, 255), TEXT_THICKNESS)

        cv2.imshow("YOLOv8 + OCR Paralel", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=True)

    print("\n" + "=" * 50)
    print("HASIL PENGHITUNGAN AKHIR")
    print("=" * 50)
    print(f"Total Motor Masuk : {motor_masuk}")
    print(f"Total Motor Keluar: {motor_keluar}")
    print("\nDaftar Plat Nomor yang Terbaca:")
    for track_id, plate in plate_texts.items():
        print(f"  - ID {track_id}: {plate}")
    print(f"\nTotal Kendaraan Dihitung: {len(counted_objects)}")
    print("Program berhenti dengan aman.")
    print("=" * 50)

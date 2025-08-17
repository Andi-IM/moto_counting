#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import numpy as np
from paddleocr import PaddleOCR

from yolo_paddle import YOLOv8_Paddle_Detector

# =============== CONFIG ===============
# Model paths
YOLO_MODEL_DIR = "../model/yolo/best_paddle_model/inference_model"
CLASS_NAMES_FILE = "../label.names"
OCR_DET_MODEL_DIR = "../model/ocr/PP-OCRv5_server_det_infer"  # tetap diset; runtime rec-only
OCR_REC_MODEL_DIR = "../model/ocr/PP-OCRv5_server_rec_infer"

# I/O
VIDEO_PATH = "../from_camera_480.mp4"  # ganti ke path video, atau set 0 utk webcam

# Target class
TARGET_CLASS = ["license-plates"]

# Benchmark & display
DEBUG_VIS = True  # matikan semua imshow saat benchmarking
PRINT_EVERY_N_FRAMES = 30  # interval cetak ringkasan
MAX_MS_PER_FRAME_WARN = 800  # warning bila frame > 800 ms (target ideal < 500 ms)

# Deteksi / OCR stride (untuk hemat komputasi)
FRAME_SKIP_DET = 0  # 0 = deteksi tiap frame; 1 = deteksi tiap 2 frame, dst.
FRAME_SKIP_OCR = 0  # 0 = OCR tiap frame; naikan jika perlu

# ----------- BIDANG DETEKSI (ROI KHUSUS DETEKSI+OCR) -----------
# Hanya area ini yang akan diproses untuk DETEKSI dan OCR.
# Koordinat piksel ABSOLUT pada frame (x1, y1), (x2, y2)
DETECT_ROI = (152, 147, 531, 388)

# Enhancement params (ringan)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance_plate(
        img_bgr: np.ndarray,
        target_min_side: int = 128,
        max_scale: float = 2.0,
        do_unsharp: bool = True,
        do_clahe: bool = True,
        do_denoise: bool = False,
        border: int = 2
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if min(h, w) <= 0: return img_bgr

    # upscale adaptif
    if min(h, w) < target_min_side:
        scale = min(max(1.0, target_min_side / float(min(h, w))), max_scale)
        if abs(scale - 1.0) > 1e-3:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if do_unsharp:
        blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=1.0, sigmaY=1.0)
        img_bgr = cv2.addWeighted(img_bgr, 1.5, blur, -0.5, 0)

    if do_clahe:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _CLAHE.apply(l)
        img_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    if do_denoise:
        img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 3, 3, 7, 15)

    if border > 0:
        img_bgr = cv2.copyMakeBorder(img_bgr, border, border, border, border, cv2.BORDER_REPLICATE)
    return img_bgr


def parse_ocr_item(item):
    """
    PaddleOCR (pipeline .predict) biasanya mengembalikan list per gambar:
      - format umum: [[text, score], ...] -> kita ambil kandidat #1
      - atau dict: {'rec_texts': [...], 'rec_scores': [...]}
    """
    if item is None: return "", 0.0
    if isinstance(item, list) and len(item) > 0:
        head = item[0]
        if isinstance(head, (list, tuple)) and len(head) >= 2:
            return str(head[0]), float(head[1])
        # fallback bila list of str
        if isinstance(head, str):
            return head, 0.0
    if isinstance(item, dict):
        txts = item.get('rec_texts', [])
        scs = item.get('rec_scores', [])
        if len(txts) > 0:
            return str(txts[0]), float(scs[0] if len(scs) > 0 else 0.0)
    return "", 0.0


def _scale_roi_to_frame(roi_xyxy, orig_wh, proc_wh):
    """Skalakan ROI jika frame di-resize (agar kotak deteksi tetap tepat)."""
    x1, y1, x2, y2 = roi_xyxy
    (ow, oh) = orig_wh
    (pw, ph) = proc_wh
    sx = pw / float(ow)
    sy = ph / float(oh)
    rx1 = int(round(x1 * sx))
    ry1 = int(round(y1 * sy))
    rx2 = int(round(x2 * sx))
    ry2 = int(round(y2 * sy))
    # clamp ke batas gambar
    rx1 = max(0, min(pw - 1, rx1))
    ry1 = max(0, min(ph - 1, ry1))
    rx2 = max(0, min(pw, rx2))
    ry2 = max(0, min(ph, ry2))
    return rx1, ry1, rx2, ry2


def main():
    # --------- INIT MODELS (sebelum video) ---------
    print("Memuat model deteksi plat (YOLOv8)...")
    t0 = time.perf_counter()
    detector = YOLOv8_Paddle_Detector(
        YOLO_MODEL_DIR, CLASS_NAMES_FILE, use_gpu=True,
        # conf=0.4, iou=0.5, max_det=5, img_size=640, fp16=True
    )
    t1 = time.perf_counter()
    print(f"Model YOLOv8 dimuat: {(t1 - t0) * 1000:.1f} ms")

    print("Memuat model OCR (recognition-only) + warmup...")
    t2 = time.perf_counter()
    ocr = PaddleOCR(
        text_detection_model_dir=OCR_DET_MODEL_DIR,  # disediakan; runtime fokus rec
        text_recognition_model_dir=OCR_REC_MODEL_DIR,
        text_recognition_batch_size=8,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    # Warmup (agar load & kompilasi kernel selesai sebelum video dimulai)
    dummy = np.full((32, 128, 3), 255, dtype=np.uint8)
    _ = ocr.predict([dummy])
    t3 = time.perf_counter()
    print(f"Model OCR siap (load+warmup): {(t3 - t2) * 1000:.1f} ms")

    # --------- OPEN VIDEO ---------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: tidak bisa membuka video: {VIDEO_PATH}")
        return

    # Optional: target size (resize frame untuk kecepatan, tanpa ubah aspek)
    TARGET_LONG_SIDE = None  # contoh: 960 atau 720; None = asli

    frame_idx = 0
    t_pipeline_start = time.perf_counter()
    per_frame_times = []  # ms
    det_times = []  # ms
    enh_times_all = []  # ms rata2 per frame
    ocr_times = []  # ms

    print("\nMulai memproses video...\n(ESC/Q untuk berhenti saat DEBUG_VIS=True)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t_frame0 = time.perf_counter()

        # optional resize untuk deteksi lebih cepat
        h, w = frame.shape[:2]
        frame_proc = frame
        if TARGET_LONG_SIDE is not None:
            long_side = max(h, w)
            if long_side > TARGET_LONG_SIDE:
                scale = TARGET_LONG_SIDE / float(long_side)
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                frame_proc = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ph, pw = frame_proc.shape[:2]

        # ---- ROI DETEKSI (skalakan jika frame di-resize) ----
        rx1, ry1, rx2, ry2 = _scale_roi_to_frame(DETECT_ROI, (w, h), (pw, ph))
        # jaga ROI valid
        if rx2 <= rx1 or ry2 <= ry1:
            rx1, ry1, rx2, ry2 = 0, 0, pw, ph  # fallback: seluruh frame

        # ---- DETEKSI (skip sesuai stride) di seluruh frame ----
        do_det = (frame_idx % (FRAME_SKIP_DET + 1) == 0)
        all_cropped_plates = []
        all_valid_boxes = []
        roi_cropped_plates = []
        roi_intersecting_boxes = []
        t_det_ms = 0.0
        if do_det:
            t_det0 = time.perf_counter()
            try:
                # Deteksi di seluruh frame
                all_cropped_plates, all_valid_boxes = detector.detect_and_crop_objects(allowed_classes=TARGET_CLASS, image_source=frame_proc)
            except TypeError:
                tmp_path = "_temp_frame_full.jpg"
                cv2.imwrite(tmp_path, frame_proc)
                all_cropped_plates, all_valid_boxes = detector.detect_and_crop_objects(
                    image_source=tmp_path, allowed_classes=TARGET_CLASS
                )
                os.remove(tmp_path)
            
            # Filter bounding box yang bersinggungan dengan ROI dan crop plat untuk OCR
            for i, box in enumerate(all_valid_boxes):
                x1, y1, x2, y2 = box
                # Cek apakah bounding box bersinggungan dengan ROI
                if not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2):
                    roi_intersecting_boxes.append(box)
                    roi_cropped_plates.append(all_cropped_plates[i])
            
            t_det1 = time.perf_counter()
            t_det_ms = (t_det1 - t_det0) * 1000.0

        # ---- ENHANCEMENT + OCR (skip sesuai stride) ----
        plate_texts = []
        t_ocr_ms = 0.0
        enh_times = []

        do_ocr = (frame_idx % (FRAME_SKIP_OCR + 1) == 0)

        if do_det and do_ocr and roi_cropped_plates:
            # Enhancement ringkas per crop (HASIL DETEKSI license-plates yang bersinggungan dengan ROI)
            plates_enh = []
            for p in roi_cropped_plates:
                t_enh0 = time.perf_counter()
                pe = enhance_plate(p)
                t_enh1 = time.perf_counter()
                enh_times.append((t_enh1 - t_enh0) * 1000.0)
                plates_enh.append(pe)

            # Batch OCR (rec-only)
            t_ocr0 = time.perf_counter()
            ocr_res = ocr.predict(plates_enh)
            t_ocr1 = time.perf_counter()
            t_ocr_ms = (t_ocr1 - t_ocr0) * 1000.0

            # Ambil kandidat #1 per crop
            for item in ocr_res:
                txt, score = parse_ocr_item(item)
                plate_texts.append((txt, score))

        # ---- RINGKASAN WAKTU ----
        t_frame1 = time.perf_counter()
        t_frame_ms = (t_frame1 - t_frame0) * 1000.0
        per_frame_times.append(t_frame_ms)
        if do_det: det_times.append(t_det_ms)
        if enh_times: enh_times_all.append(sum(enh_times) / max(1, len(enh_times)))
        if t_ocr_ms > 0: ocr_times.append(t_ocr_ms)

        if (frame_idx % PRINT_EVERY_N_FRAMES) == 0:
            def _avg(x): return (sum(x) / len(x)) if x else 0.0

            print(f"[Frame {frame_idx}] Total {t_frame_ms:.1f} ms | "
                  f"Det {_avg(det_times):.1f} ms | "
                  f"Enh {_avg(enh_times_all):.1f} ms | "
                  f"OCR {_avg(ocr_times):.1f} ms | "
                  f"All Plates: {len(all_cropped_plates)} | ROI Plates: {len(roi_cropped_plates)}")
        if t_frame_ms > MAX_MS_PER_FRAME_WARN:
            print(f"  ⚠️  Frame {frame_idx}: {t_frame_ms:.1f} ms (> {MAX_MS_PER_FRAME_WARN} ms)")

        # ---- OPTIONAL VISUALIZATION ----
        if DEBUG_VIS:
            # Gambar bidang deteksi (ROI) agar terlihat area bidikan
            vis = frame_proc.copy()
            overlay = vis.copy()
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 255, 255), -1)  # isi kuning transparan
            alpha = 0.15
            vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 200, 255), 2)
            cv2.putText(vis, "Detection+OCR area", (rx1 + 6, max(ry1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            # Gambar bounding box untuk setiap plat yang bersinggungan dengan ROI
            if do_det and roi_intersecting_boxes:
                for box in roi_intersecting_boxes:
                    x1, y1, x2, y2 = box
                    
                    # Gambar bounding box dengan warna hijau
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Tambahkan label "PLATE" di atas bounding box
                    cv2.putText(vis, "PLATE", (x1, max(y1 - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Tampilkan hasil OCR pada pojok kiri atas
            if do_det and cropped_plates and plate_texts:
                y0 = 30
                for j, (txt, sc) in enumerate(plate_texts):
                    cv2.putText(vis, f"{j + 1}: {txt} ({sc:.2f})", (10, y0 + 24 * j),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("video", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):  # ESC/Q to quit
                break

    t_pipeline_end = time.perf_counter()
    cap.release()
    if DEBUG_VIS:
        cv2.destroyAllWindows()

    # --------- RANGKUMAN AKHIR ---------
    def _avg(x):
        return (sum(x) / len(x)) if x else 0.0

    total_ms = (t_pipeline_end - t_pipeline_start) * 1000.0
    n = len(per_frame_times)
    print("\n===== Rangkuman =====")
    print(f"Total waktu video : {total_ms:.1f} ms")
    print(f"Jumlah frame      : {n}")
    print(f"Rata-rata/frame   : {_avg(per_frame_times):.1f} ms "
          f"(~{1000.0 / max(1e-6, _avg(per_frame_times)):.2f} FPS)")
    print(f"Avg Deteksi       : {_avg(det_times):.1f} ms")
    print(f"Avg Enhancement   : {_avg(enh_times_all):.1f} ms")
    print(f"Avg OCR (batch)   : {_avg(ocr_times):.1f} ms")
    print("Catatan: target < 500–800 ms per frame. Naikkan batch OCR, aktifkan FP16/TRT bila tersedia, "
          "dan gunakan FRAME_SKIP_* bila diperlukan.")


if __name__ == "__main__":
    main()

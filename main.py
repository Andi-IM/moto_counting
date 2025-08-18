#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py — Deteksi + Tracking + Counting + Tampilan Hasil OCR Terintegrasi

Aturan hitung:
- Trail dari 'di LUAR' -> 'di DALAM' ROI  => dihitung KELUAR
- Trail dari 'di DALAM' -> 'di LUAR' ROI  => dihitung MASUK

MODIFIKASI:
- Panel debug di kanan atas diganti dengan tampilan hasil OCR terbaru.
- Setiap kali ada event hitung, OCR dijalankan dan hasilnya ditampilkan
  secara persisten di panel tersebut hingga ada hasil baru.
"""

import os
import time
from typing import Dict, Tuple, List, Optional
from collections import deque

import cv2
import numpy as np
from paddleocr import PaddleOCR

# Diasumsikan file bytetrack dan main_detector ada di direktori yang sama
from bytetrack import ByteTrack
from main_detector import YOLOv8_Paddle_Detector

# ===================== CONFIG =====================
CLASS_NAMES_FILE = "label.names"
YOLO_MODEL_DIR = "model/yolo/yolov8n-old3/best_paddle_model/inference_model"
OCR_DET_MODEL_DIR = "model/ocr/PP-OCRv5_server_det_infer"
OCR_REC_MODEL_DIR = "model/ocr/PP-OCRv5_server_rec_infer"
VIDEO_PATH = "sample/video_6178.mp4"
TARGET_CLASS = {"license-plates"}
DETECT_ROI = (514, 368, 1218, 837)

# Visual
DRAW_TRAILS = True
TRAIL_MAXLEN = 40
TRAIL_THICKNESS = 2
RESIZE_WIDTH = None
RESIZE_HEIGHT = None
FRAME_SKIP_DET = 0

# BARU: Konfigurasi untuk panel hasil OCR (pengganti panel debug)
OCR_PANEL_W = 280  # Lebar panel
OCR_PANEL_H = 180  # Tinggi panel
OCR_PANEL_MARGIN = 10
OCR_PANEL_BG_COLOR = (30, 30, 30)
OCR_PANEL_BORDER_COLOR = (0, 200, 255)


# ==================================================


# =============== Util Hitung Berbasis Trail ===============
def _point_in_rect(pt: Tuple[float, float], roi: Tuple[int, int, int, int]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = roi
    return (x1 <= x <= x2) and (y1 <= y <= y2)


class TrailCrossCounter:
    """Counter trail vs ROI."""

    def __init__(self, roi_xyxy: Tuple[int, int, int, int], cooldown_frames: int = 8):
        self.roi = roi_xyxy
        self._inside: Dict[int, bool] = {}
        self._last_count_frame: Dict[int, int] = {}
        self.cooldown_frames = cooldown_frames
        self.total_masuk = 0
        self.total_keluar = 0

    def update_from_trails(self, trails: Dict[int, List[Tuple[float, float]]], frame_idx: int):
        events = []
        for tid, pts in trails.items():
            if len(pts) < 2:
                continue

            if tid in self._last_count_frame and (frame_idx - self._last_count_frame[tid]) < self.cooldown_frames:
                self._inside[tid] = _point_in_rect(pts[-1], self.roi)
                continue

            prev_inside = self._inside.get(tid, _point_in_rect(pts[-2], self.roi))
            curr_inside = _point_in_rect(pts[-1], self.roi)

            if not prev_inside and curr_inside:
                self.total_keluar += 1
                self._last_count_frame[tid] = frame_idx
                events.append(("keluar", tid))
            elif prev_inside and not curr_inside:
                self.total_masuk += 1
                self._last_count_frame[tid] = frame_idx
                events.append(("masuk", tid))

            self._inside[tid] = curr_inside
        return events

    def get_counts(self) -> Tuple[int, int]:
        return self.total_masuk, self.total_keluar


# =============== Helper ===============
def _load_class_names(path: str) -> List[str]:
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _filter_by_allowed(names: List[str], allowed: set, class_ids: np.ndarray) -> np.ndarray:
    if not names: return np.ones_like(class_ids, dtype=bool)
    allowed_idx = {i for i, nm in enumerate(names) if nm in allowed}
    return np.array([cid in allowed_idx for cid in class_ids], dtype=bool)


def _detect_plates(detector, frame_bgr, class_names, allowed_names):
    boxes, scores, class_ids = detector.detect(frame_bgr)
    if boxes is None:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,))
    mask = _filter_by_allowed(class_names, allowed_names, class_ids)
    return boxes[mask], scores[mask], class_ids[mask]


def _safe_crop(src: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = src.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1: return np.array([])
    return src[y1:y2, x1:x2]


# MODIFIKASI: Fungsi ini sekarang untuk memformat gambar ke panel OCR, bukan thumbnail umum
def _format_crop_for_panel(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    if img.size == 0: return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 30, dtype=np.uint8)
    x_off, y_off = (target_w - nw) // 2, (target_h - nh) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas


def _run_ocr_on_crop(ocr_engine: PaddleOCR, crop_img: np.ndarray) -> str:
    if crop_img.size == 0: return ""
    result = ocr_engine.predict(crop_img)
    if result and len(result) > 0:
        try:
            item = result[0]
            # Format output: [[('text', confidence)]]
            text = item.get('rec_texts', [""])[0]
            score = item.get('rec_scores', [0.0])[0]
            return f'"{text}" (conf={score:.4f})'
        except (ValueError, TypeError, IndexError):
            return "OCR GAGAL"
    return "TIDAK TERBACA"


# BARU: Fungsi untuk menggambar panel hasil OCR di frame utama
def _draw_ocr_panel(vis: np.ndarray, last_ocr_crop: Optional[np.ndarray], last_ocr_text: str):
    H, W = vis.shape[:2]
    panel_x = W - OCR_PANEL_MARGIN - OCR_PANEL_W
    panel_y = OCR_PANEL_MARGIN

    # Area untuk gambar plat (sekitar 60% tinggi panel)
    img_area_h = int(OCR_PANEL_H * 0.6)
    # Area untuk teks
    text_area_h = OCR_PANEL_H - img_area_h

    # Gambar background dan border
    cv2.rectangle(vis, (panel_x - 2, panel_y - 2),
                  (panel_x + OCR_PANEL_W + 2, panel_y + OCR_PANEL_H + 2),
                  OCR_PANEL_BORDER_COLOR, 1)
    cv2.rectangle(vis, (panel_x, panel_y),
                  (panel_x + OCR_PANEL_W, panel_y + OCR_PANEL_H),
                  OCR_PANEL_BG_COLOR, -1)

    # Tampilkan gambar plat jika ada
    if last_ocr_crop is not None:
        formatted_crop = _format_crop_for_panel(last_ocr_crop, OCR_PANEL_W, img_area_h)
        vis[panel_y: panel_y + img_area_h, panel_x: panel_x + OCR_PANEL_W] = formatted_crop

    # Tampilkan teks hasil OCR
    if last_ocr_text:
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        # Sesuaikan ukuran font agar pas
        while True:
            (text_w, text_h), _ = cv2.getTextSize(last_ocr_text, font_face, font_scale, thickness)
            if text_w < OCR_PANEL_W - 10 and font_scale > 0.4:
                break
            font_scale -= 0.05

        text_x = panel_x + (OCR_PANEL_W - text_w) // 2
        text_y = panel_y + img_area_h + (text_area_h + text_h) // 2
        cv2.putText(vis, last_ocr_text, (text_x, text_y), font_face, font_scale, (50, 255, 50), thickness)


def main():
    class_names = _load_class_names(CLASS_NAMES_FILE)
    detector = YOLOv8_Paddle_Detector(
        model_dir=YOLO_MODEL_DIR, class_names_file=CLASS_NAMES_FILE,
        use_gpu=True, device_id=0, conf_threshold=0.4, nms_threshold=0.5
    )
    ocr = PaddleOCR(
        text_detection_model_dir=OCR_DET_MODEL_DIR,
        text_recognition_model_dir=OCR_REC_MODEL_DIR,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    tracker = ByteTrack(
        high_conf_thresh=0.6, low_conf_thresh=0.1, iou_threshold=0.3,
        max_age=30, min_hits=3
    )
    trail_counter = TrailCrossCounter(roi_xyxy=DETECT_ROI)

    # MODIFIKASI: Variabel untuk menyimpan hasil OCR terakhir
    last_ocr_crop: Optional[np.ndarray] = None
    last_ocr_text: str = "Menunggu deteksi..."

    cap = cv2.VideoCapture(0 if str(VIDEO_PATH) == "0" else VIDEO_PATH)
    if not cap.isOpened():
        print(f"Gagal membuka video: {VIDEO_PATH}")
        return

    frame_idx = 0
    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret: break
        if RESIZE_WIDTH or RESIZE_HEIGHT:
            frame = cv2.resize(frame, (RESIZE_WIDTH or frame.shape[1], RESIZE_HEIGHT or frame.shape[0]))

        vis = frame.copy()
        do_det = (frame_idx % (FRAME_SKIP_DET + 1) == 0)

        boxes, scores, class_ids = (_detect_plates(detector, frame, class_names, TARGET_CLASS) if do_det
                                    else (np.empty((0, 4)), np.empty((0,)), np.empty((0,))))

        tracks = tracker.update(boxes, scores, class_ids)
        bbox_by_tid = {int(tid): (int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2, tid, _, _ in tracks}

        all_trails = tracker.get_all_trails()
        events = trail_counter.update_from_trails(all_trails, frame_idx)

        # MODIFIKASI: Proses event dan update hasil OCR terbaru
        for ev_type, tid in events:
            if tid in bbox_by_tid:
                x1, y1, x2, y2 = bbox_by_tid[tid]
                crop = _safe_crop(frame, x1, y1, x2, y2)
                if crop.size > 0:
                    # Jalankan OCR dan simpan hasilnya untuk ditampilkan
                    last_ocr_text = _run_ocr_on_crop(ocr, crop)
                    last_ocr_crop = crop.copy()
                    print(f"Deteksi Baru (ID: {tid}, Tipe: {ev_type}): {last_ocr_text}")  # Log ke konsol

        # ======== Visualisasi ========
        # ROI
        rx1, ry1, rx2, ry2 = DETECT_ROI
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 200, 255), 2)
        # Bbox, Trails, HUD (disingkat)
        for x1, y1, x2, y2, tid, score, cid in tracks:
            x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
            cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            label = f"#{int(tid)}"
            cv2.putText(vis, label, (x1i, y1i - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
        if DRAW_TRAILS:
            for pts in all_trails.values():
                if len(pts) > 1:
                    cv2.polylines(vis, [np.array(pts, dtype=np.int32)], isClosed=False, color=(255, 255, 0),
                                  thickness=TRAIL_THICKNESS)
        masuk, keluar = trail_counter.get_counts()
        cv2.putText(vis, f"Masuk : {masuk}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, f"Keluar: {keluar}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # MODIFIKASI: Render panel OCR di setiap frame
        _draw_ocr_panel(vis, last_ocr_crop, last_ocr_text)

        # FPS
        curr_time = time.perf_counter()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Deteksi Plat Nomor dengan Hasil OCR", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
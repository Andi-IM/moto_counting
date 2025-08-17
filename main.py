#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py — Deteksi + Tracking (ByteTrack) + Trail-based Counting + Debug Crops

Aturan hitung:
- Trail dari 'di LUAR' -> 'di DALAM' ROI  => dihitung KELUAR
- Trail dari 'di DALAM' -> 'di LUAR' ROI  => dihitung MASUK

Tambahan (DEBUG):
- Setiap kali ada event count (MASUK/KELUAR), ambil crop bbox track saat itu
  dan tampilkan sebagai thumbnail di kanan-atas.
"""

import os
import time
from typing import Dict, Tuple, List
from collections import deque

import cv2
import numpy as np

from bytetrack import ByteTrack
from main_detector import YOLOv8_Paddle_Detector

# ===================== CONFIG =====================
YOLO_MODEL_DIR = "model/yolo/yolov8n-fold5/best_paddle_model/inference_model"
CLASS_NAMES_FILE = "label.names"
VIDEO_PATH = "video_6178.mp4"  # ganti sesuai kebutuhan, atau 0 untuk webcam
TARGET_CLASS = {"license-plates", "license-plate", "plate", "lp"}

# ROI hitung (x1, y1, x2, y2)
DETECT_ROI = (514, 368, 1218, 837)

# Visual
DRAW_TRAILS = True
TRAIL_MAXLEN = 40
TRAIL_THICKNESS = 2
RESIZE_WIDTH = None
RESIZE_HEIGHT = None
FRAME_SKIP_DET = 0

# Debug-crop panel (kanan-atas)
DBG_THUMB_W = 180         # lebar thumbnail
DBG_THUMB_H = 110         # tinggi thumbnail
DBG_THUMB_GAP = 8         # jarak antar thumb
DBG_THUMB_MAX = 6         # jumlah thumb disimpan/tampil
DBG_PANEL_MARGIN = 10     # margin dari tepi
DBG_BG_COLOR = (30, 30, 30)  # BGR untuk background panel
DBG_BORDER_COLOR = (0, 200, 255)
DBG_TEXT_COLOR = (255, 255, 255)
# ==================================================


# =============== Util Hitung Berbasis Trail ===============
def _point_in_rect(pt: Tuple[float, float], roi: Tuple[int, int, int, int]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = roi
    return (x1 <= x <= x2) and (y1 <= y <= y2)


class TrailCrossCounter:
    """
    Counter trail vs ROI.
    - (outside -> inside)  => KELUAR++
    - (inside  -> outside) => MASUK++

    Return event list agar caller bisa melakukan aksi (misal crop debug).
    """
    def __init__(self, roi_xyxy: Tuple[int, int, int, int], cooldown_frames: int = 8):
        self.roi = roi_xyxy
        self._inside: Dict[int, bool] = {}
        self._last_count_frame: Dict[int, int] = {}
        self.cooldown_frames = cooldown_frames
        self.total_masuk = 0
        self.total_keluar = 0

    def update_from_trails(self, trails: Dict[int, List[Tuple[float, float]]], frame_idx: int):
        """
        Update state dan kembalikan daftar event crossing:
        [('masuk', track_id), ('keluar', track_id), ...]
        """
        events = []
        for tid, pts in trails.items():
            if len(pts) < 2:
                continue

            # anti double-count beruntun
            if tid in self._last_count_frame and (frame_idx - self._last_count_frame[tid]) < self.cooldown_frames:
                prev_inside = self._inside.get(tid, _point_in_rect(pts[-2], self.roi))
                curr_inside = _point_in_rect(pts[-1], self.roi)
                self._inside[tid] = curr_inside
                continue

            prev_inside = self._inside.get(tid, _point_in_rect(pts[-2], self.roi))
            curr_inside = _point_in_rect(pts[-1], self.roi)

            if (not prev_inside) and curr_inside:
                # trail MASUK ROI => hitung KELUAR
                self.total_keluar += 1
                self._last_count_frame[tid] = frame_idx
                events.append(("keluar", tid))

            elif prev_inside and (not curr_inside):
                # trail KELUAR ROI => hitung MASUK
                self.total_masuk += 1
                self._last_count_frame[tid] = frame_idx
                events.append(("masuk", tid))

            self._inside[tid] = curr_inside

        return events

    def get_counts(self) -> Tuple[int, int]:
        return self.total_masuk, self.total_keluar


# =============== Helper ===============
def _load_class_names(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _filter_by_allowed(names: List[str], allowed: set, class_ids: np.ndarray) -> np.ndarray:
    if not names:
        return np.ones_like(class_ids, dtype=bool)
    allowed_idx = {i for i, nm in enumerate(names) if nm in allowed}
    return np.array([cid in allowed_idx for cid in class_ids], dtype=bool)


def _detect_plates(detector: YOLOv8_Paddle_Detector,
                   frame_bgr: np.ndarray,
                   class_names: List[str],
                   allowed_names: set) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Prefer API .detect
    if hasattr(detector, "detect"):
        boxes, scores, class_ids = detector.detect(frame_bgr)
        if boxes is None:
            return np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32)
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        class_ids = np.asarray(class_ids, dtype=np.int32)
        mask = _filter_by_allowed(class_names, allowed_names, class_ids)
        return boxes[mask], scores[mask], class_ids[mask]

    # Fallback ke .detect_and_crop_objects
    if hasattr(detector, "detect_and_crop_objects"):
        crops, valid_boxes = detector.detect_and_crop_objects(
            allowed_classes=list(allowed_names) if allowed_names else None,
            image_source=frame_bgr
        )
        if valid_boxes is None:
            return np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32)
        boxes = np.asarray(valid_boxes, dtype=np.float32)
        scores = np.ones((boxes.shape[0],), np.float32)
        class_ids = np.zeros((boxes.shape[0],), np.int32)
        return boxes, scores, class_ids

    return np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32)


def _safe_crop(src: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop dengan clamping supaya tidak error jika di tepi frame."""
    h, w = src.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((DBG_THUMB_H, DBG_THUMB_W, 3), dtype=np.uint8)
    return src[y1:y2, x1:x2]


def _to_thumb(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Resize/pad crop ke ukuran thumbnail fixed (untuk display rapi)."""
    if img.size == 0:
        return np.zeros((th, tw, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    scale = min(tw / max(1, w), th / max(1, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    canvas[:] = DBG_BG_COLOR
    x_off = (tw - nw) // 2
    y_off = (th - nh) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas


def main():
    class_names = _load_class_names(CLASS_NAMES_FILE)
    detector = YOLOv8_Paddle_Detector(
        model_dir=YOLO_MODEL_DIR,
        class_names_file=CLASS_NAMES_FILE,
        use_gpu=True,
        device_id=0,
        conf_threshold=0.4,
        nms_threshold=0.5
    )

    tracker = ByteTrack(
        high_conf_thresh=0.6,
        low_conf_thresh=0.1,
        iou_threshold=0.3,
        max_age=30,
        min_hits=3,
        use_giou=True,
        adaptive_noise=True,
        trail_maxlen=TRAIL_MAXLEN
    )

    trail_counter = TrailCrossCounter(roi_xyxy=DETECT_ROI, cooldown_frames=8)

    # Buffer thumbnail debug (kanan-atas)
    debug_thumbs = deque(maxlen=DBG_THUMB_MAX)  # simpan (thumb_img, caption_str, color)

    cap = cv2.VideoCapture(0 if str(VIDEO_PATH) == "0" else VIDEO_PATH)
    if not cap.isOpened():
        print(f"Gagal membuka video: {VIDEO_PATH}")
        return

    frame_idx = 0
    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if RESIZE_WIDTH or RESIZE_HEIGHT:
            frame = cv2.resize(frame, (RESIZE_WIDTH or frame.shape[1],
                                       RESIZE_HEIGHT or frame.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

        vis = frame.copy()
        do_det = (frame_idx % (FRAME_SKIP_DET + 1) == 0)

        if do_det:
            boxes, scores, class_ids = _detect_plates(detector, frame, class_names, TARGET_CLASS)
        else:
            boxes = np.empty((0, 4), np.float32)
            scores = np.empty((0,), np.float32)
            class_ids = np.empty((0,), np.int32)

        # === Update tracker ===
        tracks = tracker.update(boxes, scores, class_ids)

        # Peta cepat: track_id -> bbox saat ini (untuk crop jika ada event)
        bbox_by_tid: Dict[int, Tuple[int, int, int, int]] = {}
        for (x1, y1, x2, y2, tid, score, cid) in tracks:
            bbox_by_tid[int(tid)] = (int(x1), int(y1), int(x2), int(y2))

        # Update counter dari trail; dapatkan event crossing yang terjadi frame ini
        all_trails = tracker.get_all_trails()
        events = trail_counter.update_from_trails(all_trails, frame_idx)

        # === DEBUG: ambil crop untuk setiap event dan push ke panel kanan-atas ===
        # Event berisi ('masuk'|'keluar', track_id)
        for ev_type, tid in events:
            if tid in bbox_by_tid:
                x1, y1, x2, y2 = bbox_by_tid[tid]
                crop = _safe_crop(frame, x1, y1, x2, y2)
                thumb = _to_thumb(crop, DBG_THUMB_W, DBG_THUMB_H)
                caption = f"{ev_type.upper()}  ID:{tid}"
                color = (0, 255, 0) if ev_type == "masuk" else (0, 0, 255)
                debug_thumbs.appendleft((thumb, caption, color))
            # Jika bbox tidak ada (miss), lewati—lebih aman daripada crop asal.

        # ======== Visualisasi utama ========
        # ROI
        rx1, ry1, rx2, ry2 = DETECT_ROI
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 200, 255), 2)

        # Bbox + label (ID + confidence [+ class jika ada])
        for (x1, y1, x2, y2, tid, score, cid) in tracks:
            x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
            cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            name = class_names[cid] if 0 <= cid < len(class_names) else "obj"
            label = f"{name}#{int(tid)} {score*100:.1f}%"
            cv2.putText(vis, label, (x1i, max(0, y1i - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)

        # Gambar trails
        if DRAW_TRAILS:
            for pts in all_trails.values():
                if len(pts) >= 2:
                    for i in range(1, len(pts)):
                        cv2.line(vis, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                                 (int(pts[i][0]), int(pts[i][1])), (255, 255, 0), TRAIL_THICKNESS)
                    cv2.circle(vis, (int(pts[-1][0]), int(pts[-1][1])), 2 + TRAIL_THICKNESS, (255, 255, 0), -1)

        # HUD teks
        masuk, keluar = trail_counter.get_counts()
        cv2.putText(vis, f"Masuk : {masuk}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, f"Keluar: {keluar}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(vis, f"Active Tracks: {tracker.get_track_count()}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # FPS biru
        curr_time = time.perf_counter()
        fps = 1.0 / max(1e-9, curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # === Render panel debug crops (KANAN-ATAS) ===
        # Komentar: ini HANYA untuk debugging; nonaktifkan di produksi.
        H, W = vis.shape[:2]
        panel_x = W - DBG_PANEL_MARGIN - DBG_THUMB_W
        panel_y = DBG_PANEL_MARGIN
        for i, (thumb, caption, col) in enumerate(list(debug_thumbs)):
            y = panel_y + i * (DBG_THUMB_H + DBG_THUMB_GAP)
            if y + DBG_THUMB_H + 24 > H:  # +24 kira-kira area text di bawah
                break
            x = panel_x

            # background + border
            cv2.rectangle(vis, (x - 2, y - 2), (x + DBG_THUMB_W + 2, y + DBG_THUMB_H + 22), DBG_BG_COLOR, -1)
            cv2.rectangle(vis, (x - 2, y - 2), (x + DBG_THUMB_W + 2, y + DBG_THUMB_H + 22), DBG_BORDER_COLOR, 1)

            # paste thumbnail
            roi = vis[y:y + DBG_THUMB_H, x:x + DBG_THUMB_W]
            if roi.shape[:2] == thumb.shape[:2]:
                vis[y:y + DBG_THUMB_H, x:x + DBG_THUMB_W] = thumb
            else:
                # fallback sizing jika ada mismatch tak terduga
                vis[y:y + DBG_THUMB_H, x:x + DBG_THUMB_W] = cv2.resize(thumb, (DBG_THUMB_W, DBG_THUMB_H))

            # caption di bawah thumb
            cv2.putText(vis, caption, (x, y + DBG_THUMB_H + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        # Show
        cv2.imshow("Deteksi + Tracking (Trail Counting) + Debug Crops", vis)
        if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

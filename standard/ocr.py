import os
import cv2
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR

# =========================
# KONFIGURASI
# =========================
MODEL_PATH = "../model/yolo/best_ncnn_model"
IMAGE_PATH = "../frame_22.jpg"
PADDLE_TEXT_DET_NAME="PP-OCRv5_server_det"
PADDLE_TEXT_DET_MODEL_DIR = f"model/ocr/{PADDLE_TEXT_DET_NAME}_infer"
PADDLE_TEXT_REC_NAME="PP-OCRv5_server_rec"
PADDLE_TEXT_REC_MODEL_DIR = f"model/ocr/{PADDLE_TEXT_REC_NAME}_infer"
OUT_DIR = "../output"
PLATE_CLASS_NAMES = {"license-plates", "license-plate", "plate", "lp"}
CROP_PADDING_RATIO = 0.06

# Upscale/enhance params
UPSCALE_TARGET_MIN_SIDE = 96
UPSCALE_MAX_SCALE = 3.0
DENOISE = True
USE_CLAHE = True
UNSHARP = True

# =========================
# INISIALISASI
# =========================
os.makedirs(OUT_DIR, exist_ok=True)
yolo = YOLO(MODEL_PATH)
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_name=PADDLE_TEXT_DET_NAME,
    text_detection_model_dir=PADDLE_TEXT_DET_MODEL_DIR,
    text_recognition_model_name=PADDLE_TEXT_REC_NAME,
    text_recognition_model_dir=PADDLE_TEXT_REC_MODEL_DIR,
)


# =========================
# FUNGSI PEMBANTU
# =========================
def clamp(v, lo, hi):
    return int(max(lo, min(hi, v)))


def expand_box(x1, y1, x2, y2, pad, W, H):
    return (
        clamp(x1 - pad, 0, W - 1),
        clamp(y1 - pad, 0, H - 1),
        clamp(x2 + pad, 0, W - 1),
        clamp(y2 + pad, 0, H - 1),
    )


def enhance_and_upscale_plate(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)

    if DENOISE:
        g = cv2.bilateralFilter(g, d=5, sigmaColor=75, sigmaSpace=75)

    if UNSHARP:
        blur = cv2.GaussianBlur(g, (0, 0), 1.0)
        g = cv2.addWeighted(g, 1.5, blur, -0.5, 0)

    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    h, w = bgr.shape[:2]
    scale = max(1.0, UPSCALE_TARGET_MIN_SIDE / float(min(h, w)))
    scale = min(scale, UPSCALE_MAX_SCALE)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    if scale > 1.01:
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif scale < 0.99:
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return bgr


# =========================
# PROSES
# =========================
total_start = time.time()

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Gagal membuka gambar: {IMAGE_PATH}")
H, W = img.shape[:2]

# Waktu deteksi
det_start = time.time()
det_res = yolo(IMAGE_PATH)[0]
det_end = time.time()
print(f"Waktu deteksi YOLO: {det_end - det_start:.3f} detik")

names = det_res.names

lp_boxes = []
if det_res.boxes is not None and len(det_res.boxes) > 0:
    for b in det_res.boxes:
        cls_id = int(b.cls.item())
        cls_name = names.get(cls_id, str(cls_id))
        if cls_name in PLATE_CLASS_NAMES:
            x1, y1, x2, y2 = map(int, b.xyxy.squeeze().tolist())
            score = float(b.conf.item()) if b.conf is not None else 1.0
            lp_boxes.append((x1, y1, x2, y2, score, cls_name))

if not lp_boxes:
    print("Tidak ada bbox kelas 'license-plates' yang terdeteksi.")
else:
    print(f"Ditemukan {len(lp_boxes)} bbox plat.")

for i, (x1, y1, x2, y2, score, cls_name) in enumerate(lp_boxes, 1):
    w = x2 - x1
    h = y2 - y1
    pad_px = int(max(w, h) * CROP_PADDING_RATIO)
    ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, pad_px, W, H)

    crop = img[ey1:ey2, ex1:ex2].copy()
    plate_out_dir = os.path.join(OUT_DIR, f"plate_{i:02d}")
    os.makedirs(plate_out_dir, exist_ok=True)

    raw_path = os.path.join(plate_out_dir, "crop_raw.jpg")
    cv2.imwrite(raw_path, crop)

    up = enhance_and_upscale_plate(crop)
    up_path = os.path.join(plate_out_dir, "crop_upscaled.jpg")
    cv2.imwrite(up_path, up)

    print(f"\n=== OCR bbox #{i} [{ex1},{ey1},{ex2},{ey2}] (score={score:.3f}, cls={cls_name}) ===")
    ocr_start = time.time()
    ocr_result = ocr.predict(up)
    ocr_end = time.time()
    print(f"Waktu OCR untuk bbox #{i}: {ocr_end - ocr_start:.3f} detik")

    for res in ocr_result:
        res.print()
        res.save_to_img(plate_out_dir)
        res.save_to_json(plate_out_dir)

    cv2.rectangle(img, (ex1, ey1), (ex2, ey2), (0, 255, 0), 2)
    cv2.putText(img, f"plate#{i} {score:.2f}", (ex1, max(0, ey1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

annot_path = os.path.join(OUT_DIR, "annotated.jpg")
cv2.imwrite(annot_path, img)

total_end = time.time()
print(f"\nTotal waktu eksekusi: {total_end - total_start:.3f} detik")
print(f"- Anotasi semua bbox: {annot_path}")
print(f"- Per-plat: crop_raw.jpg, crop_upscaled.jpg, result.jpg, result.json")

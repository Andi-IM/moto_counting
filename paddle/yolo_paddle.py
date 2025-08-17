# main_detector.py

import argparse
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
import paddle.inference as paddle_infer

# --- Konstanta ---
PADDING_VALUE = 114
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']


class YOLOv8_Paddle_Detector:
    """
    Kelas untuk melakukan deteksi objek menggunakan model YOLOv8
    yang telah diekspor ke format PaddlePaddle.

    Kelas ini berfokus pada logika inti: memuat model dan melakukan inferensi
    pada sebuah gambar (numpy array).
    """

    def __init__(self, model_dir: str, class_names_file: str, use_gpu: bool = False, device_id: int = 0,
                 conf_threshold: float = 0.5, nms_threshold: float = 0.45):
        """
        Inisialisasi model detector.

        Args:
            model_dir (str): Path ke direktori model PaddlePaddle.
            class_names_file (str): Path ke file .names yang berisi nama kelas.
            use_gpu (bool): Gunakan GPU untuk inferensi.
            device_id (int): ID GPU yang akan digunakan.
            conf_threshold (float): Ambang batas kepercayaan untuk deteksi.
            nms_threshold (float): Ambang batas untuk Non-Maximum Suppression.
        """
        self.model_dir = model_dir
        self.class_names = self._load_class_names(class_names_file)
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        self.input_shape = (640, 640)  # Lebar, Tinggi
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.predictor = self._create_predictor(use_gpu, device_id)
        print("Model berhasil dimuat.")

    def _load_class_names(self, file_path: str) -> List[str]:
        """Memuat nama kelas dari file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File nama kelas tidak ditemukan di: {file_path}")
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _create_predictor(self, use_gpu: bool, device_id: int) -> paddle_infer.Predictor:
        """Membuat predictor PaddlePaddle dari file model."""
        model_file = os.path.join(self.model_dir, "model.pdmodel")
        params_file = os.path.join(self.model_dir, "model.pdiparams")

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"File model tidak ditemukan di: {model_file}")
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"File parameter tidak ditemukan di: {params_file}")

        config = paddle_infer.Config(model_file, params_file)
        config.enable_memory_optim()
        config.enable_mkldnn()

        if use_gpu:
            config.enable_use_gpu(100, device_id)
        else:
            config.disable_gpu()

        return paddle_infer.create_predictor(config)

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Pra-pemrosesan gambar sebelum dimasukkan ke model."""
        h_orig, w_orig = image.shape[:2]
        w_target, h_target = self.input_shape
        ratio = min(w_target / w_orig, h_target / h_orig)
        w_new, h_new = int(w_orig * ratio), int(h_orig * ratio)
        resized_img = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_AREA)

        padded_img = np.full((h_target, w_target, 3), PADDING_VALUE, dtype=np.uint8)
        pad_top = (h_target - h_new) // 2
        pad_left = (w_target - w_new) // 2
        padded_img[pad_top:pad_top + h_new, pad_left:pad_left + w_new] = resized_img

        img_tensor = padded_img.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)
        return img_tensor, ratio

    def _postprocess(self, output_tensor: np.ndarray, ratio: float, orig_shape: Tuple[int, int],
                     allowed_classes: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pasca-pemrosesan output dari model untuk mendapatkan bounding box."""
        predictions = np.squeeze(output_tensor).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(predictions) == 0:
            return np.array([]), np.array([]), np.array([])

        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Ekstraksi Box
        boxes = predictions[:, :4]
        # Konversi dari [cx, cy, w, h] ke [x1, y1, x2, y2]
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # Koreksi skala dan padding
        h_orig, w_orig = orig_shape
        w_target, h_target = self.input_shape
        w_new, h_new = int(w_orig * ratio), int(h_orig * ratio)
        pad_x = (w_target - w_new) / 2
        pad_y = (h_target - h_new) / 2
        boxes[:, 0] = (boxes[:, 0] - pad_x) / ratio
        boxes[:, 2] = (boxes[:, 2] - pad_x) / ratio
        boxes[:, 1] = (boxes[:, 1] - pad_y) / ratio
        boxes[:, 3] = (boxes[:, 3] - pad_y) / ratio

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.nms_threshold)

        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])

        indices = indices.flatten()
        final_boxes = boxes[indices]
        final_scores = scores[indices]
        final_class_ids = class_ids[indices]

        if allowed_classes:
            indices_to_keep = [i for i, cid in enumerate(final_class_ids) if self.class_names[cid] in allowed_classes]
            final_boxes = final_boxes[indices_to_keep]
            final_scores = final_scores[indices_to_keep]
            final_class_ids = final_class_ids[indices_to_keep]

        return final_boxes.astype(np.int32), final_scores, final_class_ids

    def detect(self, image: np.ndarray, allowed_classes: Optional[List[str]] = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Menjalankan inferensi pada sebuah gambar (numpy array) dan mengembalikan hasil deteksi.

        Args:
            image (np.ndarray): Gambar input dalam format BGR.
            allowed_classes (Optional[List[str]]): Daftar kelas yang diizinkan untuk dideteksi.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: boxes, scores, class_ids.
        """
        if image is None:
            return np.array([]), np.array([]), np.array([])

        h_orig, w_orig = image.shape[:2]
        input_tensor, ratio = self._preprocess(image)

        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        input_handle.reshape(input_tensor.shape)
        input_handle.copy_from_cpu(input_tensor)

        self.predictor.run()

        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()

        return self._postprocess(output_data, ratio, (h_orig, w_orig), allowed_classes)


# --- Fungsi Utilitas (di luar kelas) ---

def draw_detections(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                    detector: YOLOv8_Paddle_Detector) -> np.ndarray:
    """Menggambar hasil deteksi pada gambar."""
    img_copy = image.copy()
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        color = detector.colors[class_id]
        class_name = detector.class_names[class_id]
        label = f'{class_name}: {score:.2f}'

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img_copy


def detect_and_crop_objects(image: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
    """Memotong objek yang terdeteksi dari gambar."""
    cropped_images = []
    h, w = image.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box
        # Pastikan koordinat berada dalam batas gambar
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x1 < x2 and y1 < y2:
            cropped_images.append(image[y1:y2, x1:x2])
    return cropped_images


# --- Logika Aplikasi (Fungsi untuk memproses input) ---

def process_image(detector: YOLOv8_Paddle_Detector, source_path: str, output_path: Optional[str],
                  allowed_classes: Optional[List[str]]):
    """Memproses satu file gambar."""
    try:
        image = cv2.imread(source_path)
        if image is None:
            raise IOError(f"Tidak dapat membaca gambar dari: {source_path}")
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Melakukan deteksi pada gambar: {source_path}")
    boxes, scores, class_ids = detector.detect(image, allowed_classes)
    result_image = draw_detections(image, boxes, scores, class_ids, detector)

    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"Deteksi gambar selesai. Hasil disimpan di: {output_path}")

    # Tampilkan gambar
    cv2.imshow("Deteksi Gambar (Tekan Q untuk keluar)", result_image)
    while cv2.waitKey(1) & 0xFF != ord('q'):
        # Cek jika jendela ditutup
        if cv2.getWindowProperty("Deteksi Gambar (Tekan Q untuk keluar)", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def process_video(detector: YOLOv8_Paddle_Detector, source_path: str, output_path: Optional[str],
                  allowed_classes: Optional[List[str]]):
    """Memproses file video atau webcam."""
    is_webcam = str(source_path) == '0'
    source_to_open = 0 if is_webcam else source_path

    cap = cv2.VideoCapture(source_to_open)
    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka video di {source_path}")
        return

    writer = None
    if output_path:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    win_name = "Deteksi Video (Tekan Q untuk keluar)"
    print("Memulai deteksi video... Tekan 'q' pada jendela video untuk keluar.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes, scores, class_ids = detector.detect(frame, allowed_classes)
        result_frame = draw_detections(frame, boxes, scores, class_ids, detector)

        cv2.imshow(win_name, result_frame)
        if writer:
            writer.write(result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Deteksi video selesai. Hasil disimpan di: {output_path}")
    cv2.destroyAllWindows()
    print("Deteksi video dihentikan.")


# --- Titik Masuk Utama (Main Entry Point) ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deteksi Objek menggunakan YOLOv8 dan PaddlePaddle")
    parser.add_argument('--model-dir', type=str, required=True, help='Path ke direktori model inference PaddlePaddle.')
    parser.add_argument('--class-names', type=str, required=True, help='Path ke file .names yang berisi nama kelas.')
    parser.add_argument('--input', type=str, required=True, help="Path ke gambar/video, atau '0' untuk webcam.")
    parser.add_argument('--output', type=str, default=None, help='Path untuk menyimpan file hasil (opsional).')
    parser.add_argument('--use-gpu', action='store_true', help='Gunakan GPU untuk inferensi.')
    parser.add_argument('--allowed-classes', nargs='*', default=None,
                        help='Daftar kelas yang ingin dideteksi (opsional). Contoh: --allowed-classes mobil motor')

    args = parser.parse_args()

    print("Menginisialisasi detektor...")
    try:
        detector = YOLOv8_Paddle_Detector(
            model_dir=args.model_dir,
            class_names_file=args.class_names,
            use_gpu=args.use_gpu
        )
    except Exception as e:
        print(f"Gagal menginisialisasi model: {e}")
        exit()

    if args.allowed_classes:
        print(f"Hanya akan mendeteksi kelas: {', '.join(args.allowed_classes)}")

    input_lower = args.input.lower()
    if input_lower == '0' or any(input_lower.endswith(ext) for ext in VIDEO_EXTENSIONS):
        process_video(detector, args.input, args.output, args.allowed_classes)
    elif any(input_lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
        process_image(detector, args.input, args.output, args.allowed_classes)
    else:
        print(f"Error: Format input tidak dikenali untuk '{args.input}'.")


"""
Examples

python yolo_paddle.py ^
    --model-dir "../model/yolo/yolov8n-fold5/best_paddle_model/inference_model" ^
    --class-names "../label.names" ^
    --input "../frame_22.jpg" ^
    --output "../result_image.png" ^
    --use-gpu
    
python yolo_paddle.py --model-dir "path/ke/model" --class-names "path/ke/label.names" --input "../video_6178.mp4" --output "../result_video.mp4"
    
python yolo_paddle.py ^
    --model-dir "path/ke/model" ^
    --class-names "path/ke/label.names" ^
    --input 0 ^
    --use-gpu
"""
from dataclasses import dataclass
import numpy as np

from scipy.optimize import linear_sum_assignment

# ------------ util bbox & IoU ------------
def _iou_xyxy(a, b):
    # a, b: [x1,y1,x2,y2]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _xyxy_to_xysr(box):
    x1, y1, x2, y2 = box
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    x = x1 + w / 2.0
    y = y1 + h / 2.0
    s = w * h
    r = w / h
    return np.array([x, y, s, r], dtype=np.float32)


def _xysr_to_xyxy(x, y, s, r):
    # w = sqrt(s*r), h = s / w = sqrt(s/r)
    w = np.sqrt(max(1e-6, s * r))
    h = max(1e-6, s / w)
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# --- Kalman Filter khusus state [x,y,s,r,vx,vy,vs] ---
class KalmanXYSR:
    def __init__(self):
        # Dimensi
        self.ndim = 7
        self.mdim = 4
        # State
        self.x = np.zeros((self.ndim, 1), dtype=np.float32)
        self.P = np.eye(self.ndim, dtype=np.float32) * 10.0
        # Model (dt = 1)
        self.F = np.eye(self.ndim, dtype=np.float32)
        self.F[0, 4] = 1.0  # x += vx
        self.F[1, 5] = 1.0  # y += vy
        self.F[2, 6] = 1.0  # s += vs
        self.H = np.zeros((self.mdim, self.ndim), dtype=np.float32)
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y
        self.H[2, 2] = 1.0  # s
        self.H[3, 3] = 1.0  # r
        # Noise
        self.R = np.eye(self.mdim, dtype=np.float32) * 1.0
        self.Q = np.eye(self.ndim, dtype=np.float32)
        self.Q[0, 0] = self.Q[1, 1] = self.Q[2, 2] = 1.0
        self.Q[4, 4] = self.Q[5, 5] = self.Q[6, 6] = 1.0

    def initiate(self, meas_xysr):
        self.x[:4, 0] = meas_xysr.reshape(-1)
        self.x[4:, 0] = 0.0  # kecepatan awal 0
        self.P = np.eye(self.ndim, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0  # ketidakpastian kecepatan lebih besar

    def predict(self):
        # s tidak boleh <= 0
        if self.x[2, 0] + self.x[6, 0] <= 1e-6:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, meas_xysr):
        z = meas_xysr.reshape((self.mdim, 1)).astype(np.float32)
        y = z - (self.H @ self.x)  # inovasi
        S = self.H @ self.P @ self.H.T + self.R  # kovarians inovasi
        K = self.P @ self.H.T @ np.linalg.inv(S)  # gain
        self.x = self.x + K @ y
        I = np.eye(self.ndim, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def get_xyxy(self):
        x, y, s, r = self.x[0, 0], self.x[1, 0], max(1e-6, self.x[2, 0]), max(1e-6, self.x[3, 0])
        return _xysr_to_xyxy(x, y, s, r)


@dataclass
class Track:
    id: int
    kf: KalmanXYSR
    time_since_update: int = 0
    hits: int = 0
    age: int = 0
    score: float = 0.0
    cls: int = -1

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, det_box_xyxy, score=None, cls_id=None):
        meas = _xyxy_to_xysr(det_box_xyxy)
        self.kf.update(meas)
        self.time_since_update = 0
        self.hits += 1
        if score is not None: self.score = float(score)
        if cls_id is not None: self.cls = int(cls_id)

    def to_xyxy(self):
        return self.kf.get_xyxy()


# ------------ SORT dengan Hungarian (SciPy) ------------
class SortKF:
    """
    Kalman + Hungarian (cost = 1 - IoU). Parameter umum:
    - iou_threshold: IoU minimum agar pasangan dianggap valid.
    - max_age: berapa frame tanpa update sebelum track dihapus.
    - min_hits: berapa kali update sebelum track dikeluarkan sebagai output tetap.
    """

    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self._next_id = 1

    # --- asosiasi deteksi <-> track (Hungarian / greedy) ---
    def _associate(self, dets_xyxy):
        N, M = len(self.tracks), len(dets_xyxy)
        if N == 0 or M == 0:
            return [], list(range(M)), list(range(N))

        cost = np.zeros((N, M), dtype=np.float32)
        for i, tr in enumerate(self.tracks):
            tb = tr.to_xyxy()
            for j, db in enumerate(dets_xyxy):
                cost[i, j] = 1.0 - _iou_xyxy(tb, db)

        # Hungarian
        rows, cols = linear_sum_assignment(cost)
        matches = []
        unmatched_t = set(range(N))
        unmatched_d = set(range(M))

        for i, j in zip(rows, cols):
            iou = 1.0 - cost[i, j]
            if iou >= self.iou_threshold:
                matches.append((i, j))
                unmatched_t.discard(i)
                unmatched_d.discard(j)

        return matches, sorted(list(unmatched_d)), sorted(list(unmatched_t))

    def update(self, det_boxes_xyxy, det_scores=None, det_classes=None):
        dets = det_boxes_xyxy.astype(np.float32)

        # 1) predict semua track
        for tr in self.tracks:
            tr.predict()

        # 2) asosiasi
        matches, unmatched_d, unmatched_t = self._associate(dets)

        # 3) update matched tracks
        for i, j in matches:
            score = det_scores[j] if det_scores is not None and len(det_scores) > j else None
            cid = det_classes[j] if det_classes is not None and len(det_classes) > j else None
            self.tracks[i].update(dets[j], score, cid)

        # 4) buat track baru dari deteksi yang belum terpasang
        for j in unmatched_d:
            kf = KalmanXYSR()
            kf.initiate(_xyxy_to_xysr(dets[j]))
            score = det_scores[j] if det_scores is not None and len(det_scores) > j else 0.0
            cid = det_classes[j] if det_classes is not None and len(det_classes) > j else -1
            self.tracks.append(Track(id=self._next_id, kf=kf, score=float(score), cls=int(cid)))
            self._next_id += 1

        # 5) buang track yang terlalu lama tak ter-update
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 6) keluaran: hanya track cukup matang atau baru diupdate
        outputs = []
        for t in self.tracks:
            if t.hits >= self.min_hits or t.time_since_update == 0:
                x1, y1, x2, y2 = t.to_xyxy().astype(int)
                outputs.append((x1, y1, x2, y2, t.id, t.score, t.cls))
        return outputs
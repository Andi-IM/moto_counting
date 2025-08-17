from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque


# ------------ Enhanced Util Functions ------------
def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate IoU between two bounding boxes in xyxy format."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _giou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Generalized IoU for better distance metric."""
    iou = _iou_xyxy(a, b)
    if iou == 1.0: return iou

    # Calculate enclosing box
    cx1, cy1 = min(a[0], b[0]), min(a[1], b[1])
    cx2, cy2 = max(a[2], b[2]), max(a[3], b[3])
    c_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)

    if c_area <= 0: return iou

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - _iou_xyxy(a, b) * min(area_a, area_b)

    return iou - (c_area - union) / c_area


def _xyxy_to_xysr(box: np.ndarray) -> np.ndarray:
    """Convert xyxy to center_x, center_y, scale, ratio format."""
    x1, y1, x2, y2 = box
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    x = x1 + w / 2.0
    y = y1 + h / 2.0
    s = w * h
    r = w / h
    return np.array([x, y, s, r], dtype=np.float32)


def _xysr_to_xyxy(x: float, y: float, s: float, r: float) -> np.ndarray:
    """Convert center_x, center_y, scale, ratio to xyxy format."""
    w = np.sqrt(max(1e-6, s * r))
    h = max(1e-6, s / w)
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# --- Enhanced Kalman Filter with Adaptive Noise ---
class AdaptiveKalmanXYSR:
    """Enhanced Kalman Filter with adaptive noise and velocity damping."""

    def __init__(self, adaptive_noise: bool = True):
        self.ndim = 7  # [x, y, s, r, vx, vy, vs]
        self.mdim = 4  # [x, y, s, r]
        self.adaptive_noise = adaptive_noise

        # State vector
        self.x = np.zeros((self.ndim, 1), dtype=np.float32)
        self.P = np.eye(self.ndim, dtype=np.float32) * 10.0

        # Transition matrix (constant velocity model)
        self.F = np.eye(self.ndim, dtype=np.float32)
        self.F[0, 4] = 1.0  # x += vx
        self.F[1, 5] = 1.0  # y += vy
        self.F[2, 6] = 1.0  # s += vs

        # Observation matrix
        self.H = np.zeros((self.mdim, self.ndim), dtype=np.float32)
        self.H[0, 0] = 1.0  # observe x
        self.H[1, 1] = 1.0  # observe y
        self.H[2, 2] = 1.0  # observe s
        self.H[3, 3] = 1.0  # observe r

        # Measurement noise covariance
        self.R = np.eye(self.mdim, dtype=np.float32)
        self.R[0, 0] = self.R[1, 1] = 1.0  # position uncertainty
        self.R[2, 2] = 10.0  # scale uncertainty
        self.R[3, 3] = 10000.0  # aspect ratio uncertainty

        # Process noise covariance
        self.Q = np.eye(self.ndim, dtype=np.float32)
        self.Q[0, 0] = self.Q[1, 1] = 1.0  # position process noise
        self.Q[2, 2] = 0.01  # scale process noise
        self.Q[4, 4] = self.Q[5, 5] = 1.0  # velocity process noise
        self.Q[6, 6] = 0.01  # scale velocity process noise

        # For adaptive noise
        self.innovation_history = []
        self.max_history = 10

    def initiate(self, measurement: np.ndarray) -> None:
        """Initialize state with first measurement."""
        self.x[:4, 0] = measurement.reshape(-1)
        self.x[4:, 0] = 0.0  # initial velocity = 0

        # Initial uncertainty
        self.P = np.eye(self.ndim, dtype=np.float32)
        self.P[:4, :4] *= 10.0  # position and scale uncertainty
        self.P[4:, 4:] *= 1000.0  # velocity uncertainty

    def predict(self) -> np.ndarray:
        """Predict next state."""
        # Constrain scale to be positive
        if self.x[2, 0] + self.x[6, 0] <= 1e-6:
            self.x[6, 0] = max(-self.x[2, 0] + 1e-6, 0)

        # Apply velocity damping to prevent drift
        self.x[4:, 0] *= 0.9

        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Ensure scale stays positive
        self.x[2, 0] = max(1e-6, self.x[2, 0])
        self.x[3, 0] = max(0.1, self.x[3, 0])  # min aspect ratio

        return self.x.copy()

    def update(self, measurement: np.ndarray) -> None:
        """Update state with measurement."""
        z = measurement.reshape((self.mdim, 1)).astype(np.float32)

        # Innovation (residual)
        y = z - (self.H @ self.x)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Adaptive noise adjustment
        if self.adaptive_noise:
            self._adapt_noise(y, S)

        # Kalman gain
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ self.H.T @ np.linalg.pinv(S)

        # Update state and covariance
        self.x = self.x + K @ y
        I = np.eye(self.ndim, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        # Ensure valid values
        self.x[2, 0] = max(1e-6, self.x[2, 0])  # scale > 0
        self.x[3, 0] = max(0.1, self.x[3, 0])  # aspect ratio > 0.1

    def _adapt_noise(self, innovation: np.ndarray, S: np.ndarray) -> None:
        """Adapt noise parameters based on innovation."""
        self.innovation_history.append(innovation.copy())
        if len(self.innovation_history) > self.max_history:
            self.innovation_history.pop(0)

        if len(self.innovation_history) >= 3:
            recent_innovations = np.hstack(self.innovation_history[-3:])
            innovation_var = np.var(recent_innovations, axis=1, keepdims=True)

            # Adapt measurement noise
            self.R = np.diag(np.maximum(innovation_var.flatten(),
                                        np.diag(self.R) * 0.1))

    def get_xyxy(self) -> np.ndarray:
        """Get bounding box in xyxy format."""
        x, y = self.x[0, 0], self.x[1, 0]
        s, r = max(1e-6, self.x[2, 0]), max(0.1, self.x[3, 0])
        return _xysr_to_xyxy(x, y, s, r)

    def get_velocity(self) -> np.ndarray:
        """Get velocity vector."""
        return self.x[4:7, 0].copy()


# --- Enhanced Track with State Management ---
@dataclass
class EnhancedTrack:
    """Enhanced track with better state management and features."""
    id: int
    kf: AdaptiveKalmanXYSR
    time_since_update: int = 0
    hits: int = 0
    age: int = 0
    score: float = 0.0
    cls: int = -1
    state: str = 'tentative'  # 'tentative', 'confirmed', 'deleted'
    trail: deque = None  # deque of (x,y)
    trail_maxlen: int = 30  # default; bisa diubah dari bytetrack

    # Additional features
    max_score: float = 0.0
    avg_score: float = 0.0
    detection_sizes: List[float] = None
    velocity_history: List[np.ndarray] = None

    def __post_init__(self):
        if self.detection_sizes is None:
            self.detection_sizes = []
        if self.velocity_history is None:
            self.velocity_history = []
        if self.trail is None:
            self.trail = deque(maxlen=self.trail_maxlen)

    def _push_trail_point(self):
        """Mengambil center point dari bbox prediksi """
        x1, y1, x2, y2 = self.to_xyxy()
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
        self.trail.append((cx, cy))

    def predict(self) -> None:
        """Predict next state and update track properties."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

        # Track velocity history
        velocity = self.kf.get_velocity()
        self.velocity_history.append(velocity.copy())
        if len(self.velocity_history) > 10:
            self.velocity_history.pop(0)
        self._push_trail_point()

    def update(self, det_box_xyxy: np.ndarray, score: Optional[float] = None,
               cls_id: Optional[int] = None) -> None:
        """Update track with detection."""
        measurement = _xyxy_to_xysr(det_box_xyxy)
        self.kf.update(measurement)

        self.time_since_update = 0
        self.hits += 1

        if score is not None:
            self.score = float(score)
            self.max_score = max(self.max_score, self.score)
            # Update running average
            alpha = 0.8  # smoothing factor
            if self.hits == 1:
                self.avg_score = self.score
            else:
                self.avg_score = alpha * self.avg_score + (1 - alpha) * self.score

        if cls_id is not None:
            self.cls = int(cls_id)

        # Track detection size for consistency check
        x1, y1, x2, y2 = det_box_xyxy
        size = (x2 - x1) * (y2 - y1)
        self.detection_sizes.append(size)
        if len(self.detection_sizes) > 5:
            self.detection_sizes.pop(0)

        # Update state
        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'
        self._push_trail_point()

    def to_xyxy(self) -> np.ndarray:
        """Get bounding box in xyxy format."""
        return self.kf.get_xyxy()

    def is_confirmed(self) -> bool:
        """Check if track is confirmed."""
        return self.state == 'confirmed'

    def mark_missed(self) -> None:
        """Mark track as missed (no detection matched)."""
        if self.state == 'tentative':
            self.state = 'deleted'


# --- ByteTrack: State-of-the-Art Multi-Object Tracking ---
class ByteTrack:
    """
    ByteTrack - State-of-the-art multi-object tracking algorithm.

    Key improvements over SORT:
    1. Two-stage association with high and low confidence detections
    2. Adaptive Kalman filter with velocity damping
    3. Better track state management
    4. Enhanced distance metrics (IoU + GIoU)
    5. Confidence-aware track initialization
    """

    def __init__(self,
                 high_conf_thresh: float = 0.6,
                 low_conf_thresh: float = 0.1,
                 iou_threshold: float = 0.3,
                 max_age: int = 30,
                 min_hits: int = 3,
                 use_giou: bool = True,
                 adaptive_noise: bool = True,
                 trail_maxlen: int = 30):
        """
        Args:
            high_conf_thresh: High confidence threshold for first association
            low_conf_thresh: Low confidence threshold for second association
            iou_threshold: IoU threshold for association
            max_age: Maximum frames without update before deletion
            min_hits: Minimum hits before track is considered confirmed
            use_giou: Whether to use GIoU instead of IoU for distance
            adaptive_noise: Whether to use adaptive Kalman filter noise
        """
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.use_giou = use_giou
        self.adaptive_noise = adaptive_noise
        self.trail_maxlen = trail_maxlen

        self.tracks: List[EnhancedTrack] = []
        self._next_id = 1

    def _compute_distance_matrix(self, tracks: List[EnhancedTrack],
                                 detections: np.ndarray) -> np.ndarray:
        """Compute distance matrix between tracks and detections."""
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((0, 0))

        distance_func = _giou_xyxy if self.use_giou else _iou_xyxy

        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

        for i, track in enumerate(tracks):
            track_box = track.to_xyxy()
            for j, det_box in enumerate(detections):
                # Use 1 - IoU/GIoU as distance
                similarity = distance_func(track_box, det_box)
                cost_matrix[i, j] = 1.0 - similarity

        return cost_matrix

    def _associate(self, tracks: List[EnhancedTrack],
                   detections: np.ndarray) -> Tuple[List[Tuple[int, int]],
    List[int], List[int]]:
        """Associate tracks with detections using Hungarian algorithm."""
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))

        # Compute cost matrix
        cost_matrix = self._compute_distance_matrix(tracks, detections)

        # Hungarian assignment
        try:
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
        except:
            return [], list(range(len(detections))), list(range(len(tracks)))

        # Filter out assignments with high cost
        matches = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for t_idx, d_idx in zip(track_indices, det_indices):
            cost = cost_matrix[t_idx, d_idx]
            similarity = 1.0 - cost

            if similarity >= self.iou_threshold:
                matches.append((t_idx, d_idx))
                unmatched_tracks.discard(t_idx)
                unmatched_dets.discard(d_idx)

        return matches, sorted(unmatched_dets), sorted(unmatched_tracks)

    def update(self, detections: np.ndarray,
               scores: Optional[np.ndarray] = None,
               classes: Optional[np.ndarray] = None) -> List[Tuple]:
        """
        Update tracker with new detections.

        Args:
            detections: Array of bounding boxes in xyxy format [N, 4]
            scores: Detection confidence scores [N]
            classes: Detection class IDs [N]

        Returns:
            List of active tracks: [(x1, y1, x2, y2, track_id, score, class)]
        """
        if len(detections) == 0:
            detections = np.empty((0, 4))
        if scores is None:
            scores = np.ones(len(detections))
        if classes is None:
            classes = np.zeros(len(detections), dtype=int)

        detections = detections.astype(np.float32)
        scores = np.array(scores, dtype=np.float32)
        classes = np.array(classes, dtype=int)

        # Separate high and low confidence detections
        high_conf_mask = scores >= self.high_conf_thresh
        low_conf_mask = (scores >= self.low_conf_thresh) & (scores < self.high_conf_thresh)

        high_conf_dets = detections[high_conf_mask]
        high_conf_scores = scores[high_conf_mask]
        high_conf_classes = classes[high_conf_mask]

        low_conf_dets = detections[low_conf_mask]
        low_conf_scores = scores[low_conf_mask]
        low_conf_classes = classes[low_conf_mask]

        # Step 1: Predict all tracks
        for track in self.tracks:
            track.predict()

        # Step 2: First association with high confidence detections
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        matches_1, unmatched_high_dets, unmatched_conf_tracks = self._associate(
            confirmed_tracks, high_conf_dets)

        # Update matched tracks
        for track_idx, det_idx in matches_1:
            track = confirmed_tracks[track_idx]
            track.update(high_conf_dets[det_idx],
                         high_conf_scores[det_idx],
                         high_conf_classes[det_idx])

        # Step 3: Second association with remaining tracks and low confidence detections
        remaining_tracks = [confirmed_tracks[i] for i in unmatched_conf_tracks]
        tentative_tracks = [t for t in self.tracks if not t.is_confirmed()]
        remaining_tracks.extend(tentative_tracks)

        # Combine remaining high confidence and low confidence detections
        remaining_dets = np.vstack([
            high_conf_dets[unmatched_high_dets],
            low_conf_dets
        ]) if len(unmatched_high_dets) > 0 and len(low_conf_dets) > 0 else (
            high_conf_dets[unmatched_high_dets] if len(unmatched_high_dets) > 0 else
            low_conf_dets if len(low_conf_dets) > 0 else
            np.empty((0, 4))
        )

        remaining_scores = np.concatenate([
            high_conf_scores[unmatched_high_dets],
            low_conf_scores
        ]) if len(unmatched_high_dets) > 0 and len(low_conf_dets) > 0 else (
            high_conf_scores[unmatched_high_dets] if len(unmatched_high_dets) > 0 else
            low_conf_scores if len(low_conf_dets) > 0 else
            np.empty(0)
        )

        remaining_classes = np.concatenate([
            high_conf_classes[unmatched_high_dets],
            low_conf_classes
        ]) if len(unmatched_high_dets) > 0 and len(low_conf_dets) > 0 else (
            high_conf_classes[unmatched_high_dets] if len(unmatched_high_dets) > 0 else
            low_conf_classes if len(low_conf_dets) > 0 else
            np.empty(0, dtype=int)
        )

        matches_2, unmatched_remaining_dets, unmatched_remaining_tracks = self._associate(
            remaining_tracks, remaining_dets)

        # Update second round matches
        for track_idx, det_idx in matches_2:
            track = remaining_tracks[track_idx]
            track.update(remaining_dets[det_idx],
                         remaining_scores[det_idx],
                         remaining_classes[det_idx])

        # Step 4: Create new tracks from unmatched high confidence detections
        new_track_dets = remaining_dets[unmatched_remaining_dets] if len(unmatched_remaining_dets) > 0 else np.empty(
            (0, 4))
        new_track_scores = remaining_scores[unmatched_remaining_dets] if len(
            unmatched_remaining_dets) > 0 else np.empty(0)
        new_track_classes = remaining_classes[unmatched_remaining_dets] if len(
            unmatched_remaining_dets) > 0 else np.empty(0, dtype=int)

        # Only create tracks from high confidence detections
        for i, (det, score, cls) in enumerate(zip(new_track_dets, new_track_scores, new_track_classes)):
            if score >= self.high_conf_thresh:  # Only high conf detections create new tracks
                kf = AdaptiveKalmanXYSR(adaptive_noise=self.adaptive_noise)
                kf.initiate(_xyxy_to_xysr(det))

                track = EnhancedTrack(
                    id=self._next_id,
                    kf=kf,
                    score=float(score),
                    cls=int(cls)
                )
                track.update(det, score, cls)

                self.tracks.append(track)
                self._next_id += 1

        # Step 5: Mark unmatched tracks as missed
        all_unmatched_track_indices = set()

        # From first association
        for i in unmatched_conf_tracks:
            all_unmatched_track_indices.add(self.tracks.index(confirmed_tracks[i]))

        # From second association
        for i in unmatched_remaining_tracks:
            track = remaining_tracks[i]
            all_unmatched_track_indices.add(self.tracks.index(track))

        for track_idx in all_unmatched_track_indices:
            self.tracks[track_idx].mark_missed()

        # Step 6: Remove dead tracks
        self.tracks = [t for t in self.tracks
                       if t.state != 'deleted' and t.time_since_update <= self.max_age]

        # Step 7: Generate output
        outputs = []
        for track in self.tracks:
            if (track.is_confirmed() and track.time_since_update <= 1) or \
                    (track.hits >= self.min_hits):
                x1, y1, x2, y2 = track.to_xyxy().astype(int)
                outputs.append((x1, y1, x2, y2, track.id, track.score, track.cls))

        return outputs

    def get_track_count(self) -> int:
        """Get current number of active tracks."""
        return len([t for t in self.tracks if t.is_confirmed()])

    def get_trail(self, track_id: int) -> List[Tuple[float, float]]:
        for t in self.tracks:
            if t.id == track_id:
                return list(t.trail)
        return []

    def get_all_trails(self) -> dict:
        # {track_id: [(x,y), ...]}
        return {t.id: list(t.trail) for t in self.tracks if t.is_confirmed()}

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks = []
        self._next_id = 1


# --- Usage Example ---
if __name__ == "__main__":
    # Initialize tracker with optimal parameters
    tracker = ByteTrack(
        high_conf_thresh=0.6,
        low_conf_thresh=0.1,
        iou_threshold=0.3,
        max_age=30,
        min_hits=3,
        use_giou=True,
        adaptive_noise=True
    )

    # Example usage
    # detections = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
    # scores = np.array([0.8, 0.9])
    # classes = np.array([0, 1])

    # tracks = tracker.update(detections, scores, classes)
    # print(f"Active tracks: {tracks}")

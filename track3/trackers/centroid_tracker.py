import math


class CentroidTracker:
    """
    Assigns a stable integer ID to each detected person by matching this
    frame's centroids to the closest tracked centroid from last frame.

    Coordinates are normalized to [0, 1] so max_distance is
    resolution-independent.

    Confirmation debounce:
        A detection that matches no existing track is not promoted to a
        real ID immediately. It must be seen for `min_confirmation_frames`
        consecutive frames first. This kills phantom IDs from 1-2 frame
        pose false positives, without hurting real re-identification
        (a real person does not disappear in 3 frames).
    """

    def __init__(self, max_distance=0.15, max_missed_frames=20,
                 min_confirmation_frames=3):
        self.next_id = 0
        self.tracked = {}          # id -> (x, y) centroid, normalized
        self.missed_frames = {}    # id -> consecutive frames not matched
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames

        # Debounce for brand-new IDs
        self.min_confirmation_frames = min_confirmation_frames
        self.pending = {}          # pending_key -> {"centroid": (x,y), "count": int}
        self.pending_next_key = 0

    def _promote_pending(self, pending_key):
        """Turn a confirmed pending detection into a real tracked ID."""
        new_id = self.next_id
        self.next_id += 1
        self.tracked[new_id] = self.pending[pending_key]["centroid"]
        self.missed_frames[new_id] = 0
        del self.pending[pending_key]
        return new_id

    def update(self, detected_centroids):
        """
        detected_centroids: list of (x, y) tuples for this frame.
        Returns: dict mapping detected_centroid_index -> stable ID.
        """
        assignments = {}
        used_ids = set()

        # --- 1. Match detections to existing tracked people ---
        unmatched = []  # indices we could not match
        for i, (dx, dy) in enumerate(detected_centroids):
            best_id, best_dist = None, self.max_distance
            for pid, (px, py) in self.tracked.items():
                if pid in used_ids:
                    continue
                dist = math.sqrt((dx - px) ** 2 + (dy - py) ** 2)
                if dist < best_dist:
                    best_id, best_dist = pid, dist

            if best_id is not None:
                assignments[i] = best_id
                self.tracked[best_id] = (dx, dy)
                self.missed_frames[best_id] = 0
                used_ids.add(best_id)
            else:
                unmatched.append(i)

        # --- 2. Match unmatched detections to pending candidates ---
        still_unmatched = []
        for i in unmatched:
            dx, dy = detected_centroids[i]
            best_key, best_dist = None, self.max_distance
            for key, info in self.pending.items():
                px, py = info["centroid"]
                dist = math.sqrt((dx - px) ** 2 + (dy - py) ** 2)
                if dist < best_dist:
                    best_key, best_dist = key, dist

            if best_key is not None:
                self.pending[best_key]["centroid"] = (dx, dy)
                self.pending[best_key]["count"] += 1
                if self.pending[best_key]["count"] >= self.min_confirmation_frames:
                    # Promote to a real ID
                    new_id = self._promote_pending(best_key)
                    assignments[i] = new_id
                    used_ids.add(new_id)
            else:
                still_unmatched.append(i)

        # --- 3. Anything still unmatched becomes a new pending candidate ---
        for i in still_unmatched:
            key = self.pending_next_key
            self.pending_next_key += 1
            self.pending[key] = {"centroid": detected_centroids[i], "count": 1}

        # --- 4. Age out unconfirmed pending candidates ---
        # A pending candidate not matched this frame is dropped immediately.
        matched_pending_keys = set()
        for i in unmatched:
            dx, dy = detected_centroids[i]
            for key, info in self.pending.items():
                px, py = info["centroid"]
                if math.sqrt((dx - px) ** 2 + (dy - py) ** 2) < self.max_distance:
                    matched_pending_keys.add(key)
                    break
        for key in list(self.pending.keys()):
            if key not in matched_pending_keys and self.pending[key]["count"] < self.min_confirmation_frames:
                del self.pending[key]

        # --- 5. Age out real tracked IDs ---
        for pid in list(self.tracked.keys()):
            if pid not in used_ids:
                self.missed_frames[pid] = self.missed_frames.get(pid, 0) + 1
                if self.missed_frames[pid] > self.max_missed_frames:
                    del self.tracked[pid]
                    del self.missed_frames[pid]

        return assignments
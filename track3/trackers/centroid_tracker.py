import math


class CentroidTracker:
    """
    Assigns a stable integer ID to each detected person by matching this
    frame's centroids to the closest tracked centroid from last frame.

    Coordinates are expected to be normalized to [0, 1] so that the
    max_distance threshold is resolution-independent (Doc 1 §2.4).
    """

    def __init__(self, max_distance=0.15, max_missed_frames=20):
        self.next_id = 0
        self.tracked = {}          # id -> (x, y) centroid, normalized
        self.missed_frames = {}    # id -> consecutive frames not matched
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames

    def update(self, detected_centroids):
        """
        detected_centroids: list of (x, y) tuples for this frame.
        Returns: dict mapping detected_centroid_index -> stable ID.
        """
        assignments = {}
        used_ids = set()

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
                new_id = self.next_id
                self.next_id += 1
                assignments[i] = new_id
                self.tracked[new_id] = (dx, dy)
                self.missed_frames[new_id] = 0
                used_ids.add(new_id)

        # Age out anyone not matched this frame; drop if missed too long
        for pid in list(self.tracked.keys()):
            if pid not in used_ids:
                self.missed_frames[pid] = self.missed_frames.get(pid, 0) + 1
                if self.missed_frames[pid] > self.max_missed_frames:
                    del self.tracked[pid]
                    del self.missed_frames[pid]

        return assignments
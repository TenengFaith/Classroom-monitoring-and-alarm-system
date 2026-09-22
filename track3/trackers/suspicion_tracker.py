class StudentSuspicionTracker:
    def __init__(self, threshold=0.6, required_frames=3):
        self.threshold = threshold
        self.required_frames = required_frames
        self.debounce_counters = {}

    def compute_score(self, head_yaw_deg, torso_lean_deg, object_near_hand):
        score = 0.0
        if head_yaw_deg is not None and abs(head_yaw_deg) > 30:
            score += 0.3
        if torso_lean_deg is not None and abs(torso_lean_deg) > 20:
            score += 0.3
        if object_near_hand:
            score += 0.4
        return score

    def update(self, student_id, head_yaw_deg, torso_lean_deg, object_near_hand):
        no_signal = (head_yaw_deg is None and torso_lean_deg is None)

        if no_signal:
            count = self.debounce_counters.get(student_id, 0)
            count = max(0, count - 1)
            self.debounce_counters[student_id] = count
            return 0.0, count >= self.required_frames

        score = self.compute_score(head_yaw_deg, torso_lean_deg, object_near_hand)
        is_over = score >= self.threshold

        count = self.debounce_counters.get(student_id, 0)
        count = count + 1 if is_over else 0
        self.debounce_counters[student_id] = count

        return score, count >= self.required_frames
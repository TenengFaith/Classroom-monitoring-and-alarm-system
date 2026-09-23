class StudentSuspicionTracker:
    """
    Maintains a debounced suspicion score per tracked student ID,
    combining head yaw, torso lean, and object proximity signals.

    Weights:
        head_yaw > 30 deg   -> +0.3
        torso_lean > 20 deg -> +0.3
        object_near_hand    -> +0.4
    An alert triggers only when the score stays >= threshold for
    `required_frames` consecutive frames for that student.
    """

    def __init__(self, threshold=0.6, required_frames=10):
        self.threshold = threshold
        self.required_frames = required_frames
        self.debounce_counters = {}   # student_id -> consecutive frames over threshold
        self.scores = {}              # student_id -> cumulative suspicion score

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
        score = self.compute_score(head_yaw_deg, torso_lean_deg, object_near_hand)
        is_over_threshold = score >= self.threshold

        count = self.debounce_counters.get(student_id, 0)
        count = count + 1 if is_over_threshold else 0
        self.debounce_counters[student_id] = count

        should_alert = count >= self.required_frames
        return score, should_alert

class StudentSuspicionTracker:
    """
    Maintains a debounced suspicion score per tracked student ID,
    combining head yaw, torso lean, and object proximity signals.

    Weights (must sum roughly to the threshold scale):
        head_yaw > 30 deg   -> +0.3
        torso_lean > 20 deg -> +0.3
        object_near_hand    -> +0.4
    An alert fires only when the score stays >= threshold for
    `required_frames` consecutive frames for that student.
    """

    def __init__(self, threshold=0.6, required_frames=15):
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

    def update_student_suspicion(self, student_id, yaw_score, lean_score):
        current_score = self.scores.get(student_id, 0.0)
        
        # Weight metrics: yaw (head turned) and lean
        frame_suspicion = (yaw_score * 0.7) + (lean_score * 0.3)
        
        if frame_suspicion > 0.3:
            # Accumulate score quickly when looking away
            current_score = min(1.0, current_score + 0.08)
        else:
            # Decay score slowly when looking forward
            current_score = max(0.0, current_score - 0.02)
            
        self.scores[student_id] = current_score

        # Trigger Alarm Call
        if current_score >= self.threshold:  # e.g., 0.50
            if hasattr(self, 'trigger_alarm'):
                self.trigger_alarm(student_id)
            
        return current_score
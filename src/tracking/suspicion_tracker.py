class StudentSuspicionTracker:
    """
    Maintains debounced suspicion scores per student ID based on exam hall cheating heuristics:
    1. Looking Under Desk / Lap: High downward head pitch (> 25 deg).
    2. Communicating / Talking: Turning head + open mouth ratio (MAR > 0.12).
    3. Turning Head: Turning head sideways towards peer (> 30 deg).
    4. Mobile Phone / Material: Phone or unauthorized material detected in hand/crop.
    """

    def __init__(self, threshold=0.55, required_frames=8):
        self.threshold = threshold
        self.required_frames = required_frames
        self.debounce_counters = {}   # student_id -> consecutive frames over threshold
        self.last_reason = {}         # student_id -> cheat reason string

    def evaluate_behavior(self, head_pitch_deg, head_yaw_deg, mouth_open_ratio, torso_lean_deg, object_near_hand):
        score = 0.0
        reasons = []

        # Vector 1: Looking Under Desk / Lap (Head bent down)
        if head_pitch_deg is not None and head_pitch_deg > 25.0:
            score += 0.4
            reasons.append("Looking Under Desk / Lap")
        elif head_pitch_deg is not None and head_pitch_deg > 18.0 and torso_lean_deg is not None and abs(torso_lean_deg) > 15.0:
            score += 0.35
            reasons.append("Looking Down & Leaning")

        # Vector 2: Communicating / Talking to Neighbor
        if head_yaw_deg is not None and abs(head_yaw_deg) > 18.0 and mouth_open_ratio is not None and mouth_open_ratio > 0.12:
            score += 0.4
            reasons.append("Communicating / Talking")

        # Vector 3: Turning Head Sideways towards Peer
        if head_yaw_deg is not None and abs(head_yaw_deg) > 30.0:
            score += 0.3
            if "Communicating / Talking" not in reasons:
                reasons.append("Turning Head to Peer")

        # Vector 4: Using Mobile Phone / Unauthorized Materials
        if object_near_hand:
            score += 0.45
            reasons.append("Mobile Phone / Material Detected")

        primary_reason = " | ".join(reasons) if reasons else "Normal"
        return min(1.0, score), primary_reason

    def update(self, student_id, head_pitch_deg, head_yaw_deg, mouth_open_ratio, torso_lean_deg, object_near_hand):
        score, reason = self.evaluate_behavior(
            head_pitch_deg=head_pitch_deg,
            head_yaw_deg=head_yaw_deg,
            mouth_open_ratio=mouth_open_ratio,
            torso_lean_deg=torso_lean_deg,
            object_near_hand=object_near_hand
        )

        is_over_threshold = score >= self.threshold

        count = self.debounce_counters.get(student_id, 0)
        count = count + 1 if is_over_threshold else 0
        self.debounce_counters[student_id] = count

        if is_over_threshold:
            self.last_reason[student_id] = reason

        should_alert = count >= self.required_frames
        active_reason = self.last_reason.get(student_id, reason) if should_alert else reason

        return score, should_alert, active_reason

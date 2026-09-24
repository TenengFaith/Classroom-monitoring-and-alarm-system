class StudentSuspicionTracker:
    """
    Maintains debounced suspicion scores per student ID based on exam hall cheating heuristics:
    1. Phone / Material Below Desk: Hands positioned in lap / under desk while head pitch is tilted down.
    2. Leaning Sideways Away from Desk: Shoulder tilt / torso lean sideways towards neighbor (> 14 deg).
    3. Turning Head to Neighbor: Turning head sideways (> 22 deg).
    4. Communicating / Talking: Turning head + open mouth ratio (MAR > 0.12).
    
    EXCLUDES normal exam activities:
    - Writing on exam paper with hands on desk.
    - Standard desk items (IDs, calculators, pens on desk).
    """

    def __init__(self, threshold=0.50, required_frames=6):
        self.threshold = threshold
        self.required_frames = required_frames
        self.debounce_counters = {}   # student_id -> consecutive frames over threshold
        self.last_reason = {}         # student_id -> cheat reason string

    def evaluate_behavior(self, head_pitch_deg, head_yaw_deg, mouth_open_ratio, torso_lean_deg, hands_under_desk, object_near_hand):
        score = 0.0
        reasons = []

        # Vector 1: Using Phone / Material Below Desk (Hands in lap under desk + head looking down into lap)
        if (hands_under_desk and head_pitch_deg is not None and head_pitch_deg > 14.0) or (hands_under_desk and object_near_hand):
            score += 0.65
            reasons.append("Using Phone / Material Below Desk")
        elif object_near_hand:
            score += 0.55
            reasons.append("Mobile Phone / Unauthorized Material")
        elif head_pitch_deg is not None and head_pitch_deg > 28.0:
            score += 0.40
            reasons.append("Looking Down Under Desk")

        # Vector 2: Leaning Sideways Away from Desk (Leaning towards neighbor)
        if torso_lean_deg is not None and torso_lean_deg > 14.0:
            score += 0.50
            reasons.append("Leaning Sideways Away from Desk")

        # Vector 3: Turning Head to Neighbor
        if head_yaw_deg is not None and abs(head_yaw_deg) > 22.0:
            score += 0.35
            if "Communicating / Talking to Peer" not in reasons:
                reasons.append("Turning Head to Neighbor")

        # Vector 4: Communicating / Talking to Peer
        if head_yaw_deg is not None and abs(head_yaw_deg) > 15.0 and mouth_open_ratio is not None and mouth_open_ratio > 0.12:
            score += 0.45
            reasons.append("Communicating / Talking to Peer")

        primary_reason = " | ".join(reasons) if reasons else "Normal"
        return min(1.0, score), primary_reason

    def update(self, student_id, head_pitch_deg, head_yaw_deg, mouth_open_ratio, torso_lean_deg, hands_under_desk, object_near_hand):
        score, reason = self.evaluate_behavior(
            head_pitch_deg=head_pitch_deg,
            head_yaw_deg=head_yaw_deg,
            mouth_open_ratio=mouth_open_ratio,
            torso_lean_deg=torso_lean_deg,
            hands_under_desk=hands_under_desk,
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

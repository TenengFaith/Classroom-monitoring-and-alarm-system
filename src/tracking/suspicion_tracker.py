import time


class StudentSuspicionTracker:
    """
    Maintains debounced suspicion scores per student ID based on exam hall cheating heuristics:
    1. Phone / Material Below Desk: Hands in lap under desk while head pitch is tilted down.
    2. Leaning Away from Desk / Reaching:
       - DO NOT FLAG: Slight lean (< 22 deg) while writing with hands on desk.
       - FLAG: Extreme lean (> 22 deg) or moderate lean (> 15 deg) combined with hands moving away / reaching.
    3. Turning Head to Neighbor: Turning head sideways (> 22 deg) sustained for >= 3.0 continuous seconds.
    4. Communicating / Talking: Turning head + open mouth ratio (MAR > 0.12) sustained for >= 3.0 continuous seconds.
    """

    def __init__(self, threshold=0.50, required_frames=6, min_turn_sec=3.0):
        self.threshold = threshold
        self.required_frames = required_frames
        self.min_turn_sec = min_turn_sec
        self.debounce_counters = {}       # student_id -> consecutive frames over threshold
        self.last_reason = {}             # student_id -> cheat reason string
        self.head_turn_start_time = {}    # student_id -> float (start timestamp of head turn)

    def evaluate_behavior(self, student_id, head_pitch_deg, head_yaw_deg, mouth_open_ratio, torso_lean_deg,
                          hands_under_desk, hands_in_writing_pos, hands_moved_away, object_near_hand, current_time=None):
        current_time = current_time or time.time()
        score = 0.0
        reasons = []

        # 1. Head turn 3-second continuous duration check
        is_turning = (head_yaw_deg is not None and abs(head_yaw_deg) > 22.0)
        is_sustained_turn = False

        if is_turning:
            if student_id not in self.head_turn_start_time or self.head_turn_start_time[student_id] is None:
                self.head_turn_start_time[student_id] = current_time

            turn_duration = current_time - self.head_turn_start_time[student_id]
            if turn_duration >= self.min_turn_sec:
                is_sustained_turn = True
        else:
            self.head_turn_start_time[student_id] = None

        # Vector 1: Using Phone / Material Below Desk
        if (hands_under_desk and head_pitch_deg is not None and head_pitch_deg > 14.0) or (hands_under_desk and object_near_hand):
            score += 0.65
            reasons.append("Using Phone / Material Below Desk")
        elif object_near_hand:
            score += 0.55
            reasons.append("Mobile Phone / Unauthorized Material")
        elif head_pitch_deg is not None and head_pitch_deg > 28.0:
            score += 0.40
            reasons.append("Looking Down Under Desk")

        # Vector 2: Leaning Away from Desk / Reaching
        # Differentiates normal slight writing lean from suspicious leaning
        if torso_lean_deg is not None:
            # Extreme lean completely away from desk script
            if torso_lean_deg > 22.0:
                score += 0.50
                reasons.append("Severe Lean Away from Desk")
            # Moderate lean combined with moving hands away from desk (reaching) or looking at peer
            elif torso_lean_deg > 15.0 and (hands_moved_away or (head_yaw_deg is not None and abs(head_yaw_deg) > 18.0)):
                score += 0.45
                if hands_moved_away:
                    reasons.append("Leaning & Reaching Away from Desk")
                else:
                    reasons.append("Leaning Sideways to Peer")
            # Slight writing lean with hands on desk is NORMAL (Score = 0.0)

        # Vector 3: Turning Head / Communicating (ONLY flagged if turn lasts >= 3.0 seconds)
        if is_sustained_turn:
            score += 0.40
            if mouth_open_ratio is not None and mouth_open_ratio > 0.12:
                reasons.append("Communicating / Talking to Peer")
            else:
                reasons.append("Turning Head to Neighbor")

        primary_reason = " | ".join(reasons) if reasons else "Normal"
        return min(1.0, score), primary_reason

    def update(self, student_id, head_pitch_deg, head_yaw_deg, mouth_open_ratio, torso_lean_deg,
               hands_under_desk, hands_in_writing_pos, hands_moved_away, object_near_hand, current_time=None):

        score, reason = self.evaluate_behavior(
            student_id=student_id,
            head_pitch_deg=head_pitch_deg,
            head_yaw_deg=head_yaw_deg,
            mouth_open_ratio=mouth_open_ratio,
            torso_lean_deg=torso_lean_deg,
            hands_under_desk=hands_under_desk,
            hands_in_writing_pos=hands_in_writing_pos,
            hands_moved_away=hands_moved_away,
            object_near_hand=object_near_hand,
            current_time=current_time
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

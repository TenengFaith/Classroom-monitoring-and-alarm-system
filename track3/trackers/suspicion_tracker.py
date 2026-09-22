class StudentSuspicionTracker:
    def __init__(self, threshold=0.6, required_frames=3):
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

<<<<<<< HEAD
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
=======
        return score, count >= self.required_frames
>>>>>>> f8f98431b66f9dab4d4a2559d1bb33303a177957

import cv2

def draw_student_overlay(frame, bbox_norm, student_id, score=0.0, alert=False):
    h, w = frame.shape[:2]
    x1 = int(bbox_norm["x_min"] * w)
    y1 = int(bbox_norm["y_min"] * h)
    x2 = int(bbox_norm["x_max"] * w)
    y2 = int(bbox_norm["y_max"] * h)

    # Determine status label and color
    if alert or score >= 0.60:
        color = (0, 0, 255)      # Red
        status = "ALERT"
    elif score >= 0.30:
        color = (0, 165, 255)    # Orange
        status = "WATCH"
    else:
        color = (0, 255, 0)      # Green
        status = "OK"

    thickness = 2
    

    # 1. Draw Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # 2. Draw Red 'X' inside the box if suspicious score is high (>= 0.60 / alert)
    if alert or score >= 0.60:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness)
        cv2.line(frame, (x1, y2), (x2, y1), (0, 0, 255), thickness)

    # 3. Draw Text Label directly above or at top-left of the bounding box
    label = f"#{student_id} {status} score={score:.2f}"
    text_pos_y = max(y1 - 8, 20)
    
    cv2.putText(
        frame,
        label,
        (x1, text_pos_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA
    )


def draw_dashboard_hud(frame, fps, active_count, total_ids, has_alert=False):
    # Top overlay text matching screenshot style in Cyan (BGR: 255, 255, 0)
    hud_text = f"FPS: {fps:.1f}  IDs seen: {total_ids}"
    cv2.putText(
        frame,
        hud_text,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )

    if has_alert:
        cv2.putText(
            frame,
            "[!] HIGH SUSPICION DETECTED",
            (frame.shape[1] - 430, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
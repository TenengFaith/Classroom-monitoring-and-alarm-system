import math

# MediaPipe Pose landmark indices
NOSE           = 0
LEFT_EAR       = 7
RIGHT_EAR      = 8
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24

# MediaPipe Face Landmarker indices
NOSE_TIP       = 1
FOREHEAD       = 10
CHIN           = 152
UPPER_LIP      = 13
LOWER_LIP      = 14
LEFT_CHEEK     = 234
RIGHT_CHEEK    = 454


def _crop_lm_to_global(lm, crop_offset, crop_size, full_w, full_h):
    """Transforms landmarks from crop-local coordinates to full-frame normalized coordinates."""
    cx, cy = crop_offset
    cw, ch = crop_size

    pixel_x = cx + (lm["x"] * cw)
    pixel_y = cy + (lm["y"] * ch)

    return {
        "x": pixel_x / full_w,
        "y": pixel_y / full_h,
        "z": lm.get("z", 0.0),
        "visibility": lm.get("visibility", 1.0)
    }


def _hip_midpoint(landmarks):
    if len(landmarks) < 25:
        return None
    lh = landmarks[LEFT_HIP]
    rh = landmarks[RIGHT_HIP]
    return ((lh["x"] + rh["x"]) / 2.0, (lh["y"] + rh["y"]) / 2.0)


def _bbox_center(bbox):
    if not bbox:
        return None
    if isinstance(bbox, dict):
        return (
            (bbox["x_min"] + bbox["x_max"]) / 2.0,
            (bbox["y_min"] + bbox["y_max"]) / 2.0,
        )
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return (
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
        )
    return None


def _shoulder_tilt_and_lean(landmarks):
    """
    Calculates sideways torso lean / shoulder line tilt angle in degrees.
    Detects when a student leans sideways completely away from their desk towards a peer.
    """
    if not landmarks or len(landmarks) < 13:
        return 0.0

    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]

    vis_min = min(ls.get("visibility", 1.0), rs.get("visibility", 1.0))
    if vis_min < 0.3:
        return 0.0

    dx = ls["x"] - rs["x"]
    dy = ls["y"] - rs["y"]

    if abs(dx) < 1e-4:
        return 90.0

    # Shoulder tilt angle from horizontal line
    tilt_deg = math.degrees(math.atan2(abs(dy), abs(dx)))

    # Also check if hip landmarks are available for spine vector lean
    if len(landmarks) >= 25:
        lh = landmarks[LEFT_HIP]
        rh = landmarks[RIGHT_HIP]
        if min(lh.get("visibility", 1.0), rh.get("visibility", 1.0)) >= 0.3:
            sx = (ls["x"] + rs["x"]) / 2.0
            sy = (ls["y"] + rs["y"]) / 2.0
            hx = (lh["x"] + rh["x"]) / 2.0
            hy = (lh["y"] + rh["y"]) / 2.0
            spine_lean = math.degrees(math.atan2(abs(sx - hx), max(1e-4, abs(sy - hy))))
            return max(tilt_deg, spine_lean)

    return tilt_deg


def _head_yaw_deg_from_face(face_landmarks):
    if not face_landmarks or len(face_landmarks) < 455:
        return 0.0
    nose = face_landmarks[NOSE_TIP]
    lc = face_landmarks[LEFT_CHEEK]
    rc = face_landmarks[RIGHT_CHEEK]

    face_mid_x = (lc["x"] + rc["x"]) / 2.0
    face_width = abs(rc["x"] - lc["x"])
    if face_width < 1e-4:
        return 0.0

    offset = (nose["x"] - face_mid_x) / (face_width / 2.0)
    offset = max(-1.0, min(1.0, offset))
    return offset * 70.0


def _head_pitch_deg(face_landmarks, pose_landmarks):
    """
    Estimates head pitch angle in degrees (downward tilt).
    When student looks down into their lap / under desk:
    - Pitch angle is positive (> 20-30 deg).
    """
    if face_landmarks and len(face_landmarks) > 152:
        forehead = face_landmarks[FOREHEAD]
        nose = face_landmarks[NOSE_TIP]
        chin = face_landmarks[CHIN]

        face_h = abs(chin["y"] - forehead["y"])
        if face_h > 1e-4:
            forehead_to_nose = nose["y"] - forehead["y"]
            ratio = forehead_to_nose / face_h
            pitch_deg = max(0.0, (ratio - 0.42) * 160.0)
            return pitch_deg

    if pose_landmarks and len(pose_landmarks) > 8:
        nose = pose_landmarks[NOSE]
        lear = pose_landmarks[LEFT_EAR]
        rear = pose_landmarks[RIGHT_EAR]

        ear_y = (lear["y"] + rear["y"]) / 2.0
        ear_vis = min(lear.get("visibility", 1.0), rear.get("visibility", 1.0))
        if ear_vis >= 0.3:
            diff_y = nose["y"] - ear_y
            if diff_y > 0.04:
                return diff_y * 300.0

    return 0.0


def _mouth_openness_ratio(face_landmarks):
    if not face_landmarks or len(face_landmarks) < 153:
        return 0.0

    upper_lip = face_landmarks[UPPER_LIP]
    lower_lip = face_landmarks[LOWER_LIP]
    forehead = face_landmarks[FOREHEAD]
    chin = face_landmarks[CHIN]

    face_h = abs(chin["y"] - forehead["y"])
    if face_h < 1e-4:
        return 0.0

    lip_dist = abs(lower_lip["y"] - upper_lip["y"])
    return lip_dist / face_h


def _is_hands_under_desk(pose_landmarks):
    """
    Detects if student's hands/wrists are positioned in their lap area / below desk height.
    In crop coordinates (0..1 top to bottom), desk surface is at ~0.55-0.65.
    When hands are in lap under desk, wrist landmark y > 0.60 or wrists are below elbows.
    """
    if not pose_landmarks or len(pose_landmarks) < 17:
        return False

    lw = pose_landmarks[LEFT_WRIST]
    rw = pose_landmarks[RIGHT_WRIST]

    lw_vis = lw.get("visibility", 1.0)
    rw_vis = rw.get("visibility", 1.0)

    # Check if left or right wrist is in crop lower half (lap region below desk)
    left_under = (lw_vis >= 0.3 and lw["y"] > 0.62)
    right_under = (rw_vis >= 0.3 and rw["y"] > 0.62)

    # Also check if wrist is significantly lower than elbow (hanging down into lap)
    if len(pose_landmarks) >= 15:
        le = pose_landmarks[LEFT_ELBOW]
        re = pose_landmarks[RIGHT_ELBOW]

        if lw_vis >= 0.3 and le.get("visibility", 1.0) >= 0.3 and (lw["y"] - le["y"]) > 0.12:
            left_under = True
        if rw_vis >= 0.3 and re.get("visibility", 1.0) >= 0.3 and (rw["y"] - re["y"]) > 0.12:
            right_under = True

    return left_under or right_under


def adapt(structured_output, full_w=1280, full_h=720):
    poses = structured_output.get("poses", [])
    objects = structured_output.get("objects", [])
    records = []

    has_detected_phone = len(objects) > 0

    for pose in poses:
        bbox = pose.get("bbox")
        raw_pose_lm = pose.get("landmarks", [])
        raw_face_lm = pose.get("face_landmarks", [])
        crop_offset = pose.get("crop_offset", (0, 0))
        crop_size = pose.get("crop_size", (1, 1))

        global_pose_lm = [
            _crop_lm_to_global(lm, crop_offset, crop_size, full_w, full_h)
            for lm in raw_pose_lm
        ]

        global_face_lm = [
            _crop_lm_to_global(lm, crop_offset, crop_size, full_w, full_h)
            for lm in raw_face_lm
        ]

        if global_pose_lm:
            centroid = _hip_midpoint(global_pose_lm)
        else:
            centroid = _bbox_center(bbox)

        if centroid is None:
            continue

        head_yaw = _head_yaw_deg_from_face(global_face_lm) if global_face_lm else 0.0
        head_pitch = _head_pitch_deg(global_face_lm, global_pose_lm)
        mouth_ratio = _mouth_openness_ratio(global_face_lm) if global_face_lm else 0.0
        torso_lean = _shoulder_tilt_and_lean(raw_pose_lm) if raw_pose_lm else 0.0
        hands_under = _is_hands_under_desk(raw_pose_lm) if raw_pose_lm else False

        records.append({
            "centroid": centroid,
            "head_yaw_deg": head_yaw,
            "head_pitch_deg": head_pitch,
            "mouth_open_ratio": mouth_ratio,
            "torso_lean_deg": torso_lean,
            "hands_under_desk": hands_under,
            "object_near_hand": has_detected_phone,
            "pose_id": pose.get("pose_id"),
            "bbox": bbox,
        })

    return records

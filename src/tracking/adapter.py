import math

# MediaPipe Pose landmark indices
NOSE           = 0
LEFT_EAR       = 7
RIGHT_EAR      = 8
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
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
LEFT_EYE       = 33
RIGHT_EYE      = 263


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


def _torso_lean_deg(landmarks):
    if len(landmarks) < 25:
        return None
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    lh = landmarks[LEFT_HIP]
    rh = landmarks[RIGHT_HIP]

    vis_min = min(ls.get("visibility", 1.0), rs.get("visibility", 1.0),
                  lh.get("visibility", 1.0), rh.get("visibility", 1.0))
    if vis_min < 0.3:
        return None

    sx = (ls["x"] + rs["x"]) / 2.0
    sy = (ls["y"] + rs["y"]) / 2.0
    hx = (lh["x"] + rh["x"]) / 2.0
    hy = (lh["y"] + rh["y"]) / 2.0

    return math.degrees(math.atan2(sx - hx, -(sy - hy)))


def _head_yaw_deg_from_face(face_landmarks):
    if not face_landmarks or len(face_landmarks) < 455:
        return None
    nose = face_landmarks[NOSE_TIP]
    lc = face_landmarks[LEFT_CHEEK]
    rc = face_landmarks[RIGHT_CHEEK]

    face_mid_x = (lc["x"] + rc["x"]) / 2.0
    face_width = abs(rc["x"] - lc["x"])
    if face_width < 1e-4:
        return None

    offset = (nose["x"] - face_mid_x) / (face_width / 2.0)
    offset = max(-1.0, min(1.0, offset))
    return offset * 70.0


def _head_pitch_deg(face_landmarks, pose_landmarks):
    """
    Estimates head pitch angle in degrees (downward tilt).
    When student looks down into their lap / under desk:
    - Pitch angle is positive (> 25-35 deg).
    """
    if face_landmarks and len(face_landmarks) > 152:
        forehead = face_landmarks[FOREHEAD]
        nose = face_landmarks[NOSE_TIP]
        chin = face_landmarks[CHIN]

        face_h = abs(chin["y"] - forehead["y"])
        if face_h > 1e-4:
            # Measure relative position of nose tip between forehead and chin
            forehead_to_nose = nose["y"] - forehead["y"]
            nose_to_chin = chin["y"] - nose["y"]

            # In normal upright face, forehead_to_nose is ~40-45% of face height.
            # When looking down at lap / under desk, nose drops relative to forehead
            # and forehead_to_nose increases significantly.
            ratio = forehead_to_nose / face_h
            # Pitch estimation mapping: ratio 0.40 -> 0 deg, 0.70 -> 50 deg
            pitch_deg = max(0.0, (ratio - 0.42) * 160.0)
            return pitch_deg

    # Fallback to pose landmarks if face Mesh not available
    if pose_landmarks and len(pose_landmarks) > 8:
        nose = pose_landmarks[NOSE]
        lear = pose_landmarks[LEFT_EAR]
        rear = pose_landmarks[RIGHT_EAR]

        ear_y = (lear["y"] + rear["y"]) / 2.0
        ear_vis = min(lear.get("visibility", 1.0), rear.get("visibility", 1.0))
        if ear_vis >= 0.3:
            diff_y = nose["y"] - ear_y
            if diff_y > 0.05:  # Nose significantly below ears indicates head bent down
                return diff_y * 300.0

    return 0.0


def _mouth_openness_ratio(face_landmarks):
    """
    Estimates Mouth Openness Ratio (MAR) for detecting talking / communicating.
    MAR > 0.15 indicates open mouth / speaking.
    """
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


def adapt(structured_output, full_w=1280, full_h=720):
    """
    Converts detection pipeline structured output into tracking person records.
    Computes pitch (looking under desk), yaw (turning), mouth openness (talking),
    torso lean, and hand proximity.
    """
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
        torso_lean = _torso_lean_deg(global_pose_lm) if global_pose_lm else 0.0

        records.append({
            "centroid": centroid,
            "head_yaw_deg": head_yaw,
            "head_pitch_deg": head_pitch,
            "mouth_open_ratio": mouth_ratio,
            "torso_lean_deg": torso_lean,
            "object_near_hand": has_detected_phone,
            "pose_id": pose.get("pose_id"),
            "bbox": bbox,
        })

    return records

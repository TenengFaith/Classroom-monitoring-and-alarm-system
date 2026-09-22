import math

# MediaPipe Pose landmark indices
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_HIP       = 23
RIGHT_HIP      = 24

# MediaPipe Face Landmarker indices
NOSE_TIP    = 1
LEFT_CHEEK  = 234
RIGHT_CHEEK = 454


def _crop_lm_to_global(lm, crop_offset, crop_size, full_w, full_h):
    """
    Transforms landmarks from crop-local coordinates (0..1 within crop)
    to full-frame normalized coordinates (0..1 within full frame).
    """
    cx, cy = crop_offset
    cw, ch = crop_size
    
    # Calculate pixel position within full image
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


def extract_head_yaw(pose_landmarks, face_landmarks):
    # Try high-precision face landmarker first
    if face_landmarks and len(face_landmarks) > 0:
        # Calculate yaw using nose tip (1) vs left cheek (234) & right cheek (454)
        nose = face_landmarks[1]
        left_cheek = face_landmarks[234]
        right_cheek = face_landmarks[454]
        mid_x = (left_cheek.x + right_cheek.x) / 2
        width = abs(right_cheek.x - left_cheek.x)
        if width > 0:
            return abs(nose.x - mid_x) / width  # Normalized offset

    # Fallback to pose landmarks if face mesh missed the turned face
    if pose_landmarks and len(pose_landmarks) > 0:
        left_ear = pose_landmarks[7]
        right_ear = pose_landmarks[8]
        nose = pose_landmarks[0]
        
        # If one ear is hidden or visibility is low, student is turned sideways
        if left_ear.visibility < 0.3 or right_ear.visibility < 0.3:
            return 0.75  # Force high yaw indicator for turned heads

    return 0.0


def adapt(structured_output, full_w=1280, full_h=720):
    """
    Converts Track 2 structured output into Track 3 person records.
    Maps cropped landmarks back into full-frame normalized coordinates.
    """
    poses = structured_output.get("poses", [])
    records = []

    for pose in poses:
        bbox = pose.get("bbox")
        raw_pose_lm = pose.get("landmarks", [])
        raw_face_lm = pose.get("face_landmarks", [])
        crop_offset = pose.get("crop_offset", (0, 0))
        crop_size = pose.get("crop_size", (1, 1))

        # Transform landmarks to full-frame normalized coordinates
        global_pose_lm = [
            _crop_lm_to_global(lm, crop_offset, crop_size, full_w, full_h)
            for lm in raw_pose_lm
        ]
        
        global_face_lm = [
            _crop_lm_to_global(lm, crop_offset, crop_size, full_w, full_h)
            for lm in raw_face_lm
        ]

        # Calculate centroid (Hip midpoint preferred, fallback to bbox center)
        if global_pose_lm:
            centroid = _hip_midpoint(global_pose_lm)
        else:
            centroid = _bbox_center(bbox)

        if centroid is None:
            continue

        head_yaw = _head_yaw_deg_from_face(global_face_lm) if global_face_lm else None
        torso_lean = _torso_lean_deg(global_pose_lm) if global_pose_lm else None

        records.append({
            "centroid": centroid,
            "head_yaw_deg": head_yaw,
            "torso_lean_deg": torso_lean,
            "object_near_hand": False,
            "pose_id": pose.get("pose_id"),
            "bbox": bbox,
        })

    return records
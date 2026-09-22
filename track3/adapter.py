import math


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_WRIST = 15
RIGHT_WRIST = 16

NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454


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


def _crop_lm_to_global(lm, crop_offset, crop_size, full_w, full_h):
    cx, cy = crop_offset
    cw, ch = crop_size
    pixel_x = cx + (lm["x"] * cw)
    pixel_y = cy + (lm["y"] * ch)
    return {
        "x": pixel_x / full_w,
        "y": pixel_y / full_h,
        "z": lm.get("z", 0.0),
        "visibility": lm.get("visibility", 1.0),
    }


def _torso_lean_deg(landmarks):
    if len(landmarks) < 25:
        return None
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    lh = landmarks[LEFT_HIP]
    rh = landmarks[RIGHT_HIP]

    vis_min = min(
        ls.get("visibility", 1.0),
        rs.get("visibility", 1.0),
        lh.get("visibility", 1.0),
        rh.get("visibility", 1.0),
    )
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


def _bboxes_overlap(a, b, margin=0.03):
    ax1, ay1, ax2, ay2 = a["x_min"], a["y_min"], a["x_max"], a["y_max"]
    bx1, by1, bx2, by2 = b["x_min"], b["y_min"], b["x_max"], b["y_max"]
    return not (
        ax2 + margin < bx1
        or bx2 + margin < ax1
        or ay2 + margin < by1
        or by2 + margin < ay1
    )


def _object_near_pose(objects, pose_bbox, pose_landmarks,
                      crop_offset, crop_size, full_w, full_h,
                      wrist_radius=0.06):
    if not objects or not pose_bbox:
        return False

    test_regions = [pose_bbox]

    for idx in (LEFT_WRIST, RIGHT_WRIST):
        if idx < len(pose_landmarks):
            lm = pose_landmarks[idx]
            gx = (crop_offset[0] + lm["x"] * crop_size[0]) / full_w
            gy = (crop_offset[1] + lm["y"] * crop_size[1]) / full_h
            test_regions.append({
                "x_min": gx - wrist_radius,
                "y_min": gy - wrist_radius,
                "x_max": gx + wrist_radius,
                "y_max": gy + wrist_radius,
            })

    for obj in objects:
        obj_bbox = obj.get("bbox")
        if not obj_bbox:
            continue
        obj_type = obj.get("type", "")
        if obj_type not in ("cell phone", "phone", "book", "remote"):
            continue
        for region in test_regions:
            if _bboxes_overlap(region, obj_bbox):
                return True

    return False


def adapt(structured_output, full_w=1280, full_h=720):
    poses = structured_output.get("poses", [])
    objects = structured_output.get("objects", [])
    records = []

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

        head_yaw = _head_yaw_deg_from_face(global_face_lm) if global_face_lm else None
        torso_lean = _torso_lean_deg(global_pose_lm) if global_pose_lm else None

        object_near_hand = _object_near_pose(
            objects,
            bbox,
            raw_pose_lm,
            crop_offset,
            crop_size,
            full_w,
            full_h,
        )

        records.append({
            "centroid": centroid,
            "head_yaw_deg": head_yaw,
            "torso_lean_deg": torso_lean,
            "object_near_hand": object_near_hand,
            "pose_id": pose.get("pose_id"),
            "bbox": bbox,
        })

    return records
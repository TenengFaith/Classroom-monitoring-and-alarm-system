"""
Track 2 -> Track 3 adapter.

Converts Faith's ClassroomDetectionPipeline.process_frame() output
into a list of Track 3 person records:

    {
        "centroid": (x_norm, y_norm),   # hip-midpoint, normalized [0,1]
        "head_yaw_deg": float | None,
        "torso_lean_deg": float | None,
        "object_near_hand": bool,       # STUB for now
        "pose_id": int,
        "face_id": int | None,
        "bbox": dict,
    }
"""

import math

LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_HIP       = 23
RIGHT_HIP      = 24

NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454


def _hip_midpoint(landmarks):
    if len(landmarks) < 25:
        return None
    lh = landmarks[LEFT_HIP]
    rh = landmarks[RIGHT_HIP]
    return ((lh["x"] + rh["x"]) / 2.0, (lh["y"] + rh["y"]) / 2.0)


def _torso_lean_deg(landmarks):
    if len(landmarks) < 25:
        return None
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    lh = landmarks[LEFT_HIP]
    rh = landmarks[RIGHT_HIP]
    if min(ls["visibility"], rs["visibility"],
           lh["visibility"], rh["visibility"]) < 0.5:
        return None
    sx = (ls["x"] + rs["x"]) / 2.0
    sy = (ls["y"] + rs["y"]) / 2.0
    hx = (lh["x"] + rh["x"]) / 2.0
    hy = (lh["y"] + rh["y"]) / 2.0
    dx = sx - hx
    dy = sy - hy
    return math.degrees(math.atan2(dx, -dy))


def _head_yaw_deg_from_face(face_landmarks):
    if not face_landmarks or len(face_landmarks) < 468:
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


def _shoulder_mid(landmarks):
    if len(landmarks) < 13:
        return None
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    return ((ls["x"] + rs["x"]) / 2.0, (ls["y"] + rs["y"]) / 2.0)


def _face_center(landmarks):
    if not landmarks:
        return None
    return (landmarks[NOSE_TIP]["x"], landmarks[NOSE_TIP]["y"])


def adapt(structured_output):
    poses = structured_output.get("poses", [])
    faces = structured_output.get("faces", [])

    face_centers = [(f["face_id"], _face_center(f["landmarks"])) for f in faces]
    face_by_id = {f["face_id"]: f for f in faces}

    records = []
    for pose in poses:
        lm = pose["landmarks"]
        centroid = _hip_midpoint(lm)
        if centroid is None:
            continue

        sm = _shoulder_mid(lm)
        matched_face_id = None
        if sm is not None:
            best = None
            best_d = 0.25
            for fid, fc in face_centers:
                if fc is None:
                    continue
                d = ((sm[0] - fc[0]) ** 2 + (sm[1] - fc[1]) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = fid, d
            matched_face_id = best

        head_yaw = None
        if matched_face_id is not None:
            head_yaw = _head_yaw_deg_from_face(
                face_by_id[matched_face_id]["landmarks"]
            )

        records.append({
            "centroid": centroid,
            "head_yaw_deg": head_yaw,
            "torso_lean_deg": _torso_lean_deg(lm),
            "object_near_hand": False,
            "pose_id": pose["pose_id"],
            "face_id": matched_face_id,
            "bbox": pose["bbox"],
        })

    return records

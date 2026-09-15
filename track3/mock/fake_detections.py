"""
Deterministic fake detections for testing trackers without a camera.
"""

def _rec(centroid, yaw=None, lean=None, obj=False):
    return {
        "centroid": centroid,
        "head_yaw_deg": yaw,
        "torso_lean_deg": lean,
        "object_near_hand": obj,
        "pose_id": 0,
        "face_id": None,
        "bbox": {"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0},
    }


def scenario_single_person_idle(n_frames=30):
    for _ in range(n_frames):
        yield [_rec((0.5, 0.5))]


def scenario_two_people_static(n_frames=30):
    for _ in range(n_frames):
        yield [_rec((0.3, 0.5)), _rec((0.7, 0.5))]


def scenario_person_leaves_and_returns(gap_frames=25):
    for _ in range(5):
        yield [_rec((0.5, 0.5))]
    for _ in range(gap_frames):
        yield []
    for _ in range(5):
        yield [_rec((0.5, 0.5))]


def scenario_two_people_crossing(n_frames=40):
    for i in range(n_frames):
        t = i / (n_frames - 1)
        yield [_rec((0.3 + 0.4 * t, 0.5)), _rec((0.7 - 0.4 * t, 0.5))]


def scenario_head_turn_only(n_frames=30):
    for _ in range(n_frames):
        yield [_rec((0.5, 0.5), yaw=45.0)]


def scenario_lean_only(n_frames=30):
    for _ in range(n_frames):
        yield [_rec((0.5, 0.5), lean=30.0)]


def scenario_yaw_and_lean(n_frames=30):
    for _ in range(n_frames):
        yield [_rec((0.5, 0.5), yaw=45.0, lean=30.0)]


def scenario_yaw_only_brief_flash(n_frames=30):
    for i in range(n_frames):
        if i < 3:
            yield [_rec((0.5, 0.5), yaw=45.0)]
        else:
            yield [_rec((0.5, 0.5))]

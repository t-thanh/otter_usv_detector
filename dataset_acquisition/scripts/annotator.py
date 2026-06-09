#!/usr/bin/env python3
"""
annotator.py
────────────
Projects the Otter USV's AR-marker corners from world frame onto the camera
image plane and returns a YOLO OBB label string.

YOLO OBB label format (Ultralytics):
    class_id  x1 y1  x2 y2  x3 y3  x4 y4
where (xi, yi) are the four corner coordinates normalised to [0, 1].
class_id = 0 (single class: otter_usv).

Coordinate chain for the overhead camera URDF
──────────────────────────────────────────────
  base_link  ──RPY(0,π/2,0)──▶  cam_link  ──RPY(−π/2,0,−π/2)──▶  cam_optical

base_link world orientation = Rz(cam_yaw) · Ry(−cam_tilt)
  cam_tilt = 0  → pure yaw → camera points nadir (original behaviour)
  cam_tilt > 0  → gimbal tilts away from nadir toward the cam_yaw direction
                  (Gremsy G-Hadron controlled tilt range: ±120°; we use 0–45°)
"""

from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation


# ─────────────────────────────────────────────────────────────────────────────
# Camera intrinsics
# ─────────────────────────────────────────────────────────────────────────────

def compute_K(width_px: int, height_px: int, hfov_deg: float) -> np.ndarray:
    """Build 3×3 intrinsic matrix from sensor specs."""
    hfov_rad = np.radians(hfov_deg)
    fx = (width_px / 2.0) / np.tan(hfov_rad / 2.0)
    fy = fx                          # square pixels
    cx = width_px  / 2.0
    cy = height_px / 2.0
    return np.array([[fx, 0,  cx],
                     [0,  fy, cy],
                     [0,  0,   1]], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Rotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_world_to_optical(cam_yaw_rad: float,
                            cam_tilt_rad: float = 0.0) -> Rotation:
    """
    Returns the scipy Rotation that transforms a vector expressed in world
    frame into the camera optical frame.

    base_link world orientation = Rz(yaw) · Ry(−tilt)
      cam_tilt_rad=0 reproduces the original nadir-only behaviour.

    Full URDF chain:
        R_optical_in_world = [Rz(yaw)·Ry(−tilt)] · Ry(π/2) · (Rz(−π/2)·Rx(−π/2))
                           =  R_base              · R_joint1 · R_joint2

    Returns R such that R.apply(v_world) = v_optical  (world → optical).
    """
    R_base   = (Rotation.from_euler('xyz', [0.0, 0.0, cam_yaw_rad]) *
                Rotation.from_euler('xyz', [0.0, -cam_tilt_rad, 0.0]))   # Ry(−tilt)
    R_joint1 = Rotation.from_euler('xyz', [0.0, np.pi / 2.0, 0.0])
    R_joint2 = Rotation.from_euler('xyz', [-np.pi / 2.0, 0.0, -np.pi / 2.0])

    R_optical_in_world = R_base * R_joint1 * R_joint2
    return R_optical_in_world.inv()   # world → optical


def backproject_pixel_to_ground(u: float, v: float,
                                 K: np.ndarray,
                                 cam_pos: np.ndarray,
                                 cam_yaw_rad: float,
                                 cam_tilt_rad: float,
                                 ground_z: float) -> Optional[tuple]:
    """
    Back-project pixel (u, v) through the tilted camera to find the world (x, y)
    position where the ray intersects the horizontal plane z = ground_z.

    Used during randomisation so the AR marker centre lands at the desired pixel,
    giving a uniform spatial distribution of bounding boxes across the image.

    Parameters
    ----------
    u, v         : target pixel coordinates
    K            : 3×3 camera intrinsic matrix
    cam_pos      : (3,) camera world position [x, y, z]
    cam_yaw_rad  : camera yaw in world frame
    cam_tilt_rad : camera tilt from nadir (0 = straight down)
    ground_z     : world Z of the target plane (AR marker height = 0.7 m)

    Returns
    -------
    (world_x, world_y) or None if the ray points upward.
    Within tilt 0–45° and vFOV ≈ 52°, this never returns None.
    """
    R_w2opt = build_world_to_optical(cam_yaw_rad, cam_tilt_rad)
    R_opt2w = R_w2opt.inv()   # optical → world

    # Ray direction in optical frame (unnormalised, homogeneous z=1)
    K_inv = np.linalg.inv(K)
    r_opt = K_inv @ np.array([u, v, 1.0])

    # Rotate ray to world frame
    r_world = R_opt2w.apply(r_opt)

    # Intersect ray with z = ground_z:  cam_pos + t·r_world,  z-component = ground_z
    if r_world[2] >= 0.0:
        return None   # Ray points upward — no ground intersection
    t = (cam_pos[2] - ground_z) / (-r_world[2])
    hit = cam_pos + t * r_world
    return float(hit[0]), float(hit[1])


# ─────────────────────────────────────────────────────────────────────────────
# AR marker corners in Otter local frame
# ─────────────────────────────────────────────────────────────────────────────

def ar_marker_corners_local(cx: float, cy: float, cz: float,
                             half_size: float) -> np.ndarray:
    """
    Returns 4×3 array of AR-marker corners in the Otter base_link frame.
    Corner order: (+x+y), (+x−y), (−x−y), (−x+y)  — consistent winding.

    Source: otter_base.urdf.xacro
        <origin xyz="-0.15 0 0.7"/>  scale="2 2 0.05"
        → cx=-0.15, cy=0.0, cz=0.7, half_size=1.0
    """
    dx, dy = half_size, half_size
    return np.array([
        [cx + dx,  cy + dy, cz],   # front-right  (port)
        [cx + dx,  cy - dy, cz],   # front-left   (starboard)
        [cx - dx,  cy - dy, cz],   # rear-left
        [cx - dx,  cy + dy, cz],   # rear-right
    ], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Core annotation
# ─────────────────────────────────────────────────────────────────────────────

def project_corners(corners_local: np.ndarray,
                    otter_pos: np.ndarray,
                    otter_quat_xyzw: np.ndarray,
                    cam_pos: np.ndarray,
                    cam_yaw_rad: float,
                    cam_tilt_rad: float,
                    K: np.ndarray) -> Optional[np.ndarray]:
    """
    Project AR-marker corners from Otter local frame to image pixel coordinates.

    Parameters
    ----------
    corners_local    : (4,3) corners in Otter base_link frame
    otter_pos        : (3,)  Otter world position (from get_model_state)
    otter_quat_xyzw  : (4,)  Otter orientation quaternion [x,y,z,w]
    cam_pos          : (3,)  Camera base_link world position
    cam_yaw_rad      : float Camera yaw
    cam_tilt_rad     : float Camera tilt from nadir (0 = straight down)
    K                : (3,3) Camera intrinsic matrix

    Returns
    -------
    (4,2) pixel coordinates (u, v), or None if any corner is behind the camera.
    """
    # 1. Otter local → world
    R_otter = Rotation.from_quat(otter_quat_xyzw)
    corners_world = R_otter.apply(corners_local) + otter_pos

    # 2. World → camera optical frame
    R_w2opt     = build_world_to_optical(cam_yaw_rad, cam_tilt_rad)
    p_rel       = corners_world - cam_pos
    corners_cam = R_w2opt.apply(p_rel)

    # 3. Depth check — all corners must be in front of the camera
    if np.any(corners_cam[:, 2] <= 0):
        return None

    # 4. Perspective projection
    u = K[0, 0] * corners_cam[:, 0] / corners_cam[:, 2] + K[0, 2]
    v = K[1, 1] * corners_cam[:, 1] / corners_cam[:, 2] + K[1, 2]
    return np.stack([u, v], axis=1)


def make_yolo_obb_label(pixels: np.ndarray,
                         img_w: int, img_h: int,
                         class_id: int = 0,
                         border_margin: int = 5,
                         min_corners: int = 4) -> Optional[str]:
    """
    Convert (4,2) pixel coordinates to a YOLO OBB label line.
    Returns None if too few corners are inside the valid image area.
    """
    margin = border_margin
    in_frame = ((pixels[:, 0] >= margin) & (pixels[:, 0] <= img_w - margin) &
                (pixels[:, 1] >= margin) & (pixels[:, 1] <= img_h - margin))
    if in_frame.sum() < min_corners:
        return None
    norm   = pixels / np.array([img_w, img_h], dtype=np.float64)
    coords = " ".join(f"{v:.6f}" for v in norm.flatten())
    return f"{class_id} {coords}"


def project_corners_fk(corners_local: np.ndarray,
                        otter_pos: np.ndarray,
                        otter_quat_xyzw: np.ndarray,
                        p_opt: np.ndarray,
                        R_opt_to_world,
                        K: np.ndarray) -> Optional[np.ndarray]:
    """
    Like project_corners() but takes the full gimbal FK result directly
    instead of the simple cam_yaw/cam_tilt model.

    Parameters
    ----------
    p_opt          : (3,) camera optical centre in world frame
    R_opt_to_world : scipy Rotation — optical frame → world frame
                     (as returned by _camera_fk / _camera_optical_world_pose)

    Returns
    -------
    (4,2) pixel coordinates or None if any corner is behind the camera.
    """
    R_otter       = Rotation.from_quat(otter_quat_xyzw)
    corners_world = R_otter.apply(corners_local) + otter_pos

    R_world_to_opt = R_opt_to_world.inv()
    corners_cam    = R_world_to_opt.apply(corners_world - p_opt)

    if np.any(corners_cam[:, 2] <= 0):
        return None

    u = K[0, 0] * corners_cam[:, 0] / corners_cam[:, 2] + K[0, 2]
    v = K[1, 1] * corners_cam[:, 1] / corners_cam[:, 2] + K[1, 2]
    return np.stack([u, v], axis=1)


def annotate(corners_local: np.ndarray,
             otter_pos: np.ndarray,
             otter_quat_xyzw: np.ndarray,
             cam_pos: np.ndarray,
             cam_yaw_rad: float,
             cam_tilt_rad: float,
             K: np.ndarray,
             img_w: int,
             img_h: int,
             border_margin: int = 5,
             min_corners: int = 4) -> Optional[str]:
    """
    High-level entry point: return the YOLO OBB label string or None if not visible.
    """
    pixels = project_corners(corners_local, otter_pos, otter_quat_xyzw,
                              cam_pos, cam_yaw_rad, cam_tilt_rad, K)
    if pixels is None:
        return None
    return make_yolo_obb_label(pixels, img_w, img_h, 0, border_margin, min_corners)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helper (used by verify_dataset.py and preview mode)
# ─────────────────────────────────────────────────────────────────────────────

def draw_obb(image: np.ndarray,
             label_line: str,
             img_w: int,
             img_h: int,
             color: tuple = (0, 255, 0),
             thickness: int = 2) -> np.ndarray:
    """
    Draw the OBB polygon from a YOLO OBB label line onto `image` (BGR).
    Returns a copy with the overlay.
    """
    import cv2
    parts = label_line.strip().split()
    if len(parts) != 9:
        return image
    coords = np.array(parts[1:], dtype=np.float64).reshape(4, 2)
    coords[:, 0] *= img_w
    coords[:, 1] *= img_h
    pts = coords.astype(np.int32).reshape((-1, 1, 2))
    img_out = image.copy()
    cv2.polylines(img_out, [pts], isClosed=True, color=color, thickness=thickness)
    for pt in coords.astype(np.int32):
        cv2.circle(img_out, tuple(pt), 4, (0, 0, 255), -1)
    return img_out


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    K = compute_K(924, 690, 67.0)
    print(f"K =\n{K}\n")

    corners    = ar_marker_corners_local(-0.15, 0.0, 0.7, 1.0)
    otter_pos  = np.array([0.0, 0.0, 0.0])
    otter_quat = np.array([0.0, 0.0, 0.0, 1.0])
    cam_pos    = np.array([0.0, 0.0, 30.0])

    # ── Test 1: nadir (tilt=0) — USV at origin, camera directly above ────────
    pixels = project_corners(corners, otter_pos, otter_quat,
                              cam_pos, 0.0, 0.0, K)
    print(f"Projected corners (nadir, tilt=0):\n{pixels}\n")
    label = make_yolo_obb_label(pixels, 924, 690)
    print(f"Label: {label}")
    centre = pixels.mean(axis=0)
    print(f"OBB centre: ({centre[0]:.1f}, {centre[1]:.1f})  "
          f"image centre: ({924/2:.1f}, {690/2:.1f})")
    assert abs(centre[0] - 462) < 5 and abs(centre[1] - 345) < 5, "Centre mismatch!"
    print("Test 1 PASSED — nadir projection\n")

    # ── Test 2: back-projection nadir — centre pixel → directly below camera ──
    hit = backproject_pixel_to_ground(462.0, 345.0, K, cam_pos, 0.0, 0.0, 0.7)
    assert hit is not None, "Back-projection returned None"
    print(f"Back-project (nadir): pixel (462,345) → world ({hit[0]:.4f}, {hit[1]:.4f})")
    assert abs(hit[0]) < 0.01 and abs(hit[1]) < 0.01, "Nadir back-projection failed!"
    print("Test 2 PASSED — back-projection nadir\n")

    # ── Test 3: back-projection tilt=20°, yaw=0° — boresight shifts in +X ────
    tilt_rad     = np.radians(20.0)
    expected_x   = (cam_pos[2] - 0.7) * np.tan(tilt_rad)   # ≈ 10.67 m
    hit3 = backproject_pixel_to_ground(462.0, 345.0, K, cam_pos, 0.0, tilt_rad, 0.7)
    assert hit3 is not None, "Back-projection returned None for tilt=20°"
    print(f"Back-project (tilt=20°, yaw=0°): pixel (462,345) → world "
          f"({hit3[0]:.3f}, {hit3[1]:.3f})")
    print(f"  Expected X ≈ {expected_x:.3f} m")
    assert abs(hit3[0] - expected_x) < 0.1, f"Tilt back-projection failed: {hit3[0]:.3f}"
    assert abs(hit3[1]) < 0.01, "Unexpected Y offset in tilt=20° yaw=0° test"
    print("Test 3 PASSED — back-projection tilted\n")

    print("All self-tests PASSED")

#!/usr/bin/env python3
"""
collect_dataset_gimbal.py
─────────────────────────
YOLO OBB dataset collector simulating the Gremsy G-Hadron gimbal camera
mounted on the MRS X500 UAV.

Unlike collect_dataset.py (standalone overhead camera + moving USV), this
script:
  - Fixes the Otter USV at world origin (no set_model_state on otter).
  - Samples drone world position + gimbal joint angles (yaw, roll=0, pitch).
  - Computes the camera optical-frame pose via forward kinematics (FK) using
    the same chain as gimbal_position_node.py.
  - Teleports the virtual camera model (overhead_cam URDF) to that FK pose
    using a full-quaternion model-state command (not the simplified yaw+tilt).
  - Annotates with project_corners_fk() — no simplified yaw/tilt model.

Three collection modes (--mode arg / ~mode ROS param):
  positive      — UAV above USV, gimbal pitch ∈ [0, π/2], USV in frame.
  hard_negative — HN-A (deck-level) + HN-B (off-target altitude), USV NOT in frame.
  all           — Both (default).

Usage (via launch file):
    roslaunch dataset_acquisition data_collection_gimbal.launch
    roslaunch dataset_acquisition data_collection_gimbal.launch preview_only:=true

Usage (standalone):
    rosrun dataset_acquisition collect_dataset_gimbal.py \\
        _params_file:=/path/to/collection_params_gimbal.yaml \\
        _preview_only:=true
"""

import os
import sys
import random
import math
import shutil
from typing import Optional, Tuple

import numpy as np
import cv2
import rospy
import rospkg
import yaml
from scipy.spatial.transform import Rotation as Rot

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion

# Local module (same scripts/ directory)
sys.path.insert(0, os.path.dirname(__file__))
from annotator import (compute_K, ar_marker_corners_local,
                        project_corners_fk, make_yolo_obb_label, draw_obb)


# ─────────────────────────────────────────────────────────────────────────────
# Gimbal forward-kinematics constants
# Must match gimbal_position_node.py and aruco_pose_test_node.py exactly.
# ─────────────────────────────────────────────────────────────────────────────

_HALF_PI = math.pi / 2.0

# Fixed optical-frame rotation: R_OPT = Rz(-π/2) · Rx(-π/2)
# Applied as: R_opt = R_cam * _R_OPT   (optical→world when R_cam is cam_link→world)
_R_OPT = Rot.from_euler('z', -_HALF_PI) * Rot.from_euler('x', -_HALF_PI)
_R_OPT_INV = _R_OPT.inv()
_Ry_neg_half_pi = Rot.from_euler('y', -_HALF_PI)

# Gimbal link offsets (match SDF macro — all in metres)
_YAW_OFF   = np.array([0.0,   0.0, -0.025])
_ROLL_OFF  = np.array([0.0,   0.0, -0.030])
_PITCH_OFF = np.array([0.0,   0.0, -0.025])
_OPT_OFF   = np.array([0.025, 0.0,  0.0  ])


# ─────────────────────────────────────────────────────────────────────────────
# FK helpers
# ─────────────────────────────────────────────────────────────────────────────

def _camera_fk(drone_pos: np.ndarray,
               drone_rot: Rot,
               base_offset: np.ndarray,
               yaw: float, roll: float, pitch: float
               ) -> Tuple[np.ndarray, Rot]:
    """
    Gimbal forward kinematics → camera optical-frame world pose.

    Parameters
    ----------
    drone_pos   : (3,) drone body origin in world frame
    drone_rot   : drone body → world rotation
    base_offset : gimbal base link offset in drone body frame [m]
    yaw, roll, pitch : gimbal joint angles [rad]
                       pitch = +π/2 → nadir  (straight down)
                       pitch = 0    → horizontal
                       pitch < 0    → above horizontal

    Returns
    -------
    p_opt : (3,) optical centre in world frame
    R_opt : Rotation  optical→world  (same convention as aruco_pose_test_node)
    """
    p_base = drone_pos + drone_rot.apply(base_offset)
    R_base = drone_rot

    p_yaw  = p_base + R_base.apply(_YAW_OFF);   R_yaw  = R_base * Rot.from_euler('z', yaw)
    p_roll = p_yaw  + R_yaw.apply(_ROLL_OFF);   R_roll = R_yaw  * Rot.from_euler('x', roll)
    p_cam  = p_roll + R_roll.apply(_PITCH_OFF);  R_cam  = R_roll * Rot.from_euler('y', pitch)
    p_opt  = p_cam  + R_cam.apply(_OPT_OFF);     R_opt  = R_cam  * _R_OPT
    return p_opt, R_opt


def _nominal_gimbal_angles(drone_pos: np.ndarray,
                            drone_rot: Rot,
                            target_world: np.ndarray
                            ) -> Tuple[float, float]:
    """
    Compute the (yaw, pitch) gimbal angles in drone body frame that aim the
    camera at target_world.  gimbal_roll = 0 assumed.

    Derivation (view direction in world frame from FK with drone at identity):
        view = [cos(p)·cos(y),  cos(p)·sin(y),  −sin(p)]
    where p=pitch, y=yaw.  Inverting:
        pitch = arcsin(−v_body_z)
        yaw   = atan2(v_body_y, v_body_x)

    Returns (gimbal_yaw_rad, gimbal_pitch_rad).
    """
    v = target_world - drone_pos
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return 0.0, _HALF_PI          # default: nadir
    v /= norm
    v_body = drone_rot.inv().apply(v)  # world → drone body frame
    pitch = math.asin(float(np.clip(-v_body[2], -1.0, 1.0)))
    yaw   = math.atan2(float(v_body[1]), float(v_body[0]))
    return yaw, pitch


def _optical_to_base_link(p_opt: np.ndarray, R_opt: Rot) -> Tuple[np.ndarray, Rot]:
    """
    Convert the FK optical pose (p_opt, R_opt) to the URDF base_link pose
    required for set_model_state so that Gazebo renders from exactly that
    optical position and orientation.

    URDF chain (all joint xyz=0 0 0, so position is the same throughout):
        base_link --Ry(π/2)--> overhead_cam_link --R_OPT--> overhead_cam_optical

    R_base = R_opt · R_OPT⁻¹ · Ry(−π/2)

    Verification: R_base · Ry(π/2) · R_OPT = R_opt ✓
    """
    R_base = R_opt * _R_OPT_INV * _Ry_neg_half_pi
    return p_opt.copy(), R_base


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers (shared with collect_dataset.py pattern)
# ─────────────────────────────────────────────────────────────────────────────

def _ros_quaternion(rot: Rot) -> Quaternion:
    q = rot.as_quat()   # scipy → [x, y, z, w]
    return Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))


def _set_model_pose_full(set_state_srv, model_name: str,
                          pos: np.ndarray, rot: Rot) -> bool:
    """Teleport a Gazebo model to pos with full-quaternion orientation."""
    state = ModelState()
    state.model_name      = model_name
    state.reference_frame = "world"
    state.pose = Pose(
        position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
        orientation=_ros_quaternion(rot),
    )
    # zero twist
    for attr in ("linear", "angular"):
        for ax in ("x", "y", "z"):
            setattr(getattr(state.twist, attr), ax, 0.0)
    return set_state_srv(state).success


def _get_model_pose(get_state_srv, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pos[3], quat_xyzw[4]) or raise on failure."""
    resp = get_state_srv(model_name, "world")
    if not resp.success:
        raise RuntimeError(f"get_model_state failed for '{model_name}'")
    p, q = resp.pose.position, resp.pose.orientation
    return np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w])


def _split_and_move(src_images: list, src_labels: list,
                    dataset_dir: str, train_r: float, val_r: float) -> None:
    indices = list(range(len(src_images)))
    random.shuffle(indices)
    n = len(indices)
    splits = {
        "train": indices[:int(n * train_r)],
        "val":   indices[int(n * train_r):int(n * (train_r + val_r))],
        "test":  indices[int(n * (train_r + val_r)):],
    }
    for split, idxs in splits.items():
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in idxs:
            shutil.move(src_images[i], img_dir)
            shutil.move(src_labels[i], lbl_dir)


def _write_yolo_yaml(dataset_dir: str) -> None:
    cfg = {"path": dataset_dir, "train": "images/train",
           "val": "images/val", "test": "images/test",
           "nc": 1, "names": ["otter_usv"]}
    out = os.path.join(dataset_dir, "otter_usv.yaml")
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    rospy.loginfo(f"[gimbal_collect] YOLO dataset config → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main collector
# ─────────────────────────────────────────────────────────────────────────────

class GimbalDatasetCollector:

    def __init__(self, params: dict, dataset_dir: str,
                 mode: str, preview_only: bool):
        self.p            = params
        self.dataset_dir  = dataset_dir
        self.mode         = mode          # "positive" | "hard_negative" | "all"
        self.preview_only = preview_only

        # Camera intrinsics
        cam = params["camera"]
        self.img_w = cam["width_px"]
        self.img_h = cam["height_px"]
        self.K     = compute_K(self.img_w, self.img_h, cam["hfov_deg"])
        self._cam_model = cam["model_name"]
        self._img_topic = cam["topic_image"]
        rospy.loginfo(f"[gimbal_collect] Camera intrinsics K:\n{self.K}")

        # Gimbal base offset in drone body frame
        self._base_off = np.array(params["gimbal"]["base_offset_m"], dtype=np.float64)

        # AR marker corners in Otter local frame
        ar = params["ar_marker"]
        self._corners_local = ar_marker_corners_local(
            ar["center_x_m"], ar["center_y_m"], ar["center_z_m"], ar["half_size_m"])
        # USV AR marker centre in world frame (used for nominal gimbal aiming)
        self._usv_target_world = np.array(
            [ar["center_x_m"], ar["center_y_m"], ar["center_z_m"]], dtype=np.float64)

        # Visibility filter
        vis = params["visibility"]
        self._border_margin = vis["border_margin_px"]
        self._min_corners   = vis["min_corners_in_frame"]

        # Otter model name
        self._otter_model = params["otter"]["model_name"]

        # Output directories
        self._tmp_dir     = os.path.join(dataset_dir, "_tmp")
        self._preview_dir = os.path.join(dataset_dir, "preview")
        os.makedirs(self._tmp_dir,     exist_ok=True)
        os.makedirs(self._preview_dir, exist_ok=True)

        # Image subscription
        self._bridge         = CvBridge()
        self._latest_img_msg = None
        rospy.Subscriber(self._img_topic, Image, self._img_cb, queue_size=1)

        # Gazebo services
        rospy.loginfo("[gimbal_collect] Waiting for Gazebo services …")
        rospy.wait_for_service("/gazebo/set_model_state")
        rospy.wait_for_service("/gazebo/get_model_state")
        self._set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self._get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        rospy.loginfo("[gimbal_collect] Gazebo services ready.")

    # ── Image subscriber ───────────────────────────────────────────────────────

    def _img_cb(self, msg: Image):
        self._latest_img_msg = msg

    # ── Image capture ──────────────────────────────────────────────────────────

    def _capture_fresh(self, timeout_s: float = 8.0) -> Optional[np.ndarray]:
        """
        Discard stale frames; wait for two consecutive new frames then return
        the second as BGR.  The two-frame wait guards against the one-frame
        lag Gazebo can produce after a camera teleport.
        """
        deadline = rospy.Time.now() + rospy.Duration(timeout_s)
        for _ in range(2):
            self._latest_img_msg = None
            while self._latest_img_msg is None and not rospy.is_shutdown():
                if rospy.Time.now() > deadline:
                    rospy.logwarn("[gimbal_collect] Timeout waiting for fresh image.")
                    return None
                rospy.sleep(0.05)
        return self._bridge.imgmsg_to_cv2(self._latest_img_msg, desired_encoding="bgr8")

    # ── Camera teleport ────────────────────────────────────────────────────────

    def _place_camera(self, drone_pos: np.ndarray, drone_rot: Rot,
                      gimbal_yaw: float, gimbal_pitch: float,
                      gimbal_roll: float = 0.0) -> Tuple[np.ndarray, Rot]:
        """
        Compute FK → convert to URDF base_link pose → teleport virtual camera.

        Returns (p_opt, R_opt) for use in annotation.
        """
        p_opt, R_opt = _camera_fk(drone_pos, drone_rot, self._base_off,
                                   gimbal_yaw, gimbal_roll, gimbal_pitch)
        p_base, R_base = _optical_to_base_link(p_opt, R_opt)
        _set_model_pose_full(self._set_state, self._cam_model, p_base, R_base)
        return p_opt, R_opt

    # ── Annotation ────────────────────────────────────────────────────────────

    def _annotate(self, otter_pos: np.ndarray, otter_quat: np.ndarray,
                  p_opt: np.ndarray, R_opt: Rot) -> Optional[str]:
        """Project USV corners through the FK camera; return YOLO OBB label or None."""
        pixels = project_corners_fk(
            self._corners_local, otter_pos, otter_quat, p_opt, R_opt, self.K)
        if pixels is None:
            return None
        return make_yolo_obb_label(
            pixels, self.img_w, self.img_h,
            border_margin=self._border_margin,
            min_corners=self._min_corners)

    # ── Corner-indexed OBB drawing (preview only) ─────────────────────────────

    def _draw_indexed_obb(self, image: np.ndarray,
                           pixels: np.ndarray) -> np.ndarray:
        """
        Draw the OBB polygon and label each corner with its index (1-4).

        Corner order (from ar_marker_corners_local):
          1 — front-port      (+x, +y) : bow, port side
          2 — front-starboard (+x, −y) : bow, starboard side
          3 — rear-starboard  (−x, −y) : stern, starboard side
          4 — rear-port       (−x, +y) : stern, port side

        The 1→2 edge encodes the USV bow direction (consistent with the
        YOLO OBB annotation convention for this dataset).
        """
        img = image.copy()
        pts = pixels.astype(np.int32)

        # Draw closed polygon
        cv2.polylines(img, [pts.reshape(-1, 1, 2)],
                      isClosed=True, color=(0, 255, 0), thickness=2)

        # Corner colours: bow=cyan, stern=magenta
        corner_colors = [
            (255, 255,   0),   # 1 front-port      yellow
            (  0, 255, 255),   # 2 front-starboard  cyan
            (255,   0, 255),   # 3 rear-starboard   magenta
            (255, 128,   0),   # 4 rear-port        orange
        ]
        corner_labels = ["1 fp", "2 fs", "3 rs", "4 rp"]

        for i, (u, v) in enumerate(pts):
            col = corner_colors[i]
            cv2.circle(img, (u, v), 6, col, -1)
            cv2.putText(img, corner_labels[i], (u + 8, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        # Legend at bottom-left
        for i, (lbl, col) in enumerate(zip(corner_labels, corner_colors)):
            cv2.putText(img, lbl, (8, self.img_h - 10 - i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

        return img

    # ── Raw pixel projection (used by preview for indexed drawing) ────────────

    def _project_pixels(self, otter_pos: np.ndarray, otter_quat: np.ndarray,
                         p_opt: np.ndarray, R_opt: Rot) -> Optional[np.ndarray]:
        """Returns (4,2) pixel array (same order as annotation corners) or None."""
        return project_corners_fk(
            self._corners_local, otter_pos, otter_quat, p_opt, R_opt, self.K)

    # ── Otter pose ────────────────────────────────────────────────────────────

    def _otter_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return _get_model_pose(self._get_state, self._otter_model)

    # ── Positive sample ────────────────────────────────────────────────────────

    def _collect_positive(self) -> Optional[Tuple[np.ndarray, str]]:
        """
        Sample one positive frame: UAV above USV, gimbal pitch ∈ [0, π/2],
        USV must appear in the image.

        Strategy:
          1. Random drone position (±xy_bound, altitude).
          2. Compute nominal gimbal angles to look at USV AR marker centre.
          3. Add jitter so the USV lands at various image positions.
          4. Clamp pitch to [0, π/2] — no pointing above horizontal.
          5. Reject if USV not visible (jitter pushed it outside FOV).
        """
        p_cfg = self.p["positive"]
        bound = p_cfg["drone_xy_bound_m"]
        p_jit = p_cfg["pitch_jitter_rad"]
        y_jit = p_cfg["yaw_jitter_rad"]

        for _ in range(int(p_cfg["max_attempts"])):
            # 1. Sample drone pose
            drone_pos = np.array([
                random.uniform(-bound, bound),
                random.uniform(-bound, bound),
                random.uniform(p_cfg["altitude_min_m"], p_cfg["altitude_max_m"]),
            ])
            drone_yaw = random.uniform(0.0, 2.0 * math.pi)
            drone_rot = Rot.from_euler('z', drone_yaw)

            # 2. Nominal gimbal angles toward USV AR marker centre
            nom_yaw, nom_pitch = _nominal_gimbal_angles(
                drone_pos, drone_rot, self._usv_target_world)

            # 3. Add jitter, clamp pitch to [0, π/2]
            g_yaw   = nom_yaw   + random.uniform(-y_jit, y_jit)
            g_pitch = nom_pitch + random.uniform(-p_jit, p_jit)
            g_pitch = float(np.clip(g_pitch, 0.0, _HALF_PI))

            # 4. Teleport camera
            p_opt, R_opt = self._place_camera(drone_pos, drone_rot, g_yaw, g_pitch)
            rospy.sleep(0.3)

            # 5. Capture + annotate
            image = self._capture_fresh()
            if image is None:
                continue
            otter_pos, otter_quat = self._otter_pose()
            label = self._annotate(otter_pos, otter_quat, p_opt, R_opt)
            if label is None:
                continue    # USV not visible — retry
            return image, label

        return None   # Exceeded max attempts

    # ── HN-A sample (deck-level) ───────────────────────────────────────────────

    def _collect_hn_a(self) -> Optional[np.ndarray]:
        """
        Hard-Negative A: UAV at deck level (0.5–1.5 m), gimbal looking
        sideways or upward — camera sees ocean/sky but NOT the USV AR panel.

        Gimbal yaw drawn from nominal 60°-increment directions + ±jitter
        so the dataset has variety without always using exact cardinal angles.
        """
        cfg = self.p["hard_negative_a"]

        for _ in range(int(cfg["max_attempts"])):
            # Drone right on / just above deck, random heading
            drone_pos = np.array([0.0, 0.0,
                                   random.uniform(cfg["altitude_min_m"],
                                                  cfg["altitude_max_m"])])
            drone_yaw = random.uniform(0.0, 2.0 * math.pi)
            drone_rot = Rot.from_euler('z', drone_yaw)

            # Gimbal yaw: nominal 60° increment + jitter (in drone body frame)
            nom_deg = random.choice(cfg["yaw_nominal_deg"])
            jit_deg = random.uniform(-cfg["yaw_jitter_deg"], cfg["yaw_jitter_deg"])
            g_yaw   = math.radians(nom_deg + jit_deg)

            # Gimbal pitch: horizontal to upward (NOT toward deck)
            g_pitch = random.uniform(cfg["gimbal_pitch_min_rad"],
                                     cfg["gimbal_pitch_max_rad"])

            p_opt, R_opt = self._place_camera(drone_pos, drone_rot, g_yaw, g_pitch)
            rospy.sleep(0.3)
            image = self._capture_fresh()
            if image is None:
                continue

            # Reject if USV accidentally visible
            otter_pos, otter_quat = self._otter_pose()
            if self._annotate(otter_pos, otter_quat, p_opt, R_opt) is not None:
                continue    # USV visible — retry with different angles

            return image   # Valid HN: no USV in frame

        return None

    # ── HN-B sample (off-target altitude) ─────────────────────────────────────

    def _collect_hn_b(self) -> Optional[np.ndarray]:
        """
        Hard-Negative B: UAV at 2–15 m, gimbal pointing sideways or upward.
        Camera sees ocean surface or sky — USV must NOT appear.
        """
        cfg = self.p["hard_negative_b"]

        for _ in range(int(cfg["max_attempts"])):
            drone_pos = np.array([0.0, 0.0,
                                   random.uniform(cfg["altitude_min_m"],
                                                  cfg["altitude_max_m"])])
            drone_yaw = random.uniform(0.0, 2.0 * math.pi)
            drone_rot = Rot.from_euler('z', drone_yaw)

            g_yaw   = random.uniform(0.0, 2.0 * math.pi)
            g_pitch = random.uniform(cfg["gimbal_pitch_min_rad"],
                                     cfg["gimbal_pitch_max_rad"])

            p_opt, R_opt = self._place_camera(drone_pos, drone_rot, g_yaw, g_pitch)
            rospy.sleep(0.3)
            image = self._capture_fresh()
            if image is None:
                continue

            otter_pos, otter_quat = self._otter_pose()
            if self._annotate(otter_pos, otter_quat, p_opt, R_opt) is not None:
                continue    # USV visible — retry

            return image

        return None

    # ── Preview grid ───────────────────────────────────────────────────────────

    def _collect_preview(self):
        """
        Collect all samples from preview_grid in collection_params_gimbal.yaml.
        For each spec:
          - Teleport camera to FK pose given the spec parameters.
          - Capture image and optionally annotate.
          - Save <label>_raw.jpg + <label>_obb.jpg (OBB drawn if USV visible).
          - Warn if expectation (expect_usv) is not met.
        """
        grid = self.p.get("preview_grid", [])
        if not grid:
            rospy.logwarn("[gimbal_collect] No preview_grid defined.")
            return

        passed = 0
        for i, spec in enumerate(grid):
            lbl = spec.get("label", f"sample_{i:02d}")
            rospy.loginfo(f"[gimbal_collect] Preview {i+1}/{len(grid)}: {lbl}")

            drone_pos = np.array([float(spec["drone_x"]),
                                   float(spec["drone_y"]),
                                   float(spec["drone_z"])])
            drone_rot = Rot.from_euler('z', math.radians(float(spec["drone_yaw_deg"])))
            g_pitch   = math.radians(float(spec["gimbal_pitch_deg"]))
            g_yaw     = math.radians(float(spec["gimbal_yaw_deg"]))

            p_opt, R_opt = self._place_camera(drone_pos, drone_rot, g_yaw, g_pitch)
            rospy.sleep(0.4)
            image = self._capture_fresh()
            if image is None:
                rospy.logwarn(f"[gimbal_collect]   '{lbl}' — capture failed, skipping.")
                continue

            otter_pos, otter_quat = self._otter_pose()
            label  = self._annotate(otter_pos, otter_quat, p_opt, R_opt)
            pixels = self._project_pixels(otter_pos, otter_quat, p_opt, R_opt)

            expect_usv  = bool(spec.get("expect_usv", True))
            usv_visible = label is not None

            if expect_usv and not usv_visible:
                rospy.logwarn(
                    f"[gimbal_collect]   '{lbl}' — expected USV visible but got None. "
                    f"Check drone_z / gimbal angles.")
            elif not expect_usv and usv_visible:
                rospy.logwarn(
                    f"[gimbal_collect]   '{lbl}' — expected HN (no USV) but USV is visible. "
                    f"Adjust gimbal angles.")
            else:
                rospy.loginfo(f"[gimbal_collect]   '{lbl}' — expectation met.")

            # Annotate preview: use indexed OBB (corner numbers) when USV is visible
            img_obb = image.copy()
            if pixels is not None:
                img_obb = self._draw_indexed_obb(img_obb, pixels)

            text = (f"alt={spec['drone_z']:.0f}m | "
                    f"pitch={spec['gimbal_pitch_deg']:.0f}deg | "
                    f"yaw={spec['gimbal_yaw_deg']:.0f}deg")
            cv2.putText(img_obb, text, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            tag = "USV" if usv_visible else "HN"
            cv2.putText(img_obb, tag, (self.img_w - 80, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0) if usv_visible else (0, 0, 255), 2, cv2.LINE_AA)

            raw_path = os.path.join(self._preview_dir, f"{lbl}_raw.jpg")
            obb_path = os.path.join(self._preview_dir, f"{lbl}_obb.jpg")
            cv2.imwrite(raw_path, image)
            cv2.imwrite(obb_path, img_obb)
            passed += 1
            rospy.loginfo(f"[gimbal_collect]   → {obb_path}")

        rospy.loginfo(
            f"[gimbal_collect] Preview done: {passed}/{len(grid)} samples "
            f"→ {self._preview_dir}")

    # ── Main run ───────────────────────────────────────────────────────────────

    def run(self):
        # Wait for models to appear in Gazebo
        rospy.loginfo("[gimbal_collect] Waiting for Gazebo models …")
        for name in [self._otter_model, self._cam_model]:
            while not rospy.is_shutdown():
                if self._get_state(name, "world").success:
                    break
                rospy.logwarn_throttle(
                    2.0, f"[gimbal_collect] Still waiting for '{name}' …")
                rospy.sleep(1.0)
        rospy.loginfo("[gimbal_collect] Models confirmed in Gazebo.")

        # Wait for the camera plugin to come online
        rospy.loginfo("[gimbal_collect] Waiting for camera first frame …")
        try:
            rospy.wait_for_message(self._img_topic, Image, timeout=120.0)
            rospy.loginfo("[gimbal_collect] Camera online.")
        except rospy.ROSException:
            rospy.logerr(
                "[gimbal_collect] Camera never published — check overhead_cam plugin.")
            return

        # ── Preview mode ─────────────────────────────────────────────────────
        if self.preview_only:
            rospy.loginfo("[gimbal_collect] === PREVIEW MODE ===")
            self._collect_preview()
            return

        # ── Full collection ───────────────────────────────────────────────────
        out_cfg  = self.p["output"]
        fmt      = out_cfg["image_format"]
        jpeg_q   = out_cfg["jpeg_quality"]

        collected_imgs   = []
        collected_labels = []
        idx = 0

        def _save(image: np.ndarray, label_str: Optional[str]) -> None:
            """Write image + label (or empty txt for HN) to tmp dir."""
            nonlocal idx
            stem     = f"{idx:05d}"
            img_path = os.path.join(self._tmp_dir, f"{stem}.{fmt}")
            lbl_path = os.path.join(self._tmp_dir, f"{stem}.txt")
            if fmt == "jpg":
                cv2.imwrite(img_path, image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
            else:
                cv2.imwrite(img_path, image)
            with open(lbl_path, "w") as f:
                if label_str:
                    f.write(label_str + "\n")
                # Empty file = background image (YOLO treats as no-object)
            collected_imgs.append(img_path)
            collected_labels.append(lbl_path)
            idx += 1

        # ── Positives ─────────────────────────────────────────────────────────
        run_positive = self.mode in ("positive", "all")
        if run_positive:
            n_pos = out_cfg["n_positive"]
            rospy.loginfo(f"[gimbal_collect] Collecting {n_pos} positive samples …")
            n_done = 0
            attempts = 0
            while n_done < n_pos and not rospy.is_shutdown():
                attempts += 1
                result = self._collect_positive()
                if result is None:
                    if attempts > n_pos * 5:
                        rospy.logerr("[gimbal_collect] Too many failed attempts — check setup.")
                        break
                    continue
                image, label = result
                _save(image, label)
                n_done += 1
                rospy.loginfo(
                    f"[gimbal_collect] +POS {n_done}/{n_pos}  (attempts: {attempts})")

        # ── Hard negatives ─────────────────────────────────────────────────────
        run_hn = self.mode in ("hard_negative", "all")
        if run_hn:
            n_hn_a = out_cfg["n_hn_a"]
            n_hn_b = out_cfg["n_hn_b"]
            rospy.loginfo(
                f"[gimbal_collect] Collecting {n_hn_a} HN-A + {n_hn_b} HN-B samples …")

            for label_kind, target_n, collect_fn in [
                ("HN-A", n_hn_a, self._collect_hn_a),
                ("HN-B", n_hn_b, self._collect_hn_b),
            ]:
                n_done = 0
                attempts = 0
                while n_done < target_n and not rospy.is_shutdown():
                    attempts += 1
                    image = collect_fn()
                    if image is None:
                        if attempts > target_n * 5:
                            rospy.logerr(
                                f"[gimbal_collect] Too many {label_kind} failures — check config.")
                            break
                        continue
                    _save(image, None)  # Empty label → background
                    n_done += 1
                    rospy.loginfo(
                        f"[gimbal_collect] +{label_kind} {n_done}/{target_n}  "
                        f"(attempts: {attempts})")

        # ── Finalise ──────────────────────────────────────────────────────────
        total = len(collected_imgs)
        rospy.loginfo(f"[gimbal_collect] Collected {total} samples total.")

        if total == 0:
            rospy.logerr("[gimbal_collect] No samples collected — aborting.")
            return

        _split_and_move(collected_imgs, collected_labels, self.dataset_dir,
                        out_cfg["train_ratio"], out_cfg["val_ratio"])
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
        _write_yolo_yaml(self.dataset_dir)

        for split in ("train", "val", "test"):
            d = os.path.join(self.dataset_dir, "images", split)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            rospy.loginfo(f"[gimbal_collect]   {split}: {n} images")

        rospy.loginfo("[gimbal_collect] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rospy.init_node("collect_dataset_gimbal", anonymous=False)

    params_file = rospy.get_param("~params_file", "")
    if not params_file:
        pkg = rospkg.RosPack().get_path("dataset_acquisition")
        params_file = os.path.join(pkg, "config", "collection_params_gimbal.yaml")

    with open(params_file) as f:
        params = yaml.safe_load(f)

    dataset_dir = rospy.get_param("~dataset_dir", "")
    if not dataset_dir:
        pkg = rospkg.RosPack().get_path("dataset_acquisition")
        dataset_dir = os.path.join(pkg, "dataset_gimbal")
    os.makedirs(dataset_dir, exist_ok=True)

    mode         = rospy.get_param("~mode",         "all")
    preview_only = rospy.get_param("~preview_only", False)

    rospy.loginfo(f"[gimbal_collect] Dataset dir  : {dataset_dir}")
    rospy.loginfo(f"[gimbal_collect] Params file  : {params_file}")
    rospy.loginfo(f"[gimbal_collect] Mode         : {mode}")
    rospy.loginfo(f"[gimbal_collect] Preview only : {preview_only}")

    rospy.loginfo("[gimbal_collect] Waiting 5 s for Gazebo to stabilise …")
    rospy.sleep(5.0)

    collector = GimbalDatasetCollector(params, dataset_dir, mode, preview_only)
    collector.run()


if __name__ == "__main__":
    main()

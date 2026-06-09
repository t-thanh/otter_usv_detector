#!/usr/bin/env python3
"""
collect_test_samples.py
────────────────────────
Collect full-pose ground-truth test samples for pose-estimator evaluation.

For each deterministic scenario in test_grid.yaml:
  1. Compute gimbal FK → camera optical pose (p_opt, R_opt).
  2. Teleport overhead_cam URDF so its optical frame matches the FK pose.
  3. Teleport Otter USV to origin with the configured yaw.
  4. Capture a rendered Gazebo frame.
  5. Verify the AR panel is geometrically visible.
  6. Save full ground-truth: camera position + UAV orientation in USV frame.

Output
------
  <output_dir>/images/sample_NNNN.jpg
  <output_dir>/metadata.yaml

Run via:
    roslaunch otter_usv_detector collect_test_samples.launch
"""

import os
import math
import yaml
import numpy as np
import cv2
import rospy
import rospkg

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation as Rot


# ── Gimbal FK constants — identical to yolo_pose_estimator_node.py ────────────
_HALF_PI   = math.pi / 2.0
_R_OPT     = Rot.from_euler('z', -_HALF_PI) * Rot.from_euler('x', -_HALF_PI)
_BASE_OFF  = np.array([0.10, 0.00,  0.00])
_YAW_OFF   = np.array([0.0,  0.0,  -0.025])
_ROLL_OFF  = np.array([0.0,  0.0,  -0.030])
_PITCH_OFF = np.array([0.0,  0.0,  -0.025])
_OPT_OFF   = np.array([0.025, 0.0,  0.00 ])

# Inverse of the URDF base_link→cam_link joint (Ry(π/2))
_R_URDF_INV = Rot.from_euler('y', -_HALF_PI)

# ── Camera constants (Hadron 640R EO) ─────────────────────────────────────────
IMG_W, IMG_H = 924, 690
HFOV_DEG     = 67.0

# ── AR-panel corners in Otter base_link frame (from otter_base.urdf.xacro) ───
_CX, _CY, _CZ, _H = -0.15, 0.0, 0.70, 1.25
_OBJ_PTS = np.array([
    [_CX + _H, _CY + _H, _CZ],   # front-port
    [_CX + _H, _CY - _H, _CZ],   # front-starboard
    [_CX - _H, _CY - _H, _CZ],   # rear-starboard
    [_CX - _H, _CY + _H, _CZ],   # rear-port
], dtype=np.float64)


def _build_K():
    fx = (IMG_W / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)
    return np.array([[fx, 0,  IMG_W / 2.0],
                     [0,  fx, IMG_H / 2.0],
                     [0,  0,  1.0         ]])


def _camera_fk(drone_pos: np.ndarray, drone_rot: Rot,
               yaw: float, roll: float, pitch: float):
    """Gimbal FK → (p_opt, R_opt).  Identical to yolo_pose_estimator_node.py."""
    p, R = drone_pos + drone_rot.apply(_BASE_OFF), drone_rot
    p, R = p + R.apply(_YAW_OFF),   R * Rot.from_euler('z', yaw)
    p, R = p + R.apply(_ROLL_OFF),  R * Rot.from_euler('x', roll)
    p, R = p + R.apply(_PITCH_OFF), R * Rot.from_euler('y', pitch)
    p, R = p + R.apply(_OPT_OFF),   R * _R_OPT
    return p, R


def _cam_teleport_pose(p_opt: np.ndarray, R_opt: Rot):
    """
    Convert FK optical pose to overhead_cam base_link pose for SetModelState.

    URDF chain: base_link -Ry(π/2)→ cam_link -R_OPT→ optical
    → R_base = R_opt · R_OPT⁻¹ · Ry(−π/2)
    """
    R_base = R_opt * _R_OPT.inv() * _R_URDF_INV
    q = R_base.as_quat()   # scipy: [x, y, z, w]
    return p_opt, q


def _all_corners_visible(p_opt: np.ndarray, R_opt: Rot,
                         usv_pos: np.ndarray, usv_rot: Rot,
                         K: np.ndarray, margin: int = 5) -> bool:
    """Return True iff all 4 AR-panel corners project inside the image."""
    for corner_usv in _OBJ_PTS:
        corner_world = usv_pos + usv_rot.apply(corner_usv)
        c_cam = R_opt.inv().apply(corner_world - p_opt)
        if c_cam[2] <= 0:
            return False
        u = K[0, 0] * c_cam[0] / c_cam[2] + K[0, 2]
        v = K[1, 1] * c_cam[1] / c_cam[2] + K[1, 2]
        if not (margin <= u <= IMG_W - margin and margin <= v <= IMG_H - margin):
            return False
    return True


def _make_model_state(name: str, pos: np.ndarray, quat_xyzw: np.ndarray) -> ModelState:
    st = ModelState()
    st.model_name      = name
    st.reference_frame = 'world'
    st.pose = Pose(
        position    = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
        orientation = Quaternion(x=float(quat_xyzw[0]), y=float(quat_xyzw[1]),
                                 z=float(quat_xyzw[2]), w=float(quat_xyzw[3])))
    return st


class TestSampleCollector:

    def __init__(self):
        rospy.init_node('collect_test_samples')

        pkg        = rospkg.RosPack().get_path('otter_usv_detector')
        grid_file  = rospy.get_param('~test_grid_file', '')
        output_dir = rospy.get_param('~output_dir',     '')
        self.fmt   = rospy.get_param('~image_format',   'jpg')
        self.jpegq = rospy.get_param('~jpeg_quality',   95)

        if not grid_file:
            grid_file  = os.path.join(pkg, 'config', 'test_grid.yaml')
        if not output_dir:
            output_dir = os.path.join(pkg, 'test_samples')

        with open(grid_file) as f:
            self.grid = yaml.safe_load(f)['test_grid']

        self.img_dir   = os.path.join(output_dir, 'images')
        self.meta_path = os.path.join(output_dir, 'metadata.yaml')
        os.makedirs(self.img_dir, exist_ok=True)

        self.K      = _build_K()
        self.bridge = CvBridge()
        self._latest: Image = None

        rospy.Subscriber('/overhead_cam/image_raw', Image,
                         lambda m: setattr(self, '_latest', m), queue_size=1)

        rospy.loginfo('[collect_test] Waiting for Gazebo set_model_state …')
        rospy.wait_for_service('/gazebo/set_model_state')
        self._set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        rospy.loginfo('[collect_test] Waiting for first camera frame …')
        rospy.wait_for_message('/overhead_cam/image_raw', Image, timeout=120.0)
        rospy.loginfo('[collect_test] Camera online — starting collection.')

    def _fresh_image(self, timeout: float = 8.0):
        """Discard potentially stale frame; return the next two-frame-consecutive image."""
        t0 = rospy.Time.now()
        for _ in range(2):
            self._latest = None
            while self._latest is None and not rospy.is_shutdown():
                if (rospy.Time.now() - t0).to_sec() > timeout:
                    return None
                rospy.sleep(0.05)
        return self.bridge.imgmsg_to_cv2(self._latest, desired_encoding='bgr8')

    def run(self):
        samples = []

        for idx, spec in enumerate(self.grid):
            label = spec.get('label', f'sample_{idx:04d}')

            # ── Parse spec ────────────────────────────────────────────────────
            drone_pos = np.array([float(spec['drone_x']),
                                   float(spec['drone_y']),
                                   float(spec['drone_z'])])
            drone_yaw = math.radians(float(spec['drone_yaw_deg']))
            drone_rot = Rot.from_euler('z', drone_yaw)

            # gimbal_pitch_deg: 90 = nadir → joint angle = +π/2 = Ry(+π/2)
            joint_yaw   = math.radians(float(spec['gimbal_yaw_deg']))
            joint_pitch = math.radians(float(spec['gimbal_pitch_deg']))

            usv_yaw = math.radians(float(spec.get('usv_yaw_deg', 0.0)))
            usv_pos = np.zeros(3)
            usv_rot = Rot.from_euler('z', usv_yaw)

            # ── FK ────────────────────────────────────────────────────────────
            p_opt, R_opt = _camera_fk(drone_pos, drone_rot, joint_yaw, 0.0, joint_pitch)

            # ── Visibility check (geometry) ───────────────────────────────────
            if not _all_corners_visible(p_opt, R_opt, usv_pos, usv_rot, self.K):
                rospy.logwarn(f'[collect_test] {idx+1:2d}/{len(self.grid)} '
                              f'"{label}": AR panel NOT visible — skipping')
                continue

            # ── Teleport USV ──────────────────────────────────────────────────
            usv_q = usv_rot.as_quat()
            self._set_state(_make_model_state('otter', usv_pos, usv_q))
            rospy.sleep(0.5)   # buoyancy settle

            # ── Teleport camera ───────────────────────────────────────────────
            cam_pos, cam_q = _cam_teleport_pose(p_opt, R_opt)
            self._set_state(_make_model_state('overhead_cam', cam_pos, cam_q))
            rospy.sleep(0.4)   # renderer catch-up

            # ── Capture ───────────────────────────────────────────────────────
            image = self._fresh_image()
            if image is None:
                rospy.logwarn(f'[collect_test] {idx+1:2d}/{len(self.grid)} '
                              f'"{label}": image timeout — skipping')
                continue

            # ── Ground-truth full pose ────────────────────────────────────────
            # Camera optical centre in USV frame
            gt_pos = usv_rot.inv().apply(p_opt - usv_pos)

            # Drone (UAV body) orientation in USV frame.
            # Hovering drone has no roll/pitch → only yaw matters.
            R_gt          = usv_rot.inv() * drone_rot
            gt_roll, gt_pitch, gt_yaw = R_gt.as_euler('xyz', degrees=True)
            gt_dist       = float(np.linalg.norm(gt_pos))

            # ── Save image ────────────────────────────────────────────────────
            img_name = f'sample_{idx:04d}.{self.fmt}'
            img_path = os.path.join(self.img_dir, img_name)
            enc_params = ([cv2.IMWRITE_JPEG_QUALITY, self.jpegq]
                          if self.fmt == 'jpg' else [])
            cv2.imwrite(img_path, image, enc_params)

            samples.append({
                'sample_id':        idx,
                'label':            label,
                'image_file':       img_name,
                # Synthetic drone state
                'drone_pos':        drone_pos.tolist(),
                'drone_yaw_deg':    math.degrees(drone_yaw),
                'drone_quat':       drone_rot.as_quat().tolist(),   # [x,y,z,w]
                # Gimbal joint angles (raw FK angles used in yolo_pose_estimator)
                'gimbal_yaw_rad':   float(joint_yaw),
                'gimbal_roll_rad':  0.0,
                'gimbal_pitch_rad': float(joint_pitch),
                # USV pose
                'usv_pos':          usv_pos.tolist(),
                'usv_yaw_deg':      math.degrees(usv_yaw),
                'usv_quat':         usv_rot.as_quat().tolist(),     # [x,y,z,w]
                # Ground truth: camera in USV frame
                'gt_cam_in_usv':    gt_pos.tolist(),        # [x, y, z] m
                'gt_roll_deg':      float(gt_roll),
                'gt_pitch_deg':     float(gt_pitch),
                'gt_yaw_deg':       float(gt_yaw),
                'gt_dist_m':        gt_dist,
            })

            rospy.loginfo(
                f'[collect_test] {idx+1:2d}/{len(self.grid)} "{label}"  '
                f'drone=({drone_pos[0]:+.0f},{drone_pos[1]:+.0f},{drone_pos[2]:.0f})m  '
                f'gt=({gt_pos[0]:+.2f},{gt_pos[1]:+.2f},{gt_pos[2]:+.2f})m  '
                f'yaw={gt_yaw:+.1f}°  dist={gt_dist:.2f}m')

        with open(self.meta_path, 'w') as f:
            yaml.dump({'samples': samples}, f, default_flow_style=False)

        rospy.loginfo(
            f'[collect_test] Done — {len(samples)}/{len(self.grid)} samples '
            f'saved to {os.path.dirname(self.meta_path)}')


if __name__ == '__main__':
    try:
        TestSampleCollector().run()
    except rospy.ROSInterruptException:
        pass

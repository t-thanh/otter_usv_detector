#!/usr/bin/env python3
"""
yolo_pose_estimator_node.py
───────────────────────────
Guidance-grade USV pose estimator from YOLO OBB detections.

Purpose
-------
Provides the UAV navigation controller with the position of the Otter USV
centre (base_link, z ≈ 0 = waterline) and the USV heading, all expressed in
the camera optical frame.  Intended for basic approach navigation; once the
UAV is close enough, ArUco-based precise landing takes over.

Algorithm
---------
1.  Receive OBB detection (4 corners + angle_deg) from the YOLO OBB detector.

2.  Camera pose in world (p_opt, R_opt) is computed from the gimbal FK chain
    using drone odometry + gimbal joint states.

3.  CENTROID BACK-PROJECTION — back-project the OBB centre pixel to the
    AR-panel plane (z = _CZ = 0.70 m).  The resulting panel_world position is
    the foundation of both the position estimate and the yaw estimate.

4.  USV YAW — back-project the OBB angle_deg as a world-space axis direction
    (1-pixel step from the OBB centre along the axis, onto the panel plane).
    Try 4 candidates (base_yaw + k×90°).  For each, project all 4 object
    corners and use the Hungarian algorithm (min-assignment) to match projected
    vs. detected corners.  Select the candidate with lowest mean reproj error.

5.  USV BASE_LINK POSITION — from the geometric estimate:
        usv_pos = panel_world − R_usv × _PANEL_CENTER_USV
    This corrects the 0.70 m vertical offset so the output is at the waterline.

6.  OUTPUT IN CAMERA FRAME:
        usv_in_cam = R_opt.inv() × (usv_pos − p_opt)
        R_usv_in_cam = R_opt.inv() × R_usv_est

7.  CENTROID FALLBACK (low altitude / high reproj error):
    When the reprojection error exceeds max_reproj_err_px, publish a position-
    only estimate (orientation = identity) from centroid back-projection with
    Z correction.  This ensures the navigator always gets a valid position fix.

8.  Compare with Gazebo ground truth (when available); publish error metrics.

Subscriptions
─────────────
  /usv_detection/result            ObbDetectionArray  — YOLO OBB detections
  /<ns>/overhead_cam/camera_info   CameraInfo         — camera intrinsics
  /<ns>/overhead_cam/image_raw     Image              — for debug viz
  /<ns>/ground_truth               Odometry           — UAV world pose
  /<ns>/gimbal/joint_states        JointState         — gimbal angles
  /gazebo/model_states             ModelStates        — USV ground-truth pose

Publications
────────────
  /yolo_pose/usv_in_cam    PoseStamped   USV centre in camera optical frame
                                          position: (x_cam, y_cam, z_cam) [m]
                                          orientation: USV yaw in cam frame
                                            (identity = yaw unknown / centroid mode)
  /yolo_pose/gt_usv_in_cam PoseStamped   Ground-truth USV in camera frame
  /yolo_pose/error         PointStamped  x=lateral_err  y=range_err  z=3d_err  [m]
  /yolo_pose/image         Image         debug: detected + reprojected corners
"""

import math
import numpy as np
import cv2
import rospy
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import linear_sum_assignment

from sensor_msgs.msg import Image, CameraInfo, JointState
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped, PointStamped, Point, Quaternion

from otter_usv_detector.msg import ObbDetectionArray

# ── Gimbal FK constants — identical to aruco_pose_test_node.py ───────────────
_HALF_PI   = math.pi / 2.0
_R_OPT     = Rot.from_euler('z', -_HALF_PI) * Rot.from_euler('x', -_HALF_PI)
_YAW_OFF   = np.array([0.0,   0.0,  -0.025])
_ROLL_OFF  = np.array([0.0,   0.0,  -0.030])
_PITCH_OFF = np.array([0.0,   0.0,  -0.025])
_OPT_OFF   = np.array([0.025, 0.0,   0.0  ])

# ── AR-panel geometry in Otter base_link frame ────────────────────────────────
# Source: otter_base.urdf.xacro  origin="-0.15 0 0.7"  scale="2.5 2.5 0.05"
# collection_params_gimbal.yaml: half_size_m=1.25  (±1.25 m → 2.5×2.5 m panel)
# The trained model (Nested_Otter_YOLO2) was trained on dataset_gimbal only,
# so the estimator must use half_size=1.25 to match training geometry.
# Corner order matches training labels (collect_dataset_gimbal.py):
#   fp = front-port      (+x +y)
#   fs = front-starboard (+x -y)
#   rs = rear-starboard  (-x -y)
#   rp = rear-port       (-x +y)
_CX, _CY, _CZ, _H = -0.15, 0.0, 0.70, 1.25
_OBJ_PTS = np.array([
    [_CX + _H,  _CY + _H,  _CZ],   # 0  fp
    [_CX + _H,  _CY - _H,  _CZ],   # 1  fs
    [_CX - _H,  _CY - _H,  _CZ],   # 2  rs
    [_CX - _H,  _CY + _H,  _CZ],   # 3  rp
], dtype=np.float64)

# Panel centre in USV frame — used to convert panel_world → usv_base_link
_PANEL_CENTER_USV = np.array([_CX, _CY, _CZ], dtype=np.float64)

_CORNER_LABELS = ['fp', 'fs', 'rs', 'rp']
_CORNER_COLORS = [
    (255, 255,   0),   # fp  yellow
    (  0, 255, 255),   # fs  cyan
    (255,   0, 255),   # rs  magenta
    (255, 128,   0),   # rp  orange
]


# ─────────────────────────────────────────────────────────────────────────────

def _camera_fk(drone_pos: np.ndarray,
               drone_rot: Rot,
               base_off: np.ndarray,
               yaw: float, roll: float, pitch: float):
    """Gimbal FK → (p_opt, R_opt).  Identical to aruco_pose_test_node."""
    p, R = drone_pos + drone_rot.apply(base_off), drone_rot
    p, R = p + R.apply(_YAW_OFF),   R * Rot.from_euler('z', yaw)
    p, R = p + R.apply(_ROLL_OFF),  R * Rot.from_euler('x', roll)
    p, R = p + R.apply(_PITCH_OFF), R * Rot.from_euler('y', pitch)
    p, R = p + R.apply(_OPT_OFF),   R * _R_OPT
    return p, R


def _back_project_to_z(px: np.ndarray, z_world: float,
                       p_opt: np.ndarray, R_opt: Rot,
                       K: np.ndarray):
    """
    Back-project image pixel px=(u,v) to a world-frame point at height z_world.
    Returns the 3-D world point, or None if the ray does not intersect the plane.
    """
    ray_cam   = np.array([(px[0] - K[0, 2]) / K[0, 0],
                           (px[1] - K[1, 2]) / K[1, 1],
                           1.0])
    ray_world = R_opt.apply(ray_cam)
    if abs(ray_world[2]) < 1e-6:
        return None
    t = (z_world - p_opt[2]) / ray_world[2]
    if t < 0:
        return None
    return p_opt + t * ray_world


def _project_to_image(pt_world: np.ndarray,
                      p_opt: np.ndarray, R_opt: Rot,
                      K: np.ndarray):
    """Project a 3-D world point into the image.  Returns (u, v) or None."""
    pt_cam = R_opt.inv().apply(pt_world - p_opt)
    if pt_cam[2] <= 0:
        return None
    u = K[0, 0] * pt_cam[0] / pt_cam[2] + K[0, 2]
    v = K[1, 1] * pt_cam[1] / pt_cam[2] + K[1, 2]
    return np.array([u, v])


def _angle_diff_deg(a: float, b: float) -> float:
    """Signed angular difference a−b, normalised to (−180, +180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def _to_imgmsg(bgr: np.ndarray, header) -> Image:
    msg = Image()
    msg.header   = header
    msg.height   = bgr.shape[0]
    msg.width    = bgr.shape[1]
    msg.encoding = 'bgr8'
    msg.step     = bgr.shape[1] * 3
    msg.data     = bgr.tobytes()
    return msg


# ─────────────────────────────────────────────────────────────────────────────

class YoloPoseEstimatorNode:

    def __init__(self):
        rospy.init_node('yolo_pose_estimator', anonymous=False)

        # ── Parameters ───────────────────────────────────────────────────────
        ns             = rospy.get_param('~gimbal_ns',         'uav1')
        self._ns       = ns
        self._usv_mdl  = rospy.get_param('~usv_model_name',   'otter')
        det_topic      = rospy.get_param('~detection_topic',   '/usv_detection/result')

        self._base_off = np.array([
            rospy.get_param('~base_x_offset', 0.10),
            rospy.get_param('~base_y_offset', 0.00),
            rospy.get_param('~base_z_offset', 0.00),
        ])

        self._conf_thr       = float(rospy.get_param('~conf_threshold',    0.50))
        self._max_reproj_err = float(rospy.get_param('~max_reproj_err_px', 25.0))

        # Camera intrinsics — overwritten by camera_info if published
        hfov  = float(rospy.get_param('~hfov_deg',    67.0))
        img_w = int(rospy.get_param('~image_width',   924))
        img_h = int(rospy.get_param('~image_height',  690))
        fx    = (img_w / 2.0) / math.tan(math.radians(hfov) / 2.0)
        self.K        = np.array([[fx, 0, img_w / 2.0],
                                   [0, fx, img_h / 2.0],
                                   [0,  0,          1.0]], dtype=np.float64)
        self.img_w    = img_w
        self.img_h    = img_h
        self._k_ready = False

        # ── State ─────────────────────────────────────────────────────────────
        self._drone_pos   = None
        self._drone_rot   = None
        self._g_yaw       = 0.0
        self._g_roll      = 0.0
        self._g_pitch     = 0.0
        self._usv_pos     = None   # from Gazebo model_states (GT only)
        self._usv_rot     = None
        self._latest_bgr  = None

        self._n_frames    = 0
        self._n_full      = 0   # estimates with full pose (position + yaw)
        self._n_centroid  = 0   # estimates with centroid fallback (position only)

        # Online Welford statistics for accepted estimates
        self._err_lat_mean = 0.0;  self._err_lat_M2 = 0.0
        self._err_3d_mean  = 0.0;  self._err_3d_M2  = 0.0
        self._n_stat = 0

        # ── Subscriptions ─────────────────────────────────────────────────────
        rospy.Subscriber(f'/{ns}/overhead_cam/camera_info', CameraInfo,
                         self._info_cb,  queue_size=1)
        rospy.Subscriber(f'/{ns}/overhead_cam/image_raw',   Image,
                         self._img_cb,   queue_size=1, buff_size=2**24)
        rospy.Subscriber(f'/{ns}/ground_truth',             Odometry,
                         self._odom_cb,  queue_size=1)
        rospy.Subscriber(f'/{ns}/gimbal/joint_states',      JointState,
                         self._js_cb,    queue_size=1)
        rospy.Subscriber('/gazebo/model_states',            ModelStates,
                         self._model_cb, queue_size=1)
        rospy.Subscriber(det_topic,                         ObbDetectionArray,
                         self._det_cb,   queue_size=1)

        # ── Publications ──────────────────────────────────────────────────────
        self._pub_pose  = rospy.Publisher('/yolo_pose/usv_in_cam',
                                          PoseStamped,  queue_size=5)
        self._pub_gt    = rospy.Publisher('/yolo_pose/gt_usv_in_cam',
                                          PoseStamped,  queue_size=5)
        self._pub_error = rospy.Publisher('/yolo_pose/error',
                                          PointStamped, queue_size=5)
        self._pub_image = rospy.Publisher('/yolo_pose/image',
                                          Image,        queue_size=1)

        rospy.loginfo(
            f'[yolo_pose] Node ready | ns={ns} | det={det_topic} | '
            f'conf≥{self._conf_thr:.2f} | reproj_lim={self._max_reproj_err:.0f}px')
        rospy.spin()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo):
        if not self._k_ready:
            self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
            self._k_ready = True
            rospy.loginfo(
                f'[yolo_pose] K from camera_info: '
                f'fx={self.K[0,0]:.1f}  cx={self.K[0,2]:.1f}  cy={self.K[1,2]:.1f}')

    def _img_cb(self, msg: Image):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1)
        if msg.encoding == 'bgr8':
            self._latest_bgr = arr.copy()
        elif msg.encoding == 'rgb8':
            self._latest_bgr = arr[:, :, ::-1].copy()

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._drone_pos = np.array([p.x, p.y, p.z])
        self._drone_rot = Rot.from_quat([q.x, q.y, q.z, q.w])

    def _js_cb(self, msg: JointState):
        ns  = self._ns
        lut = dict(zip(msg.name, msg.position))
        self._g_yaw   = lut.get(f'{ns}_gimbal_yaw_joint',   0.0)
        self._g_roll  = lut.get(f'{ns}_gimbal_roll_joint',   0.0)
        self._g_pitch = lut.get(f'{ns}_gimbal_pitch_joint',  0.0)

    def _model_cb(self, msg: ModelStates):
        if self._usv_mdl not in msg.name:
            return
        idx = msg.name.index(self._usv_mdl)
        p = msg.pose[idx].position
        q = msg.pose[idx].orientation
        self._usv_pos = np.array([p.x, p.y, p.z])
        self._usv_rot = Rot.from_quat([q.x, q.y, q.z, q.w])

    def _det_cb(self, msg: ObbDetectionArray):
        if self._drone_pos is None:
            rospy.logwarn_throttle(5.0, '[yolo_pose] Waiting for odometry …')
            return
        if not msg.detections:
            return

        self._n_frames += 1

        best = max(
            (d for d in msg.detections if d.confidence >= self._conf_thr),
            key=lambda d: d.confidence,
            default=None,
        )
        if best is None:
            return

        self._process(msg.header, best)

    # ── Core estimation ───────────────────────────────────────────────────────

    def _process(self, header, det):
        # ── Step 1: Camera pose in world from gimbal FK ───────────────────────
        p_opt, R_opt = _camera_fk(
            self._drone_pos, self._drone_rot, self._base_off,
            self._g_yaw, self._g_roll, self._g_pitch)

        corners = np.array(det.corners, dtype=np.float64).reshape(4, 2)
        occ     = corners.mean(axis=0)   # OBB centre pixel

        # ── Step 2: Centroid back-projection (always computable) ──────────────
        # Back-project OBB centre to the AR-panel plane (z = _CZ = 0.70 m).
        panel_world = _back_project_to_z(occ, _CZ, p_opt, R_opt, self.K)
        if panel_world is None:
            rospy.logwarn_throttle(2.0,
                '[yolo_pose] Back-projection failed (ray parallel to ground?)')
            return

        # ── Step 3: Full geometric estimate (yaw + accurate position) ─────────
        result = self._geometric_pose(corners, det.angle_deg, p_opt, R_opt,
                                      panel_world)
        if result is not None:
            usv_pos_est, usv_rot_est, reproj_err, assignment = result
        else:
            reproj_err = float('inf')
            usv_pos_est, usv_rot_est, assignment = None, None, None

        # ── Step 4: Decide output mode ────────────────────────────────────────
        if reproj_err <= self._max_reproj_err:
            # Full pose: accurate base_link position + estimated yaw
            usv_pos_out  = usv_pos_est         # base_link world, z ≈ 0
            usv_rot_out  = usv_rot_est          # USV yaw in world frame
            full_pose    = True
            self._n_full += 1
        else:
            # Centroid fallback: approximate position only, yaw unknown
            # Correct for panel height (_CZ = 0.70 m) to get waterline (z ≈ 0).
            # Horizontal offset _CX = -0.15 m is yaw-dependent but <0.15 m error.
            usv_pos_out  = panel_world - np.array([0.0, 0.0, _CZ])
            usv_rot_out  = Rot.identity()
            full_pose    = False
            self._n_centroid += 1
            if reproj_err < float('inf'):
                rospy.logwarn_throttle(2.0,
                    f'[yolo_pose] reproj={reproj_err:.1f}px > {self._max_reproj_err:.0f}px'
                    f' — centroid fallback (position only, yaw=unknown)')
            else:
                rospy.logwarn_throttle(2.0,
                    '[yolo_pose] Yaw estimation failed — centroid fallback')

        # ── Step 5: Express in camera optical frame ───────────────────────────
        usv_in_cam     = R_opt.inv().apply(usv_pos_out - p_opt)
        R_usv_in_cam   = R_opt.inv() * usv_rot_out

        # ── Publish ───────────────────────────────────────────────────────────
        self._pub_posestamped(header, self._pub_pose, usv_in_cam, R_usv_in_cam)

        # ── Ground truth + error (requires model_states) ──────────────────────
        if self._usv_pos is not None:
            # GT: USV base_link in camera frame
            gt_usv_in_cam = R_opt.inv().apply(self._usv_pos - p_opt)
            gt_usv_yaw    = float(np.degrees(
                self._usv_rot.as_euler('xyz')[2]))   # world-frame yaw [°]
            self._pub_posestamped(header, self._pub_gt,
                                   gt_usv_in_cam, R_opt.inv() * self._usv_rot)

            err      = usv_in_cam - gt_usv_in_cam
            err_lat  = float(np.linalg.norm(err[:2]))
            err_rng  = float(abs(err[2]))
            err_3d   = float(np.linalg.norm(err))
            self._pub_errormsg(header, err_lat, err_rng, err_3d)
            self._update_stats(err_lat, err_3d)

            if full_pose:
                est_yaw = float(np.degrees(usv_rot_out.as_euler('xyz')[2]))
                err_yaw = _angle_diff_deg(est_yaw, gt_usv_yaw)
                yaw_str = f'yaw={est_yaw:+.1f}° (err={err_yaw:+.1f}°)'
            else:
                yaw_str = 'yaw=N/A'

            mode_tag = 'FULL' if full_pose else 'CENTROID'
            rospy.loginfo(
                f'[yolo_pose|{mode_tag}] '
                f'usv_cam=({usv_in_cam[0]:+.2f},{usv_in_cam[1]:+.2f},{usv_in_cam[2]:+.2f})m  '
                f'gt_cam=({gt_usv_in_cam[0]:+.2f},{gt_usv_in_cam[1]:+.2f},{gt_usv_in_cam[2]:+.2f})m  '
                f'lat={err_lat:.3f}m rng={err_rng:.3f}m 3D={err_3d:.3f}m  '
                f'{yaw_str}  reproj={reproj_err:.1f}px  '
                f'conf={det.confidence:.2f} '
                f'[full={self._n_full}/cen={self._n_centroid}/{self._n_frames}]')
        else:
            mode_tag = 'FULL' if full_pose else 'CENTROID'
            rospy.loginfo_throttle(2.0,
                f'[yolo_pose|{mode_tag}] usv_in_cam='
                f'({usv_in_cam[0]:+.2f},{usv_in_cam[1]:+.2f},{usv_in_cam[2]:+.2f})m  '
                f'reproj={reproj_err:.1f}px [no GT]')

        if (self._n_full + self._n_centroid) % 60 == 0 and self._n_stat > 0:
            lat_std = math.sqrt(self._err_lat_M2 / self._n_stat)
            d3_std  = math.sqrt(self._err_3d_M2  / self._n_stat)
            rospy.loginfo(
                f'[yolo_pose] ── Stats ({self._n_stat} samples) ──  '
                f'lat mean={self._err_lat_mean:.3f}m std={lat_std:.3f}m  '
                f'3D mean={self._err_3d_mean:.3f}m std={d3_std:.3f}m')

        # ── Debug visualisation ───────────────────────────────────────────────
        if full_pose and assignment is not None:
            self._publish_viz(header, corners, assignment, usv_pos_est,
                              usv_rot_est, p_opt, R_opt, reproj_err,
                              det.confidence)
        else:
            self._publish_viz_centroid(header, corners, occ, panel_world,
                                       p_opt, R_opt, det.confidence)

    # ── Geometric pose estimation ─────────────────────────────────────────────

    def _geometric_pose(self, corners_4x2: np.ndarray, angle_deg: float,
                        p_opt: np.ndarray, R_opt: Rot,
                        panel_center_world: np.ndarray):
        """
        Estimate USV yaw and accurate base_link position.

        Uses the OBB angle_deg to derive a world-space axis direction (by
        back-projecting a 1-px step along the image axis onto the panel plane).
        Tries 4 candidate yaws (base + k×90°).  For each, projects all 4
        object corners and uses the Hungarian algorithm to find the optimal
        corner assignment.  Returns the candidate with lowest mean reproj error.

        Returns (usv_pos, usv_rot, reproj_err, assignment) or None.
          usv_pos: base_link world position (z ≈ 0 = waterline)
          usv_rot: USV yaw rotation in world frame
          reproj_err: mean corner reprojection error [px]
          assignment: col_ind[i] = detected corner index for OBJ_PTS[i]
        """
        K = self.K
        occ = corners_4x2.mean(axis=0)

        # OBB axis direction in world via back-projection
        angle_rad   = math.radians(angle_deg)
        axis_tip_px = occ + np.array([math.cos(angle_rad), math.sin(angle_rad)])
        axis_tip_w  = _back_project_to_z(axis_tip_px, _CZ, p_opt, R_opt, K)
        if axis_tip_w is None:
            return None
        axis_dir = axis_tip_w[:2] - panel_center_world[:2]
        if np.linalg.norm(axis_dir) < 1e-6:
            return None
        base_yaw = math.atan2(axis_dir[1], axis_dir[0])

        best     = None
        best_err = float('inf')

        for k in range(4):
            usv_yaw = base_yaw + k * math.pi / 2.0
            usv_rot = Rot.from_euler('z', usv_yaw)
            # base_link: remove panel offset from panel_center_world
            usv_pos = panel_center_world - usv_rot.apply(_PANEL_CENTER_USV)

            # Project all 4 object corners
            proj_pts = []
            valid    = True
            for obj_pt in _OBJ_PTS:
                px = _project_to_image(usv_pos + usv_rot.apply(obj_pt),
                                       p_opt, R_opt, K)
                if px is None:
                    valid = False
                    break
                proj_pts.append(px)
            if not valid:
                continue
            proj_pts = np.array(proj_pts)   # (4, 2)

            # Hungarian min-assignment: cost[i,j] = proj_pts[i] ↔ corners[j]
            cost = np.linalg.norm(
                proj_pts[:, None, :] - corners_4x2[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)
            reproj_err = float(cost[row_ind, col_ind].mean())

            if reproj_err < best_err:
                best_err = reproj_err
                best = (usv_pos, usv_rot, reproj_err, col_ind.copy())

        return best

    # ── Publishers ────────────────────────────────────────────────────────────

    def _pub_posestamped(self, header, pub, pos: np.ndarray, rot: Rot):
        q   = rot.as_quat()
        msg = PoseStamped()
        msg.header.stamp    = header.stamp
        msg.header.frame_id = 'overhead_cam_optical'
        msg.pose.position   = Point(
            x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        msg.pose.orientation = Quaternion(
            x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
        pub.publish(msg)

    def _pub_errormsg(self, header, err_lat, err_rng, err_3d):
        msg = PointStamped()
        msg.header.stamp    = header.stamp
        msg.header.frame_id = 'overhead_cam_optical'
        msg.point.x = err_lat
        msg.point.y = err_rng
        msg.point.z = err_3d
        self._pub_error.publish(msg)

    # ── Debug visualisation ───────────────────────────────────────────────────

    def _publish_viz(self, header, corners_4x2: np.ndarray,
                     assignment: np.ndarray,
                     usv_pos: np.ndarray, usv_rot: Rot,
                     p_opt: np.ndarray, R_opt: Rot,
                     reproj_err: float, conf: float):
        """Full-pose visualisation: colour-coded corners + bow arrow."""
        if self._latest_bgr is None:
            return
        vis = self._latest_bgr.copy()

        # Detected corners colour-coded by assigned label
        for i in range(4):
            pt  = corners_4x2[assignment[i]].astype(np.int32)
            col = _CORNER_COLORS[i]
            cv2.circle(vis, tuple(pt), 7, col, -1)
            cv2.putText(vis, _CORNER_LABELS[i], (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        ring = corners_4x2[assignment].astype(np.int32)
        cv2.polylines(vis, [ring.reshape(-1, 1, 2)],
                      isClosed=True, color=(0, 255, 0), thickness=2)

        # Reprojected corners from estimated pose
        for obj_pt in _OBJ_PTS:
            px = _project_to_image(usv_pos + usv_rot.apply(obj_pt),
                                   p_opt, R_opt, self.K)
            if px is not None:
                cv2.drawMarker(vis, (int(px[0]), int(px[1])), (0, 128, 255),
                               cv2.MARKER_CROSS, 14, 2)

        # Bow direction arrow
        bow_tail = _project_to_image(
            usv_pos + usv_rot.apply(np.array([_CX, _CY, _CZ])),
            p_opt, R_opt, self.K)
        bow_tip  = _project_to_image(
            usv_pos + usv_rot.apply(np.array([_CX + 2.0, _CY, _CZ])),
            p_opt, R_opt, self.K)
        if bow_tail is not None and bow_tip is not None:
            cv2.arrowedLine(vis,
                            (int(bow_tail[0]), int(bow_tail[1])),
                            (int(bow_tip[0]),  int(bow_tip[1])),
                            (0, 200, 0), 2, tipLength=0.3)

        # Legend
        for i, (lbl, col) in enumerate(zip(_CORNER_LABELS, _CORNER_COLORS)):
            cv2.putText(vis, lbl, (8, self.img_h - 10 - i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.putText(vis, 'O=detected  +=reprojected',
                    (8, self.img_h - 10 - 4 * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(vis,
            f'FULL POSE | conf={conf:.2f} reproj={reproj_err:.1f}px',
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        if self._n_stat > 0:
            cv2.putText(vis,
                f'3D mean={self._err_3d_mean:.3f}m '
                f'[{self._n_full}+{self._n_centroid}/{self._n_frames}]',
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        self._pub_image.publish(_to_imgmsg(vis, header))

    def _publish_viz_centroid(self, header, corners_4x2: np.ndarray,
                              occ: np.ndarray, panel_world: np.ndarray,
                              p_opt: np.ndarray, R_opt: Rot,
                              conf: float):
        """Simplified visualisation for centroid-fallback estimates."""
        if self._latest_bgr is None:
            return
        vis = self._latest_bgr.copy()

        # Draw OBB outline (uncoloured — no corner assignment)
        ring = corners_4x2.astype(np.int32)
        cv2.polylines(vis, [ring.reshape(-1, 1, 2)],
                      isClosed=True, color=(0, 200, 255), thickness=2)

        # Mark the OBB centre
        c = occ.astype(np.int32)
        cv2.drawMarker(vis, tuple(c), (0, 200, 255),
                       cv2.MARKER_CROSS, 20, 2)

        # Project the estimated USV position (waterline)
        usv_approx = panel_world - np.array([0.0, 0.0, _CZ])
        px_usv = _project_to_image(usv_approx, p_opt, R_opt, self.K)
        if px_usv is not None:
            cv2.circle(vis, (int(px_usv[0]), int(px_usv[1])),
                       10, (0, 100, 255), -1)

        cv2.putText(vis,
            f'CENTROID | conf={conf:.2f} | yaw=unknown',
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

        self._pub_image.publish(_to_imgmsg(vis, header))

    # ── Online statistics (Welford) ───────────────────────────────────────────

    def _update_stats(self, err_lat: float, err_3d: float):
        self._n_stat += 1
        n = self._n_stat
        for val, mean_attr, m2_attr in [
            (err_lat, '_err_lat_mean', '_err_lat_M2'),
            (err_3d,  '_err_3d_mean',  '_err_3d_M2'),
        ]:
            mean = getattr(self, mean_attr)
            m2   = getattr(self, m2_attr)
            d = val - mean
            mean += d / n
            m2   += d * (val - mean)
            setattr(self, mean_attr, mean)
            setattr(self, m2_attr,   m2)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        YoloPoseEstimatorNode()
    except rospy.ROSInterruptException:
        pass

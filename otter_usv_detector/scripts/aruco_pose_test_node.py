#!/usr/bin/env python3
"""
aruco_pose_test_node.py — Nested marker detection & pose error analysis.

Detects two fiducial markers on the Otter USV ar_code panel (Target_2_board.png)
as seen by the Gremsy G-Hadron gimbal camera (FLIR Hadron 640R):

  Outer  DICT_APRILTAG_36h11  ID=10   physical side = 2.000 m
  Inner  DICT_4X4_50          ID=1    physical side = 0.200 m  (nested inside outer)

Panel is 2.5 m × 2.5 m (COLLADA ×0.5 scale + URDF scale="2.5 2.5 0.05").
URDF rpy="0 0 M_PI/2" orients the board so the USV bow (+X) is at the top of
the texture when viewed from above.

Pose estimation pipeline (per detected marker)
  1. solvePnP (IPPE_SQUARE) → tvec in camera optical frame
  2. Forward kinematics (drone odom + gimbal joint states) → (p_opt, R_opt)
  3. marker_world = p_opt + R_opt @ tvec

Error analysis
  Estimated marker_world compared against Gazebo ground truth:
    model "otter" pose  = ar_code link world position (ar_code is the URDF root link)
    marker centre offset = [-0.15, 0, 0.75] m in ar_code body frame
  Error reported as XY (horizontal), Z (vertical), and 3-D Euclidean distance.

Subscriptions
  /<ns>/overhead_cam/image_raw     sensor_msgs/Image
  /<ns>/overhead_cam/camera_info   sensor_msgs/CameraInfo
  /<ns>/ground_truth               nav_msgs/Odometry
  /<ns>/gimbal/joint_states        sensor_msgs/JointState
  /gazebo/model_states             gazebo_msgs/ModelStates

Publications
  /aruco_test/image                sensor_msgs/Image           annotated frame
  /aruco_test/outer/pose           geometry_msgs/PoseStamped   outer marker world pose
  /aruco_test/inner/pose           geometry_msgs/PoseStamped   inner marker world pose
  /aruco_test/error                geometry_msgs/PointStamped
                                     .x = xy error  [m]
                                     .y = z  error  [m]
                                     .z = 3D error  [m]
"""

import math
import numpy as np
import rospy
import cv2
import cv_bridge

from sensor_msgs.msg import Image, CameraInfo, JointState
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped, PointStamped, Point, Quaternion

from scipy.spatial.transform import Rotation as Rot

# Dictionary IDs
_DICT_ARUCO  = cv2.aruco.DICT_4X4_50
_DICT_APRIL  = cv2.aruco.DICT_APRILTAG_36h11
_OUTER_ID    = 10   # AprilTag 36h11 ID=10
_INNER_ID    = 1    # ArUco 4x4_50  ID=1

# ── Gimbal FK constants — must match gimbal_position_node.py ─────────────────
_HALF_PI           = math.pi / 2.0
_R_OPT             = Rot.from_euler('z', -_HALF_PI) * Rot.from_euler('x', -_HALF_PI)
_GIMBAL_YAW_OFF    = np.array([0.0,   0.0,  -0.025])
_GIMBAL_ROLL_OFF   = np.array([0.0,   0.0,  -0.030])
_GIMBAL_PITCH_OFF  = np.array([0.0,   0.0,  -0.025])
_GIMBAL_OPT_OFF    = np.array([0.025, 0.0,   0.0  ])


def _camera_optical_world_pose(drone_pos, drone_rot, base_offset,
                                yaw, roll, pitch):
    """Compute camera optical centre world position and rotation (optical→world)."""
    R_d    = drone_rot
    p_base = drone_pos + R_d.apply(base_offset)
    R_base = R_d

    p_yaw  = p_base + R_base.apply(_GIMBAL_YAW_OFF)
    R_yaw  = R_base * Rot.from_euler('z', yaw)

    p_roll = p_yaw  + R_yaw.apply(_GIMBAL_ROLL_OFF)
    R_roll = R_yaw  * Rot.from_euler('x', roll)

    p_cam  = p_roll + R_roll.apply(_GIMBAL_PITCH_OFF)
    R_cam  = R_roll * Rot.from_euler('y', pitch)

    p_opt  = p_cam  + R_cam.apply(_GIMBAL_OPT_OFF)
    R_opt  = R_cam  * _R_OPT
    return p_opt, R_opt


# ── Online statistics (Welford) ───────────────────────────────────────────────

class _Stats:
    def __init__(self, label):
        self.label = label
        self.n     = 0
        self.mean  = 0.0
        self._M2   = 0.0

    def push(self, x):
        self.n  += 1
        d        = x - self.mean
        self.mean += d / self.n
        self._M2  += d * (x - self.mean)

    @property
    def std(self):
        return math.sqrt(self._M2 / self.n) if self.n > 1 else 0.0

    def summary(self):
        return (f'{self.label}: n={self.n}  '
                f'mean={self.mean:.4f} m  std={self.std:.4f} m')


# ── Node ─────────────────────────────────────────────────────────────────────

class ArucoPoseTestNode:

    # Physical marker sizes on the 2.5 m × 2.5 m ar_code panel (Target_2_board.png)
    _OUTER_SIZE_M = 2000.0 / 2500.0 * 2.5  # = 2.000 m  (outer AprilTag 36h11 ID=10)
    _INNER_SIZE_M =  200.0 / 2500.0 * 2.5  # = 0.200 m  (inner ArUco 4x4_50 ID=1)

    # ar_code visual centre in ar_code link frame
    # (visual origin xyz="-0.15 0 0.7", mesh top face 0.05 m above origin → 0.75 m)
    _AR_OFFSET = np.array([-0.15, 0.0, 0.75])

    def __init__(self):
        rospy.init_node('aruco_pose_test_node', anonymous=False)

        ns            = rospy.get_param('~gimbal_ns',       'uav1')
        self._ns      = ns
        usv_model     = rospy.get_param('~usv_model_name',  'otter')
        self._usv_mdl = usv_model

        x_off = rospy.get_param('~base_x_offset', 0.10)
        y_off = rospy.get_param('~base_y_offset', 0.00)
        z_off = rospy.get_param('~base_z_offset', 0.00)
        self._base_off = np.array([x_off, y_off, z_off])

        outer_size = rospy.get_param('~outer_marker_size_m', self._OUTER_SIZE_M)
        inner_size = rospy.get_param('~inner_marker_size_m', self._INNER_SIZE_M)
        self._sizes = {0: float(outer_size), 1: float(inner_size)}

        # ── Detectors ────────────────────────────────────────────────────────
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin  = 3
        params.adaptiveThreshWinSizeMax  = 53
        params.adaptiveThreshWinSizeStep = 10
        params.minMarkerPerimeterRate    = 0.02
        params.cornerRefinementMethod    = cv2.aruco.CORNER_REFINE_SUBPIX
        self._aruco_det = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(_DICT_ARUCO), params)
        self._april_det = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(_DICT_APRIL), params)

        # ── State ─────────────────────────────────────────────────────────────
        self._bridge      = cv_bridge.CvBridge()
        self._cam_K       = None
        self._cam_D       = None
        self._drone_pos   = None
        self._drone_rot   = None
        self._yaw         = 0.0
        self._roll        = 0.0
        self._pitch       = 0.0
        self._usv_pos     = None
        self._usv_rot     = None

        # Per-marker statistics: outer AprilTag (ID=10), inner ArUco (ID=1)
        self._stats = {
            _OUTER_ID: {'xy': _Stats('outer XY'), 'z': _Stats('outer Z'),
                        '3d': _Stats('outer 3D'), 'det': 0},
            _INNER_ID: {'xy': _Stats('inner XY'), 'z': _Stats('inner Z'),
                        '3d': _Stats('inner 3D'), 'det': 0},
        }
        self._frames = 0

        # ── Subscriptions ─────────────────────────────────────────────────────
        rospy.Subscriber(f'/{ns}/overhead_cam/image_raw',   Image,
                         self._img_cb,   queue_size=1, buff_size=2**24)
        rospy.Subscriber(f'/{ns}/overhead_cam/camera_info', CameraInfo,
                         self._info_cb,  queue_size=1)
        rospy.Subscriber(f'/{ns}/ground_truth',             Odometry,
                         self._odom_cb,  queue_size=1)
        rospy.Subscriber(f'/{ns}/gimbal/joint_states',      JointState,
                         self._js_cb,    queue_size=1)
        rospy.Subscriber('/gazebo/model_states',            ModelStates,
                         self._model_cb, queue_size=1)

        # ── Publications ──────────────────────────────────────────────────────
        self._pub_img        = rospy.Publisher('/aruco_test/image',
                                               Image,         queue_size=1)
        self._pub_outer_pose = rospy.Publisher('/aruco_test/outer/pose',
                                               PoseStamped,   queue_size=5)
        self._pub_inner_pose = rospy.Publisher('/aruco_test/inner/pose',
                                               PoseStamped,   queue_size=5)
        self._pub_error      = rospy.Publisher('/aruco_test/error',
                                               PointStamped,  queue_size=5)

        rospy.loginfo('[aruco_test] Node ready — waiting for camera_info + odom …')
        rospy.loginfo(f'[aruco_test] Outer: DICT_APRILTAG_36h11 ID={_OUTER_ID} {outer_size:.3f}m | '
                      f'Inner: DICT_4X4_50 ID={_INNER_ID} {inner_size:.3f}m')
        rospy.spin()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo):
        if self._cam_K is None:
            self._cam_K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
            self._cam_D = np.array(msg.D, dtype=np.float64)
            rospy.loginfo(
                f'[aruco_test] Camera intrinsics: '
                f'fx={self._cam_K[0,0]:.1f} fy={self._cam_K[1,1]:.1f} '
                f'cx={self._cam_K[0,2]:.1f} cy={self._cam_K[1,2]:.1f}')

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._drone_pos = np.array([p.x, p.y, p.z])
        self._drone_rot = Rot.from_quat([q.x, q.y, q.z, q.w])

    def _js_cb(self, msg: JointState):
        ns = self._ns
        lut = dict(zip(msg.name, msg.position))
        self._yaw   = lut.get(f'{ns}_gimbal_yaw_joint',   0.0)
        self._roll  = lut.get(f'{ns}_gimbal_roll_joint',   0.0)
        self._pitch = lut.get(f'{ns}_gimbal_pitch_joint',  0.0)

    def _model_cb(self, msg: ModelStates):
        if self._usv_mdl not in msg.name:
            return
        idx = msg.name.index(self._usv_mdl)
        p = msg.pose[idx].position
        q = msg.pose[idx].orientation
        self._usv_pos = np.array([p.x, p.y, p.z])
        self._usv_rot = Rot.from_quat([q.x, q.y, q.z, q.w])

    def _img_cb(self, msg: Image):
        if self._cam_K is None or self._drone_pos is None:
            rospy.logwarn_throttle(5.0,
                '[aruco_test] Waiting for camera_info + odometry …')
            return

        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except cv_bridge.CvBridgeError as e:
            rospy.logerr(f'[aruco_test] cv_bridge: {e}')
            return

        self._frames += 1
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        vis  = bgr.copy()

        # ── FK → camera world pose ────────────────────────────────────────────
        p_opt, R_opt = _camera_optical_world_pose(
            self._drone_pos, self._drone_rot, self._base_off,
            self._yaw, self._roll, self._pitch)

        # ── Detect — run both dictionaries ───────────────────────────────────
        april_c, april_ids, april_rej = self._april_det.detectMarkers(gray)
        aruco_c, aruco_ids, aruco_rej = self._aruco_det.detectMarkers(gray)

        if april_ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, april_c, april_ids, (0, 200, 0))
        if aruco_ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, aruco_c, aruco_ids, (0, 128, 255))

        # Merge into a single iterable (corners, marker_id, size_m)
        all_detections = []
        if april_ids is not None:
            for c, i in zip(april_c, april_ids):
                mid = int(i[0])
                if mid in self._sizes:
                    all_detections.append((c, mid))
        if aruco_ids is not None:
            for c, i in zip(aruco_c, aruco_ids):
                mid = int(i[0])
                if mid in self._sizes:
                    all_detections.append((c, mid))

        rejected_total = (len(april_rej) if april_rej else 0) + \
                         (len(aruco_rej) if aruco_rej else 0)

        # ── Estimate pose for each detected marker ─────────────────────────────
        for corners, marker_id in all_detections:

            size_m = self._sizes[marker_id]
            ok, rvec, tvec = self._solvepnp(corners, size_m)
            if not ok:
                continue

            # Draw axes (length = half marker side)
            cv2.drawFrameAxes(vis, self._cam_K, self._cam_D,
                              rvec, tvec, size_m * 0.5)

            # Transform to world
            t_flat = tvec.flatten()
            dist   = float(np.linalg.norm(t_flat))
            if dist > 200.0 or dist < 0.01:
                rospy.logwarn_throttle(2.0,
                    f'[aruco_test] ID={marker_id} implausible dist={dist:.1f}m')
                continue

            est_world = p_opt + R_opt.apply(t_flat)
            self._stats[marker_id]['det'] += 1

            # Publish pose
            pub = self._pub_inner_pose if marker_id == _INNER_ID else self._pub_outer_pose
            self._pub_pose_msg(msg.header, pub, est_world, R_opt)

            # Annotate image
            c_mean = corners[0].mean(axis=0).astype(int)
            label  = f"ID={marker_id} {dist:.1f}m"
            cv2.putText(vis, label, tuple(c_mean),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Error vs ground truth
            if self._usv_pos is not None:
                gt_world = self._usv_pos + self._usv_rot.apply(self._AR_OFFSET)
                err_xy = float(np.linalg.norm(est_world[:2] - gt_world[:2]))
                err_z  = float(abs(est_world[2] - gt_world[2]))
                err_3d = float(np.linalg.norm(est_world - gt_world))

                self._stats[marker_id]['xy'].push(err_xy)
                self._stats[marker_id]['z'].push(err_z)
                self._stats[marker_id]['3d'].push(err_3d)

                self._pub_error_msg(msg.header, err_xy, err_z, err_3d)

                rospy.loginfo(
                    f'[aruco_test] ID={marker_id} sz={size_m:.3f}m | '
                    f'est ({est_world[0]:.2f},{est_world[1]:.2f},{est_world[2]:.2f}) '
                    f'gt  ({gt_world[0]:.2f},{gt_world[1]:.2f},{gt_world[2]:.2f}) | '
                    f'XY={err_xy:.3f}m  Z={err_z:.3f}m  3D={err_3d:.3f}m')
            else:
                rospy.loginfo_throttle(2.0,
                    f'[aruco_test] ID={marker_id} dist={dist:.2f}m '
                    f'world=({est_world[0]:.2f},{est_world[1]:.2f},{est_world[2]:.2f}) '
                    f'[no GT]')

        # ── HUD overlay ───────────────────────────────────────────────────────
        r_out = self._stats[_OUTER_ID]['det'] / self._frames * 100 if self._frames else 0
        r_in  = self._stats[_INNER_ID]['det'] / self._frames * 100 if self._frames else 0
        cv2.putText(vis,
                    f'Frame {self._frames}  '
                    f'April(ID={_OUTER_ID}): {r_out:.0f}%  ArUco(ID={_INNER_ID}): {r_in:.0f}%',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(vis,
                    f'gimbal pitch={self._pitch:.2f} rad  rejected={rejected_total}',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Log stats every 60 frames
        if self._frames % 60 == 0:
            self._log_stats()

        try:
            self._pub_img.publish(self._bridge.cv2_to_imgmsg(vis, 'bgr8'))
        except cv_bridge.CvBridgeError:
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _solvepnp(self, corners_arr, size_m):
        """solvePnP with IPPE_SQUARE. Returns (ok, rvec, tvec)."""
        half    = size_m / 2.0
        obj_pts = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float64)
        img_pts = corners_arr[0].astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self._cam_K, self._cam_D,
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
        return ok, rvec, tvec

    def _pub_pose_msg(self, header, pub, world_pos, R_cam_world):
        q = R_cam_world.as_quat()
        msg = PoseStamped()
        msg.header.stamp    = header.stamp
        msg.header.frame_id = 'world'
        msg.pose.position   = Point(x=float(world_pos[0]),
                                    y=float(world_pos[1]),
                                    z=float(world_pos[2]))
        msg.pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]),
                                          z=float(q[2]), w=float(q[3]))
        pub.publish(msg)

    def _pub_error_msg(self, header, err_xy, err_z, err_3d):
        msg = PointStamped()
        msg.header.stamp    = header.stamp
        msg.header.frame_id = 'world'
        msg.point.x = err_xy
        msg.point.y = err_z
        msg.point.z = err_3d
        self._pub_error.publish(msg)

    def _log_stats(self):
        labels = {_OUTER_ID: 'AprilTag36h11', _INNER_ID: 'ArUco4x4_50'}
        rospy.loginfo('[aruco_test] ══ Statistics ══')
        for mid, s in self._stats.items():
            rate = s['det'] / self._frames * 100 if self._frames else 0
            rospy.loginfo(f'[aruco_test]  {labels[mid]} ID={mid}: '
                          f'{s["det"]}/{self._frames} frames ({rate:.1f}%)')
            for key in ('xy', 'z', '3d'):
                if s[key].n > 0:
                    rospy.loginfo(f'[aruco_test]    {s[key].summary()}')


if __name__ == '__main__':
    try:
        ArucoPoseTestNode()
    except rospy.ROSInterruptException:
        pass

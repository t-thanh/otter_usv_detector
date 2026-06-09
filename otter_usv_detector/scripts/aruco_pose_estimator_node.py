#!/usr/bin/env python3
"""
aruco_pose_estimator_node.py
─────────────────────────────
Full-pose estimator for the nested ArUco/AprilTag markers on the Otter USV.

Detects two fiducial markers on the Otter's ar_code panel (Target_2_board.png):
  Outer  DICT_APRILTAG_36h11  ID=10   physical side = 2.000 m
  Inner  DICT_4X4_50          ID=1    physical side = 0.200 m

For each detected marker the node:
  1. Runs solvePnP (IPPE_SQUARE) to get marker pose in camera optical frame.
  2. Applies a fixed Rz(+π/2) correction to align the solvePnP marker frame
     with the USV body frame.  The URDF visual has rpy="0 0 π/2", so the
     marker face frame is Rz(−π/2) relative to the USV body frame.  The
     inverse correction Rz(+π/2) is applied to recover USV orientation.
     Without this correction the published yaw would always read −90° relative
     to the true USV heading and the Euler roll/pitch would show large apparent
     errors that are entirely artefacts of the yaw offset.
  3. Publishes PoseStamped in the overhead_cam_optical frame:
       position    = AR-panel visual centre in camera optical frame
       orientation = USV body frame orientation in camera optical frame

Topic layout (per marker type)
  /aruco_pose/outer/usv_in_cam   — outer AprilTag 36h11  (ID=10, 2.0 m)
  /aruco_pose/inner/usv_in_cam   — inner ArUco 4×4_50   (ID=1,  0.2 m)

Both use frame_id = 'overhead_cam_optical'.

Subscriptions (same as yolo_pose_estimator_node)
  /<ns>/overhead_cam/image_raw      sensor_msgs/Image
  /<ns>/overhead_cam/camera_info    sensor_msgs/CameraInfo
  /<ns>/ground_truth                nav_msgs/Odometry
  /<ns>/gimbal/joint_states         sensor_msgs/JointState

Parameters
  ~gimbal_ns          UAV namespace          (default: uav1)
  ~base_x/y/z_offset  gimbal base in drone   (default: 0.10, 0.00, 0.00)
  ~outer_size_m       outer marker side [m]  (default: 2.000)
  ~inner_size_m       inner marker side [m]  (default: 0.200)
"""

import math
import numpy as np
import rospy
import cv2
import cv_bridge

from sensor_msgs.msg import Image, CameraInfo, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from scipy.spatial.transform import Rotation as Rot

# ── Marker geometry ───────────────────────────────────────────────────────────
_DICT_ARUCO = cv2.aruco.DICT_4X4_50
_DICT_APRIL = cv2.aruco.DICT_APRILTAG_36h11
_OUTER_ID   = 10   # AprilTag 36h11
_INNER_ID   = 1    # ArUco 4×4_50

# Default physical sizes (from URDF scale="2.5 2.5 0.05", texture resolution 2500px)
#   Outer: 2000 px / 2500 px × 2.5 m = 2.000 m
#   Inner:  200 px / 2500 px × 2.5 m = 0.200 m
_OUTER_SIZE_DEFAULT = 2000.0 / 2500.0 * 2.5   # 2.000 m
_INNER_SIZE_DEFAULT =  200.0 / 2500.0 * 2.5   # 0.200 m

# AR-panel visual centre in the Otter ar_code link frame.
# URDF: <visual origin xyz="-0.15 0 0.7" rpy="0 0 M_PI/2"/>
#       mesh top face = 0.7 + 0.05 (slab thickness) = 0.75 m
# ar_code is the URDF root link; Gazebo reports its pose as the "otter" model state.
_AR_CENTER_IN_USV = np.array([-0.15, 0.0, 0.75], dtype=np.float64)

# ── Gimbal FK constants (must match gimbal_position_node.py exactly) ──────────
_HALF_PI   = math.pi / 2.0
_R_OPT     = Rot.from_euler('z', -_HALF_PI) * Rot.from_euler('x', -_HALF_PI)

# Fixed correction: rotate solvePnP marker frame → USV body frame.
# Source: URDF <visual rpy="0 0 π/2"/> means the marker face frame is rotated
# Rz(+90°) relative to the USV body frame, i.e. R_USV_TO_MARKER = Rz(-90°).
# To recover USV orientation: R_usv_in_cam = R_marker_in_cam * R_USV_TO_MARKER.inv()
#                                          = R_marker_in_cam * Rz(+90°)
_R_MARKER_TO_USV = Rot.from_euler('z', +_HALF_PI)
_YAW_OFF   = np.array([0.0,   0.0,  -0.025])
_ROLL_OFF  = np.array([0.0,   0.0,  -0.030])
_PITCH_OFF = np.array([0.0,   0.0,  -0.025])
_OPT_OFF   = np.array([0.025, 0.0,   0.0  ])


def _camera_fk(drone_pos, drone_rot, base_off, yaw, roll, pitch):
    """Gimbal FK → (p_opt, R_opt) — optical centre world position and rotation."""
    p, R = drone_pos + drone_rot.apply(base_off), drone_rot
    p, R = p + R.apply(_YAW_OFF),   R * Rot.from_euler('z', yaw)
    p, R = p + R.apply(_ROLL_OFF),  R * Rot.from_euler('x', roll)
    p, R = p + R.apply(_PITCH_OFF), R * Rot.from_euler('y', pitch)
    p, R = p + R.apply(_OPT_OFF),   R * _R_OPT
    return p, R


class ArucoPoseEstimatorNode:

    # solvePnP object points for a flat square marker.
    # Convention: x=right, y=up, z=outward (toward camera).
    # Matches cv2.aruco corner ordering: TL, TR, BR, BL.
    @staticmethod
    def _obj_pts(half):
        return np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float64)

    def __init__(self):
        rospy.init_node('aruco_pose_estimator_node', anonymous=False)

        ns             = rospy.get_param('~gimbal_ns',      'uav1')
        self._ns       = ns
        self._base_off = np.array([
            rospy.get_param('~base_x_offset', 0.10),
            rospy.get_param('~base_y_offset', 0.00),
            rospy.get_param('~base_z_offset', 0.00),
        ])
        self._sizes = {
            _OUTER_ID: float(rospy.get_param('~outer_size_m', _OUTER_SIZE_DEFAULT)),
            _INNER_ID: float(rospy.get_param('~inner_size_m', _INNER_SIZE_DEFAULT)),
        }

        # ── ArUco / AprilTag detectors ────────────────────────────────────────
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin  = 3
        params.adaptiveThreshWinSizeMax  = 53
        params.adaptiveThreshWinSizeStep = 10
        params.minMarkerPerimeterRate    = 0.02
        params.cornerRefinementMethod    = cv2.aruco.CORNER_REFINE_SUBPIX
        self._april_det = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(_DICT_APRIL), params)
        self._aruco_det = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(_DICT_ARUCO), params)

        # ── State ─────────────────────────────────────────────────────────────
        self._bridge    = cv_bridge.CvBridge()
        self._K         = None
        self._D         = None
        self._drone_pos = None
        self._drone_rot = None
        self._g_yaw     = 0.0
        self._g_roll    = 0.0
        self._g_pitch   = 0.0
        self._n_frames  = 0
        self._n_outer   = 0
        self._n_inner   = 0

        # ── Subscriptions ─────────────────────────────────────────────────────
        rospy.Subscriber(f'/{ns}/overhead_cam/image_raw',   Image,
                         self._img_cb,  queue_size=1, buff_size=2**24)
        rospy.Subscriber(f'/{ns}/overhead_cam/camera_info', CameraInfo,
                         self._info_cb, queue_size=1)
        rospy.Subscriber(f'/{ns}/ground_truth',             Odometry,
                         self._odom_cb, queue_size=1)
        rospy.Subscriber(f'/{ns}/gimbal/joint_states',      JointState,
                         self._js_cb,   queue_size=5)

        # ── Publications ──────────────────────────────────────────────────────
        self._pub = {
            _OUTER_ID: rospy.Publisher('/aruco_pose/outer/usv_in_cam',
                                       PoseStamped, queue_size=5),
            _INNER_ID: rospy.Publisher('/aruco_pose/inner/usv_in_cam',
                                       PoseStamped, queue_size=5),
        }

        rospy.loginfo(
            f'[aruco_pose_est] Ready — ns={ns} | '
            f'outer_id={_OUTER_ID} {self._sizes[_OUTER_ID]:.3f}m | '
            f'inner_id={_INNER_ID} {self._sizes[_INNER_ID]:.3f}m')
        rospy.spin()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo):
        if self._K is None:
            self._K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
            self._D = np.array(msg.D, dtype=np.float64)
            rospy.loginfo(
                f'[aruco_pose_est] K: fx={self._K[0,0]:.1f} '
                f'cx={self._K[0,2]:.1f} cy={self._K[1,2]:.1f}')

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._drone_pos = np.array([p.x, p.y, p.z])
        self._drone_rot = Rot.from_quat([q.x, q.y, q.z, q.w])

    def _js_cb(self, msg: JointState):
        lut = dict(zip(msg.name, msg.position))
        ns  = self._ns
        self._g_yaw   = lut.get(f'{ns}_gimbal_yaw_joint',   0.0)
        self._g_roll  = lut.get(f'{ns}_gimbal_roll_joint',  0.0)
        self._g_pitch = lut.get(f'{ns}_gimbal_pitch_joint', 0.0)

    def _img_cb(self, msg: Image):
        if self._K is None or self._drone_pos is None:
            rospy.logwarn_throttle(5.0,
                '[aruco_pose_est] Waiting for camera_info + odometry …')
            return

        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except cv_bridge.CvBridgeError as e:
            rospy.logerr(f'[aruco_pose_est] cv_bridge: {e}')
            return

        self._n_frames += 1

        # Camera FK
        p_opt, R_opt = _camera_fk(
            self._drone_pos, self._drone_rot, self._base_off,
            self._g_yaw, self._g_roll, self._g_pitch)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Detect both dictionaries
        april_c, april_ids, _ = self._april_det.detectMarkers(gray)
        aruco_c, aruco_ids, _ = self._aruco_det.detectMarkers(gray)

        # Merge
        detections = []
        if april_ids is not None:
            for c, i in zip(april_c, april_ids):
                mid = int(i[0])
                if mid == _OUTER_ID:
                    detections.append((c, mid))
        if aruco_ids is not None:
            for c, i in zip(aruco_c, aruco_ids):
                mid = int(i[0])
                if mid == _INNER_ID:
                    detections.append((c, mid))

        for corners, marker_id in detections:
            size_m = self._sizes[marker_id]
            half   = size_m / 2.0

            ok, rvec, tvec = cv2.solvePnP(
                self._obj_pts(half),
                corners[0].astype(np.float64),
                self._K, self._D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)

            if not ok:
                continue

            t = tvec.flatten()
            dist = float(np.linalg.norm(t))
            if dist < 0.01 or dist > 300.0:
                rospy.logwarn_throttle(
                    2.0, f'[aruco_pose_est] ID={marker_id} implausible dist={dist:.1f}m')
                continue

            # ── Pose in camera optical frame ──────────────────────────────────
            # position: AR-panel centre in camera optical frame
            #   tvec is the marker centre (= AR-panel visual centre) directly in
            #   camera optical frame — no further FK transformation needed.
            pos_in_cam = t.copy()

            # orientation: marker face rotation in camera optical frame.
            #   R from Rodrigues maps marker frame → camera frame.
            #   Marker frame convention (IPPE_SQUARE object points):
            #     x = right across marker face
            #     y = up   across marker face
            #     z = outward normal (toward camera)
            R_marker_cam_mat, _ = cv2.Rodrigues(rvec)
            R_marker_in_cam = Rot.from_matrix(R_marker_cam_mat)
            # Correct for the fixed 90° rotation between marker face frame and
            # USV body frame (URDF visual rpy="0 0 π/2").
            R_usv_in_cam = R_marker_in_cam * _R_MARKER_TO_USV
            q = R_usv_in_cam.as_quat()   # [x, y, z, w]

            # ── Publish ───────────────────────────────────────────────────────
            pose_msg = PoseStamped()
            pose_msg.header.stamp    = msg.header.stamp
            pose_msg.header.frame_id = 'overhead_cam_optical'
            pose_msg.pose.position   = Point(
                x=float(pos_in_cam[0]),
                y=float(pos_in_cam[1]),
                z=float(pos_in_cam[2]))
            pose_msg.pose.orientation = Quaternion(
                x=float(q[0]), y=float(q[1]),
                z=float(q[2]), w=float(q[3]))
            self._pub[marker_id].publish(pose_msg)

            if marker_id == _OUTER_ID:
                self._n_outer += 1
            else:
                self._n_inner += 1

            rospy.logdebug(
                f'[aruco_pose_est] ID={marker_id} '
                f'pos_cam=({pos_in_cam[0]:.3f},{pos_in_cam[1]:.3f},{pos_in_cam[2]:.3f})m '
                f'dist={dist:.2f}m')

        if self._n_frames % 30 == 0:
            rospy.loginfo(
                f'[aruco_pose_est] frames={self._n_frames} '
                f'outer={self._n_outer} inner={self._n_inner}')


if __name__ == '__main__':
    try:
        ArucoPoseEstimatorNode()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python3
"""
eval_aruco_driver_node.py
──────────────────────────
Offline evaluation driver for aruco_pose_estimator_node.

Replays test samples collected by collect_test_samples.launch and measures
the accuracy of the ArUco/AprilTag pose estimator for both the outer
(AprilTag 36h11 ID=10, 2.0 m) and inner (ArUco 4×4_50 ID=1, 0.2 m) markers.

No Gazebo required.  The driver publishes synthetic drone state and images,
then waits for aruco_pose_estimator_node to publish pose estimates.

For each sample the driver:
  1. Publishes synthetic Odometry, JointState, CameraInfo.
  2. Publishes the saved image on /<ns>/overhead_cam/image_raw.
  3. Waits for aruco_pose_estimator → /aruco_pose/{outer,inner}/usv_in_cam.
  4. Computes ground-truth AR-panel-centre in camera frame via FK + metadata.
  5. Reports per-marker errors.

Ground truth
────────────
  GT position (camera optical frame):
    marker_world = usv_pos + R_usv × _AR_CENTER_IN_USV
    gt_pos_in_cam = R_opt⁻¹ × (marker_world − p_opt)

  GT orientation (camera optical frame):
    R_usv_in_cam = R_opt⁻¹ × R_usv
    (Note: a fixed rotation offset exists between this "USV frame in camera"
     and the "marker face frame in camera" reported by solvePnP.  The offset
     is constant across all samples, so relative yaw tracking is meaningful.)

Metrics (both markers)
──────────────────────
  ex, ey, ez [m]       position error components in camera optical frame
  e_lat  [m]           lateral error magnitude √(ex²+ey²)
  e_3d   [m]           full 3-D Euclidean position error
  e_dist [m]           error in camera-to-marker range
  e_roll, e_pitch, e_yaw [°]   orientation errors (USV frame vs marker frame;
                                 includes the fixed marker→USV offset)

Parameters
──────────
  ~ns               UAV namespace           (default: uav1)
  ~test_samples_dir path to collected data  (default: <pkg>/test_samples)
  ~detect_timeout_s per-image timeout [s]   (default: 5.0)
"""

import os
import math
import time
import threading
import yaml
import numpy as np
import cv2
import rospy
import rospkg

from sensor_msgs.msg import Image, CameraInfo, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, PoseStamped
from scipy.spatial.transform import Rotation as Rot

# ── Gimbal FK (identical to aruco_pose_estimator_node) ────────────────────────
_HALF_PI   = math.pi / 2.0
_R_OPT     = Rot.from_euler('z', -_HALF_PI) * Rot.from_euler('x', -_HALF_PI)
_YAW_OFF   = np.array([0.0,   0.0,  -0.025])
_ROLL_OFF  = np.array([0.0,   0.0,  -0.030])
_PITCH_OFF = np.array([0.0,   0.0,  -0.025])
_OPT_OFF   = np.array([0.025, 0.0,   0.0  ])
_BASE_OFF  = np.array([0.10,  0.00,  0.00 ])

# AR-panel visual centre in ar_code link (= Otter USV root) frame.
# URDF: <visual origin xyz="-0.15 0 0.7"/>, mesh top face at 0.7+0.05 = 0.75 m.
_AR_CENTER_IN_USV = np.array([-0.15, 0.0, 0.75], dtype=np.float64)

# Camera constants (FLIR Hadron 640R EO)
IMG_W, IMG_H = 924, 690
HFOV_DEG     = 67.0


def _camera_fk(drone_pos, drone_rot, yaw, roll, pitch):
    p, R = drone_pos + drone_rot.apply(_BASE_OFF), drone_rot
    p, R = p + R.apply(_YAW_OFF),   R * Rot.from_euler('z', yaw)
    p, R = p + R.apply(_ROLL_OFF),  R * Rot.from_euler('x', roll)
    p, R = p + R.apply(_PITCH_OFF), R * Rot.from_euler('y', pitch)
    p, R = p + R.apply(_OPT_OFF),   R * _R_OPT
    return p, R


def _make_camera_info(ns, stamp):
    fx = (IMG_W / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)
    ci = CameraInfo()
    ci.header.stamp    = stamp
    ci.header.frame_id = f'{ns}/overhead_cam_optical'
    ci.width  = IMG_W;  ci.height = IMG_H
    ci.distortion_model = 'plumb_bob'
    ci.D = [0.0] * 5
    ci.K = [fx, 0.0, IMG_W/2.0,  0.0, fx, IMG_H/2.0,  0.0, 0.0, 1.0]
    ci.R = [1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0]
    ci.P = [fx, 0.0, IMG_W/2.0, 0.0,  0.0, fx, IMG_H/2.0, 0.0,  0.0, 0.0, 1.0, 0.0]
    return ci


_OUTER_ID = 10
_INNER_ID = 1
_MARKER_NAMES = {_OUTER_ID: 'outer', _INNER_ID: 'inner'}


class EvalArucoDriver:

    def __init__(self):
        rospy.init_node('eval_aruco_driver_node')

        pkg         = rospkg.RosPack().get_path('otter_usv_detector')
        self._ns    = rospy.get_param('~ns', 'uav1')
        samples_dir = rospy.get_param('~test_samples_dir', '')
        self._timeout = float(rospy.get_param('~detect_timeout_s', 5.0))

        if not samples_dir:
            samples_dir = os.path.join(pkg, 'test_samples')
        self._results_path = os.path.join(samples_dir, 'eval_aruco_results.yaml')

        meta_path = os.path.join(samples_dir, 'metadata.yaml')
        with open(meta_path) as f:
            self.samples = yaml.safe_load(f)['samples']
        self._img_dir = os.path.join(samples_dir, 'images')

        ns = self._ns
        self._pub_img  = rospy.Publisher(f'/{ns}/overhead_cam/image_raw',
                                          Image,      queue_size=1)
        self._pub_info = rospy.Publisher(f'/{ns}/overhead_cam/camera_info',
                                          CameraInfo, queue_size=1, latch=True)
        self._pub_odom = rospy.Publisher(f'/{ns}/ground_truth',
                                          Odometry,   queue_size=1)
        self._pub_js   = rospy.Publisher(f'/{ns}/gimbal/joint_states',
                                          JointState, queue_size=1)

        self._lock       = threading.Lock()
        self._img_stamp  = rospy.Time(0)
        self._last_pose  = {_OUTER_ID: None, _INNER_ID: None}

        rospy.Subscriber('/aruco_pose/outer/usv_in_cam', PoseStamped,
                         lambda m: self._pose_cb(m, _OUTER_ID))
        rospy.Subscriber('/aruco_pose/inner/usv_in_cam', PoseStamped,
                         lambda m: self._pose_cb(m, _INNER_ID))

        self._pub_info.publish(_make_camera_info(ns, rospy.Time.now()))

        rospy.loginfo(f'[eval_aruco] {len(self.samples)} samples | timeout={self._timeout:.0f}s')
        rospy.loginfo('[eval_aruco] Waiting for aruco_pose_estimator to subscribe …')
        t0 = time.monotonic()
        while self._pub_img.get_num_connections() == 0 and not rospy.is_shutdown():
            if time.monotonic() - t0 > 60.0:
                rospy.logfatal('[eval_aruco] Estimator did not subscribe within 60 s')
                raise RuntimeError('Estimator timeout')
            rospy.sleep(0.5)
        rospy.loginfo('[eval_aruco] Estimator ready — starting evaluation.')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _pose_cb(self, msg: PoseStamped, marker_id: int):
        with self._lock:
            if msg.header.stamp >= self._img_stamp:
                self._last_pose[marker_id] = msg

    # ── Publishers ────────────────────────────────────────────────────────────

    def _publish_state(self, meta):
        ns    = self._ns
        stamp = rospy.Time.now()

        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'world'
        odom.child_frame_id  = f'{ns}/base_link'
        pos = meta['drone_pos'];  q = meta['drone_quat']
        odom.pose.pose.position.x    = pos[0]
        odom.pose.pose.position.y    = pos[1]
        odom.pose.pose.position.z    = pos[2]
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        self._pub_odom.publish(odom)

        js = JointState()
        js.header.stamp = stamp
        js.name     = [f'{ns}_gimbal_yaw_joint',
                       f'{ns}_gimbal_roll_joint',
                       f'{ns}_gimbal_pitch_joint']
        js.position = [meta['gimbal_yaw_rad'],
                       meta['gimbal_roll_rad'],
                       meta['gimbal_pitch_rad']]
        self._pub_js.publish(js)

    def _publish_image(self, img_path, stamp):
        bgr = cv2.imread(img_path)
        if bgr is None:
            return False
        msg = Image()
        msg.header.stamp    = stamp
        msg.header.frame_id = f'{self._ns}/overhead_cam_optical'
        msg.height   = bgr.shape[0]
        msg.width    = bgr.shape[1]
        msg.encoding = 'bgr8'
        msg.step     = bgr.shape[1] * 3
        msg.data     = bgr.tobytes()
        self._pub_img.publish(msg)
        return True

    def _wait_for_poses(self):
        """Wait up to timeout for any new pose from either marker."""
        t0 = time.monotonic()
        while not rospy.is_shutdown():
            with self._lock:
                if any(v is not None for v in self._last_pose.values()):
                    return dict(self._last_pose)
            if time.monotonic() - t0 > self._timeout:
                return {_OUTER_ID: None, _INNER_ID: None}
            rospy.sleep(0.02)
        return {_OUTER_ID: None, _INNER_ID: None}

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        # results[marker_id] = list of row dicts
        results = {_OUTER_ID: [], _INNER_ID: []}

        for meta in self.samples:
            sid   = meta['sample_id']
            label = meta['label']

            # ── Ground truth ──────────────────────────────────────────────────
            drone_pos = np.array(meta['drone_pos'])
            drone_rot = Rot.from_quat(meta['drone_quat'])
            p_opt, R_opt = _camera_fk(
                drone_pos, drone_rot,
                meta['gimbal_yaw_rad'],
                meta['gimbal_roll_rad'],
                meta['gimbal_pitch_rad'])

            usv_pos_w  = np.array(meta['usv_pos'])
            usv_rot    = Rot.from_quat(meta['usv_quat'])

            # AR-panel centre in world frame
            marker_world = usv_pos_w + usv_rot.apply(_AR_CENTER_IN_USV)

            # GT position: AR-panel centre in camera optical frame
            gt_pos_in_cam = R_opt.inv().apply(marker_world - p_opt)
            gt_dist       = float(np.linalg.norm(gt_pos_in_cam))

            # GT orientation: USV (ar_code) frame expressed in camera optical frame.
            # Note: solvePnP outputs marker face frame, which differs from ar_code
            # frame by a fixed rotation from the URDF visual rpy + texture layout.
            # This fixed offset is the same for every sample, so e_roll/e_pitch/e_yaw
            # contain that constant bias; what varies across samples is the actual
            # tracking error.
            R_usv_in_cam  = R_opt.inv() * usv_rot
            gt_rpy_deg    = R_usv_in_cam.as_euler('xyz', degrees=True)  # roll,pitch,yaw

            img_path = os.path.join(self._img_dir, meta['image_file'])

            # ── Clear stale data ──────────────────────────────────────────────
            stamp = rospy.Time.now()
            with self._lock:
                self._last_pose  = {_OUTER_ID: None, _INNER_ID: None}
                self._img_stamp  = stamp

            self._publish_state(meta)
            rospy.sleep(0.2)

            if not self._publish_image(img_path, stamp):
                rospy.logwarn(f'[eval_aruco] #{sid:02d} "{label}": image not found')
                continue

            poses = self._wait_for_poses()
            rospy.sleep(0.3)

            # ── Build result row for each marker ──────────────────────────────
            nan = float('nan')

            for mid in (_OUTER_ID, _INNER_ID):
                mname = _MARKER_NAMES[mid]
                row = dict(
                    sample_id=sid, label=label,
                    marker=mname,
                    gt_x=float(gt_pos_in_cam[0]),
                    gt_y=float(gt_pos_in_cam[1]),
                    gt_z=float(gt_pos_in_cam[2]),
                    gt_dist=gt_dist,
                    gt_roll=float(gt_rpy_deg[0]),
                    gt_pitch=float(gt_rpy_deg[1]),
                    gt_yaw=float(gt_rpy_deg[2]),
                    detected=False,
                    est_x=nan, est_y=nan, est_z=nan,
                    est_dist=nan,
                    est_roll=nan, est_pitch=nan, est_yaw=nan,
                    err_x=nan, err_y=nan, err_z=nan,
                    err_lat=nan, err_3d=nan, err_dist=nan,
                    err_roll=nan, err_pitch=nan, err_yaw=nan,
                )

                pose_msg = poses[mid]
                if pose_msg is None:
                    results[mid].append(row)
                    continue

                row['detected'] = True
                p = pose_msg.pose.position
                o = pose_msg.pose.orientation

                est_pos  = np.array([p.x, p.y, p.z])
                est_dist = float(np.linalg.norm(est_pos))
                R_est    = Rot.from_quat([o.x, o.y, o.z, o.w])
                est_rpy  = R_est.as_euler('xyz', degrees=True)

                err     = est_pos - gt_pos_in_cam
                err_lat = float(np.linalg.norm(err[:2]))
                err_3d  = float(np.linalg.norm(err))

                # Orientation error (includes fixed marker→USV frame offset)
                R_err      = R_est * R_usv_in_cam.inv()
                err_rpy    = R_err.as_euler('xyz', degrees=True)

                row.update(
                    est_x=float(p.x), est_y=float(p.y), est_z=float(p.z),
                    est_dist=est_dist,
                    est_roll=float(est_rpy[0]),
                    est_pitch=float(est_rpy[1]),
                    est_yaw=float(est_rpy[2]),
                    err_x=float(err[0]),   err_y=float(err[1]),  err_z=float(err[2]),
                    err_lat=err_lat,        err_3d=err_3d,
                    err_dist=float(abs(est_dist - gt_dist)),
                    err_roll=float(err_rpy[0]),
                    err_pitch=float(err_rpy[1]),
                    err_yaw=float(err_rpy[2]),
                )

                tag = f'[eval_aruco|{mname.upper()}]'
                rospy.loginfo(
                    f'{tag} #{sid:02d} "{label}"  '
                    f'pos_err({err[0]:+.3f},{err[1]:+.3f},{err[2]:+.3f})m '
                    f'lat={err_lat:.3f}m 3D={err_3d:.3f}m dist_err={abs(est_dist-gt_dist):.3f}m  '
                    f'rpy_err=({err_rpy[0]:+.1f},{err_rpy[1]:+.1f},{err_rpy[2]:+.1f})°')

                results[mid].append(row)

            rospy.sleep(0.3)

        self._print_summary(results)
        self._save_results(results)
        rospy.signal_shutdown('Evaluation complete.')

    # ── Reporting ─────────────────────────────────────────────────────────────

    def _print_summary(self, results):
        W = 120
        for mid, mname in _MARKER_NAMES.items():
            rows  = results[mid]
            ok    = [r for r in rows if r['detected']]
            total = len(rows)

            rospy.loginfo('=' * W)
            rospy.loginfo(
                f' ARUCO POSE EVALUATION — {mname.upper()} '
                f'(ID={mid})   {len(ok)}/{total} detected')
            rospy.loginfo('-' * W)
            rospy.loginfo(
                f'  {"#":>3}  {"label":<12}  '
                f'{"ex[m]":>8} {"ey[m]":>8} {"ez[m]":>8} '
                f'{"elat[m]":>8} {"e3D[m]":>8} {"edist[m]":>8}  '
                f'{"eroll°":>7} {"epitch°":>7} {"eyaw°":>7}')
            rospy.loginfo('-' * W)

            def _f(v):
                return f'{v:+8.3f}' if not math.isnan(v) else '     ---'
            def _fd(v):
                return f'{v:+7.1f}' if not math.isnan(v) else '    ---'

            for r in rows:
                rospy.loginfo(
                    f'  {r["sample_id"]:>3}  {r["label"]:<12}  '
                    f'{_f(r["err_x"])} {_f(r["err_y"])} {_f(r["err_z"])} '
                    f'{_f(r["err_lat"])} {_f(r["err_3d"])} {_f(r["err_dist"])}  '
                    f'{_fd(r["err_roll"])} {_fd(r["err_pitch"])} {_fd(r["err_yaw"])}')

            rospy.loginfo('-' * W)
            metrics = [
                ('x_cam [m]',   'err_x'),
                ('y_cam [m]',   'err_y'),
                ('z_cam [m]',   'err_z'),
                ('lat [m]',     'err_lat'),
                ('3D [m]',      'err_3d'),
                ('dist [m]',    'err_dist'),
                ('roll [°]',    'err_roll'),
                ('pitch [°]',   'err_pitch'),
                ('yaw [°]',     'err_yaw'),
            ]
            for lab, key in metrics:
                vals = [abs(r[key]) for r in ok
                        if not math.isnan(r.get(key, float('nan')))]
                if vals:
                    rospy.loginfo(
                        f'  {lab:<12}: mean={np.mean(vals):.4f}  '
                        f'std={np.std(vals):.4f}  '
                        f'max={np.max(vals):.4f}  '
                        f'median={np.median(vals):.4f}')
                else:
                    rospy.loginfo(f'  {lab:<12}: --- (no detections)')
            rospy.loginfo('=' * W)

    def _save_results(self, results):
        out = {}
        for mid, mname in _MARKER_NAMES.items():
            rows = results[mid]
            ok   = [r for r in rows if r['detected']]

            summary = {}
            for key in ['err_x', 'err_y', 'err_z', 'err_lat', 'err_3d',
                        'err_dist', 'err_roll', 'err_pitch', 'err_yaw']:
                vals = [abs(r[key]) for r in ok
                        if not math.isnan(r.get(key, float('nan')))]
                if vals:
                    summary[key] = dict(
                        mean=float(np.mean(vals)),
                        std=float(np.std(vals)),
                        max=float(np.max(vals)),
                        median=float(np.median(vals)))

            out[mname] = dict(
                marker_id=mid,
                n_total=len(rows),
                n_detected=len(ok),
                summary=summary,
                results=[dict(r) for r in rows],
            )

        with open(self._results_path, 'w') as f:
            yaml.dump(out, f, default_flow_style=False)
        rospy.loginfo(f'[eval_aruco] Results saved → {self._results_path}')


if __name__ == '__main__':
    try:
        EvalArucoDriver().run()
    except rospy.ROSInterruptException:
        pass

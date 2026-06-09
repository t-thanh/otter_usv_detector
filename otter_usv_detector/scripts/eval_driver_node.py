#!/usr/bin/env python3
"""
eval_driver_node.py
────────────────────
Offline evaluation driver for yolo_pose_estimator_node.

Drives the YOLO OBB pose estimation pipeline offline using samples collected
by collect_test_samples.launch.  No Gazebo is required.

For each sample the driver:
  1. Publishes synthetic Odometry, JointState, ModelStates, CameraInfo.
  2. Publishes the saved image on /<ns>/overhead_cam/image_raw.
  3. Waits for detector_node → /usv_detection/result.
  4. Waits for yolo_pose_estimator → /yolo_pose/usv_in_cam (PoseStamped).
  5. Computes ground-truth USV-in-camera position via camera FK from metadata.
  6. Reports errors:
       ex, ey [m]  — lateral (image-plane) position error components
       ez [m]      — range (depth) error
       e_lat [m]   — lateral error magnitude sqrt(ex²+ey²)
       e_3d [m]    — full 3-D Euclidean position error
       e_dist [m]  — error in camera-to-USV range magnitude
       e_yaw [°]   — USV yaw error (world frame); NaN for centroid-fallback poses

  Position GT: R_opt_gt⁻¹ × (usv_pos_world − p_opt_gt)   [camera optical frame]
  Yaw GT: USV yaw in world frame from metadata usv_quat.

Parameters
----------
  ~ns               UAV namespace           (default: uav1)
  ~test_samples_dir path to collected data  (default: <pkg>/test_samples)
  ~detect_timeout_s per-image timeout [s]   (default: 20.0)
  ~min_confidence   OBB confidence gate     (default: 0.35)
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
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, PoseStamped
from scipy.spatial.transform import Rotation as Rot
from otter_usv_detector.msg import ObbDetectionArray

# ── Gimbal FK — must match yolo_pose_estimator_node.py exactly ───────────────
_HALF_PI   = math.pi / 2.0
_R_OPT     = Rot.from_euler('z', -_HALF_PI) * Rot.from_euler('x', -_HALF_PI)
_YAW_OFF   = np.array([0.0,   0.0,  -0.025])
_ROLL_OFF  = np.array([0.0,   0.0,  -0.030])
_PITCH_OFF = np.array([0.0,   0.0,  -0.025])
_OPT_OFF   = np.array([0.025, 0.0,   0.0  ])
_BASE_OFF  = np.array([0.10,  0.00,  0.00 ])   # base_x/y/z_offset params

# AR-panel centre in USV base_link frame (matches estimator)
_PANEL_CENTER_USV = np.array([-0.15, 0.0, 0.70])


def _camera_fk(drone_pos, drone_rot, yaw, roll, pitch):
    """Gimbal FK → (p_opt, R_opt).  Identical to yolo_pose_estimator_node."""
    p, R = drone_pos + drone_rot.apply(_BASE_OFF), drone_rot
    p, R = p + R.apply(_YAW_OFF),   R * Rot.from_euler('z', yaw)
    p, R = p + R.apply(_ROLL_OFF),  R * Rot.from_euler('x', roll)
    p, R = p + R.apply(_PITCH_OFF), R * Rot.from_euler('y', pitch)
    p, R = p + R.apply(_OPT_OFF),   R * _R_OPT
    return p, R


# ── Camera constants (Hadron 640R EO) ─────────────────────────────────────────
IMG_W, IMG_H = 924, 690
HFOV_DEG     = 67.0


def _build_fx():
    return (IMG_W / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)


def _make_camera_info(ns: str, stamp) -> CameraInfo:
    fx = _build_fx()
    ci = CameraInfo()
    ci.header.stamp      = stamp
    ci.header.frame_id   = f'{ns}/overhead_cam_optical'
    ci.width             = IMG_W
    ci.height            = IMG_H
    ci.distortion_model  = 'plumb_bob'
    ci.D = [0.0] * 5
    ci.K = [fx, 0.0, IMG_W/2.0, 0.0, fx, IMG_H/2.0, 0.0, 0.0, 1.0]
    ci.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ci.P = [fx, 0.0, IMG_W/2.0, 0.0, 0.0, fx, IMG_H/2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    return ci


def _rpy_deg(qx, qy, qz, qw):
    """Extract (roll, pitch, yaw) in degrees from a quaternion."""
    return Rot.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=True)


class EvalDriver:

    def __init__(self):
        rospy.init_node('eval_driver_node')

        pkg         = rospkg.RosPack().get_path('otter_usv_detector')
        self._ns    = rospy.get_param('~ns',               'uav1')
        samples_dir = rospy.get_param('~test_samples_dir', '')
        self._timeout  = float(rospy.get_param('~detect_timeout_s', 20.0))
        self._min_conf = float(rospy.get_param('~min_confidence',   0.35))

        if not samples_dir:
            samples_dir = os.path.join(pkg, 'test_samples')
        self._results_path = os.path.join(samples_dir, 'eval_results.yaml')

        meta_path = os.path.join(samples_dir, 'metadata.yaml')
        with open(meta_path) as f:
            self.samples = yaml.safe_load(f)['samples']
        self._img_dir = os.path.join(samples_dir, 'images')

        ns = self._ns
        self._pub_img    = rospy.Publisher(f'/{ns}/overhead_cam/image_raw',
                                            Image,        queue_size=1)
        self._pub_info   = rospy.Publisher(f'/{ns}/overhead_cam/camera_info',
                                            CameraInfo,   queue_size=1, latch=True)
        self._pub_odom   = rospy.Publisher(f'/{ns}/ground_truth',
                                            Odometry,     queue_size=1)
        self._pub_js     = rospy.Publisher(f'/{ns}/gimbal/joint_states',
                                            JointState,   queue_size=1)
        self._pub_models = rospy.Publisher('/gazebo/model_states',
                                            ModelStates,  queue_size=1)

        self._lock       = threading.Lock()
        self._last_pose  : PoseStamped       = None
        self._last_det   : ObbDetectionArray = None
        self._img_stamp  = rospy.Time(0)

        rospy.Subscriber('/yolo_pose/usv_in_cam',    PoseStamped,       self._pose_cb)
        rospy.Subscriber('/usv_detection/result', ObbDetectionArray, self._det_cb)

        # Latched camera_info — yolo_pose_estimator receives it on subscribe
        self._pub_info.publish(_make_camera_info(ns, rospy.Time.now()))

        rospy.loginfo(f'[eval_driver] {len(self.samples)} samples | '
                      f'timeout={self._timeout:.0f}s')
        rospy.loginfo('[eval_driver] Waiting for detector to subscribe (model loading …)')
        t0 = time.monotonic()
        while self._pub_img.get_num_connections() == 0 and not rospy.is_shutdown():
            if time.monotonic() - t0 > 180.0:
                rospy.logfatal('[eval_driver] Detector did not subscribe within 180 s')
                raise RuntimeError('Detector timeout')
            rospy.sleep(0.5)
        rospy.loginfo('[eval_driver] Detector ready — allowing 3 s for estimator init …')
        rospy.sleep(3.0)
        rospy.loginfo('[eval_driver] Starting evaluation.')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _pose_cb(self, msg: PoseStamped):
        with self._lock:
            # Accept only poses derived from the image we just published
            if msg.header.stamp >= self._img_stamp:
                self._last_pose = msg

    def _det_cb(self, msg: ObbDetectionArray):
        with self._lock:
            self._last_det = msg

    # ── Publishers ────────────────────────────────────────────────────────────

    def _publish_state(self, meta: dict):
        """Publish synthetic drone odometry, gimbal joint states, USV model state."""
        ns    = self._ns
        stamp = rospy.Time.now()

        # Odometry (drone world pose)
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'world'
        odom.child_frame_id  = f'{ns}/base_link'
        pos = meta['drone_pos']
        q   = meta['drone_quat']   # [x, y, z, w]
        odom.pose.pose.position.x    = pos[0]
        odom.pose.pose.position.y    = pos[1]
        odom.pose.pose.position.z    = pos[2]
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        self._pub_odom.publish(odom)

        # Gimbal joint states
        js = JointState()
        js.header.stamp = stamp
        js.name     = [f'{ns}_gimbal_yaw_joint',
                       f'{ns}_gimbal_roll_joint',
                       f'{ns}_gimbal_pitch_joint']
        js.position = [meta['gimbal_yaw_rad'],
                       meta['gimbal_roll_rad'],
                       meta['gimbal_pitch_rad']]
        self._pub_js.publish(js)

        # USV model state (Gazebo ModelStates format)
        usv_p = meta['usv_pos']
        usv_q = meta['usv_quat']   # [x, y, z, w]
        models = ModelStates()
        models.name = ['otter']
        p = Pose()
        p.position.x    = usv_p[0];  p.position.y    = usv_p[1];  p.position.z    = usv_p[2]
        p.orientation.x = usv_q[0];  p.orientation.y = usv_q[1]
        p.orientation.z = usv_q[2];  p.orientation.w = usv_q[3]
        models.pose  = [p]
        models.twist = [Twist()]
        self._pub_models.publish(models)

    def _publish_image(self, img_path: str, stamp) -> bool:
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

    def _wait_for_pose(self) -> 'PoseStamped | None':
        t0 = time.monotonic()
        while not rospy.is_shutdown():
            with self._lock:
                if self._last_pose is not None:
                    return self._last_pose
            if time.monotonic() - t0 > self._timeout:
                return None
            rospy.sleep(0.05)
        return None

    # ── Main evaluation loop ──────────────────────────────────────────────────

    def run(self):
        results = []

        for meta in self.samples:
            sid   = meta['sample_id']
            label = meta['label']

            # ── Ground truth in camera frame via FK ───────────────────────────
            drone_pos  = np.array(meta['drone_pos'])
            drone_rot  = Rot.from_quat(meta['drone_quat'])
            p_opt_gt, R_opt_gt = _camera_fk(
                drone_pos, drone_rot,
                meta['gimbal_yaw_rad'],
                meta['gimbal_roll_rad'],
                meta['gimbal_pitch_rad'])

            usv_pos_w  = np.array(meta['usv_pos'])   # base_link world (z ≈ 0)
            usv_rot_gt = Rot.from_quat(meta['usv_quat'])

            # GT: USV base_link in camera optical frame
            gt_usv_in_cam = R_opt_gt.inv().apply(usv_pos_w - p_opt_gt)
            gt_dist       = float(np.linalg.norm(gt_usv_in_cam))
            gt_yaw_deg    = float(np.degrees(usv_rot_gt.as_euler('xyz')[2]))

            img_path = os.path.join(self._img_dir, meta['image_file'])

            # ── Clear stale data & set timestamp gate ─────────────────────────
            stamp = rospy.Time.now()
            with self._lock:
                self._last_pose  = None
                self._last_det   = None
                self._img_stamp  = stamp

            # ── Publish state then image ──────────────────────────────────────
            self._publish_state(meta)
            rospy.sleep(0.3)

            if not self._publish_image(img_path, stamp):
                rospy.logwarn(f'[eval_driver] #{sid:02d} "{label}": image not found')
                continue

            # ── Wait for pose estimate ────────────────────────────────────────
            pose_msg = self._wait_for_pose()

            # Build result row
            nan = float('nan')
            row = dict(
                sample_id=sid, label=label,
                gt_x=float(gt_usv_in_cam[0]),
                gt_y=float(gt_usv_in_cam[1]),
                gt_z=float(gt_usv_in_cam[2]),
                gt_dist=gt_dist, gt_yaw=gt_yaw_deg,
                detected=False, pose_ok=False, centroid_only=False,
                confidence=nan,
                est_x=nan, est_y=nan, est_z=nan,
                est_dist=nan, est_yaw=nan,
                err_x=nan, err_y=nan, err_z=nan,
                err_lat=nan, err_3d=nan, err_dist=nan, err_yaw=nan,
            )

            # Was anything detected?
            with self._lock:
                det = self._last_det
            if det is not None and det.detections:
                best = max(
                    (d for d in det.detections if d.confidence >= self._min_conf),
                    key=lambda d: d.confidence, default=None)
                if best:
                    row['detected']   = True
                    row['confidence'] = float(best.confidence)

            if pose_msg is None:
                status = 'no pose published' if row['detected'] else 'not detected'
                rospy.logwarn(f'[eval_driver] #{sid:02d} "{label}": {status}')
            else:
                row['pose_ok'] = True
                p = pose_msg.pose.position
                o = pose_msg.pose.orientation

                est_pos  = np.array([p.x, p.y, p.z])
                est_dist = float(np.linalg.norm(est_pos))

                # Detect centroid-fallback mode: orientation ≈ identity
                R_est = Rot.from_quat([o.x, o.y, o.z, o.w])
                q_id  = Rot.identity().as_quat()
                q_est = R_est.as_quat()
                is_centroid = float(np.linalg.norm(q_est - q_id)) < 1e-4

                row['centroid_only'] = is_centroid

                # Estimated yaw: extract from orientation unless centroid mode
                if not is_centroid:
                    # USV yaw in world frame: R_usv = R_opt · R_usv_in_cam
                    R_usv_w = R_opt_gt * R_est
                    est_yaw_deg = float(np.degrees(R_usv_w.as_euler('xyz')[2]))
                    row['est_yaw'] = est_yaw_deg
                else:
                    est_yaw_deg = nan
                    row['est_yaw'] = nan

                row.update(est_x=float(p.x), est_y=float(p.y), est_z=float(p.z),
                           est_dist=est_dist)

                err     = est_pos - gt_usv_in_cam
                err_lat = float(np.linalg.norm(err[:2]))
                err_3d  = float(np.linalg.norm(err))

                err_yaw = nan
                if not is_centroid and not math.isnan(est_yaw_deg):
                    d = (est_yaw_deg - gt_yaw_deg + 180.0) % 360.0 - 180.0
                    err_yaw = float(d)

                row.update(
                    err_x=float(err[0]),   err_y=float(err[1]),  err_z=float(err[2]),
                    err_lat=err_lat,        err_3d=err_3d,
                    err_dist=float(abs(est_dist - gt_dist)),
                    err_yaw=err_yaw,
                )

                mode = 'CENTROID' if is_centroid else 'FULL'
                yaw_s = f'yaw_err={err_yaw:+.1f}°' if not math.isnan(err_yaw) else 'yaw=N/A'
                rospy.loginfo(
                    f'[eval_driver|{mode}] #{sid:02d} "{label}"  '
                    f'pos({err[0]:+.3f},{err[1]:+.3f},{err[2]:+.3f})m '
                    f'lat={err_lat:.3f}m 3D={err_3d:.3f}m dist_err={abs(est_dist-gt_dist):.3f}m  '
                    f'{yaw_s}  conf={row["confidence"]:.2f}')

            results.append(row)
            rospy.sleep(0.5)

        self._print_summary(results)
        self._save_results(results)
        rospy.signal_shutdown('Evaluation complete.')

    # ── Reporting ─────────────────────────────────────────────────────────────

    def _print_summary(self, results):
        ok       = [r for r in results if r['pose_ok']]
        ok_full  = [r for r in ok if not r.get('centroid_only', False)]
        ok_cen   = [r for r in ok if r.get('centroid_only', False)]
        det      = [r for r in results if r['detected']]
        W        = 112

        def _fmt(v):
            return f'{v:+8.3f}' if not math.isnan(v) else '     ---'

        rospy.loginfo('=' * W)
        rospy.loginfo(
            f' YOLO OBB POSE EVALUATION  '
            f'{len(ok)}/{len(results)} pose OK  '
            f'({len(ok_full)} full + {len(ok_cen)} centroid)  '
            f'{len(det)}/{len(results)} detected')
        rospy.loginfo('-' * W)
        rospy.loginfo(
            f'  {"#":>3}  {"label":<12}  {"mode":<8}  '
            f'{"ex[m]":>8} {"ey[m]":>8} {"ez[m]":>8} '
            f'{"elat[m]":>8} {"e3D[m]":>8} {"edist[m]":>8} '
            f'{"eyaw°":>7}  conf')
        rospy.loginfo('-' * W)

        for r in results:
            conf_s = f'{r["confidence"]:5.2f}' if not math.isnan(r['confidence']) else '  ---'
            mode   = 'CENTROID' if r.get('centroid_only') else ('FULL' if r['pose_ok'] else '---')
            rospy.loginfo(
                f'  {r["sample_id"]:>3}  {r["label"]:<12}  {mode:<8}  '
                f'{_fmt(r["err_x"])} {_fmt(r["err_y"])} {_fmt(r["err_z"])} '
                f'{_fmt(r["err_lat"])} {_fmt(r["err_3d"])} {_fmt(r["err_dist"])} '
                f'{_fmt(r["err_yaw"])}  {conf_s}')

        rospy.loginfo('-' * W)
        metrics = [
            ('x_cam [m]',  'err_x'),
            ('y_cam [m]',  'err_y'),
            ('z_cam [m]',  'err_z'),
            ('lat [m]',    'err_lat'),
            ('3D [m]',     'err_3d'),
            ('dist [m]',   'err_dist'),
            ('yaw [°]',    'err_yaw'),
        ]
        for lab, key in metrics:
            vals = [abs(r[key]) for r in ok if not math.isnan(r.get(key, float('nan')))]
            if vals:
                rospy.loginfo(
                    f'  {lab:<12}: mean={np.mean(vals):.4f}  '
                    f'std={np.std(vals):.4f}  '
                    f'max={np.max(vals):.4f}  '
                    f'median={np.median(vals):.4f}')
            else:
                rospy.loginfo(f'  {lab:<12}: ---')
        rospy.loginfo('=' * W)

    def _save_results(self, results):
        ok  = [r for r in results if r['pose_ok']]
        det = [r for r in results if r['detected']]

        summary = {}
        for key in ['err_x', 'err_y', 'err_z', 'err_lat', 'err_3d',
                    'err_dist', 'err_yaw']:
            vals = [abs(r[key]) for r in ok
                    if not math.isnan(r.get(key, float('nan')))]
            if vals:
                summary[key] = dict(
                    mean=float(np.mean(vals)),
                    std=float(np.std(vals)),
                    max=float(np.max(vals)),
                    median=float(np.median(vals)),
                )

        ok_full = [r for r in ok if not r.get('centroid_only', False)]
        ok_cen  = [r for r in ok if r.get('centroid_only', False)]

        with open(self._results_path, 'w') as f:
            yaml.dump(dict(
                n_total=len(results),
                n_detected=len(det),
                n_pose_ok=len(ok),
                n_full_pose=len(ok_full),
                n_centroid_fallback=len(ok_cen),
                summary=summary,
                results=[dict(r) for r in results],
            ), f, default_flow_style=False)
        rospy.loginfo(f'[eval_driver] Results saved → {self._results_path}')


if __name__ == '__main__':
    try:
        EvalDriver().run()
    except rospy.ROSInterruptException:
        pass

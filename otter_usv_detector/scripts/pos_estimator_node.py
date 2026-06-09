#!/usr/bin/env python3
"""
pos_estimator_node.py — USV 3-D position estimator from nadir OBB detection.

Estimates the position of the Otter USV relative to the Hadron 640R EO camera
on the X500 UAV by ray-casting through the pinhole model onto the water-surface
plane (z = 0 in world ENU).

Algorithm
---------
1. For each OBB detection, back-project the OBB centre pixel through the pinhole
   to get a unit ray in camera optical frame.
2. Rotate the ray to world ENU using:
      ray_world = R_body2world  @  R_cam2body  @  ray_cam
3. Intersect the ray with the plane z = water_z (default 0.0 = water surface).
4. Report USV position relative to the camera in ENU (Z-up) convention.

As a sanity check the node also estimates altitude from the apparent OBB size
(Otter USV length ≈ 2.0 m) and compares it to odometry altitude.

Camera-to-body rotation (R_cam2body)
-------------------------------------
Gazebo camera convention: sensor +X = look direction, +Y = image-right, +Z = image-down.
With SDF pitch=+π/2 (passive convention), the nadir camera has:
  • image top    → UAV forward  (+body X)
  • image right  → UAV left    (+body Y)   ← verify in rqt_image_view; flip with ~cam_flip_u
  • into scene   → UAV down    (-body Z)

If image appears mirrored/transposed set:
  ~cam_flip_u: true  (negate u axis → image right becomes UAV right)
  ~cam_flip_v: true  (negate v axis → image top becomes UAV backward)

Published topics
----------------
  /usv_position/estimate  (geometry_msgs/PoseStamped)
      position: USV relative to camera, expressed in world ENU (Z-up)
      frame_id: world

  /usv_position/debug  (geometry_msgs/PointStamped)
      x: altitude estimate from OBB size  [m]
      y: altitude from UAV odometry       [m]
      z: OBB longer side in pixels

Parameters
----------
  ~detection_topic   /usv_detection/result
  ~odom_topic        /uav1/ground_truth
  ~output_topic      /usv_position/estimate
  ~debug_topic       /usv_position/debug
  ~camera_hfov_deg   67.0
  ~image_width       924
  ~image_height      690
  ~cam_body_x        0.10    # camera x-offset from FCU in body frame [m]
  ~cam_body_y        0.00
  ~cam_body_z       -0.046   # cylinder tip (optical centre) below base_link [m]
  ~usv_length        2.0     # Otter USV length for altitude cross-check [m]
  ~water_z           0.0     # water surface altitude in world frame [m]
  ~min_confidence    0.30
  ~cam_flip_u        false   # negate image u axis in body transform
  ~cam_flip_v        false   # negate image v axis in body transform
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Odometry
from otter_usv_detector.msg import ObbDetectionArray


def _quat_to_rot(q):
    """geometry_msgs/Quaternion → 3×3 rotation matrix (world ← body)."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


class PositionEstimator:
    def __init__(self):
        rospy.init_node('pos_estimator_node')

        # ── camera intrinsics ─────────────────────────────────────────────────
        hfov_deg = rospy.get_param('~camera_hfov_deg', 67.0)
        img_w    = rospy.get_param('~image_width',  924)
        img_h    = rospy.get_param('~image_height', 690)
        self.fx  = (img_w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        self.fy  = self.fx            # square pixels assumed
        self.cx  = img_w / 2.0
        self.cy  = img_h / 2.0

        # ── camera position offset from FCU (body frame) ──────────────────────
        self.cam_offset = np.array([
            rospy.get_param('~cam_body_x',  0.10),
            rospy.get_param('~cam_body_y',  0.00),
            rospy.get_param('~cam_body_z', -0.046),
        ])

        # ── camera-to-body rotation ───────────────────────────────────────────
        # Nadir Gazebo camera (pitch=+π/2 mount): image top=fwd, image right=left.
        # Columns: how cam axes [right, down, into-scene] map to body axes [fwd, left, up].
        flip_u = rospy.get_param('~cam_flip_u', False)
        flip_v = rospy.get_param('~cam_flip_v', False)
        su = -1.0 if flip_u else 1.0
        sv = -1.0 if flip_v else 1.0
        # body_x = -sv * cam_y  (image top  = +body_x)
        # body_y =  su * cam_x  (image right= +body_y, i.e. UAV left)
        # body_z = -cam_z       (into scene = -body_z, i.e. downward)
        self.R_cam2body = np.array([
            [ 0.0, -sv,   0.0],
            [ su,   0.0,  0.0],
            [ 0.0,  0.0, -1.0],
        ])

        # ── misc parameters ───────────────────────────────────────────────────
        self.usv_length = rospy.get_param('~usv_length',     2.0)
        self.water_z    = rospy.get_param('~water_z',        0.0)
        self.min_conf   = rospy.get_param('~min_confidence', 0.30)

        # ── state ─────────────────────────────────────────────────────────────
        self.odom = None

        # ── subscribers ───────────────────────────────────────────────────────
        odom_topic = rospy.get_param('~odom_topic',      '/uav1/ground_truth')
        det_topic  = rospy.get_param('~detection_topic', '/usv_detection/result')
        rospy.Subscriber(odom_topic, Odometry,           self._cb_odom, queue_size=1)
        rospy.Subscriber(det_topic,  ObbDetectionArray,  self._cb_det,  queue_size=1)

        # ── publishers ────────────────────────────────────────────────────────
        out_topic   = rospy.get_param('~output_topic', '/usv_position/estimate')
        debug_topic = rospy.get_param('~debug_topic',  '/usv_position/debug')
        self.pub_pose  = rospy.Publisher(out_topic,   PoseStamped,  queue_size=5)
        self.pub_debug = rospy.Publisher(debug_topic, PointStamped, queue_size=5)

        rospy.loginfo(
            f'[pos_estimator] fx={self.fx:.1f}  cx={self.cx}  cy={self.cy} | '
            f'cam_offset={self.cam_offset}'
        )
        rospy.spin()

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _cb_odom(self, msg):
        self.odom = msg

    def _cb_det(self, msg):
        if self.odom is None:
            rospy.logwarn_throttle(5.0, '[pos_estimator] Waiting for odometry …')
            return
        if not msg.detections:
            return

        # highest-confidence detection that passes threshold
        best = max(
            (d for d in msg.detections if d.confidence >= self.min_conf),
            key=lambda d: d.confidence,
            default=None,
        )
        if best is None:
            return

        self._estimate(msg.header, best)

    # ── core estimation ───────────────────────────────────────────────────────

    def _estimate(self, header, det):
        odom = self.odom

        # UAV position and attitude in world (ENU)
        p = odom.pose.pose.position
        uav_world    = np.array([p.x, p.y, p.z])
        R_body2world = _quat_to_rot(odom.pose.pose.orientation)

        # Camera world position (apply body-frame offset)
        cam_world = uav_world + R_body2world @ self.cam_offset

        # Back-project OBB centre to a ray in camera optical frame
        ray_cam = np.array([
            (det.cx - self.cx) / self.fx,
            (det.cy - self.cy) / self.fy,
            1.0,
        ])
        ray_cam /= np.linalg.norm(ray_cam)

        # Rotate ray to world frame
        ray_world = R_body2world @ self.R_cam2body @ ray_cam

        # Intersect with water-surface plane z = water_z
        dz = ray_world[2]
        if abs(dz) < 1e-6:
            rospy.logwarn_throttle(2.0, '[pos_estimator] Ray parallel to water — skipped')
            return
        t = (self.water_z - cam_world[2]) / dz
        if t < 0:
            rospy.logwarn_throttle(2.0, '[pos_estimator] Intersection behind camera — skipped')
            return

        usv_world = cam_world + t * ray_world

        # Relative position: USV minus camera, in world ENU (Z=up)
        rel = usv_world - cam_world

        # Altitude cross-check from OBB size (Otter length ≈ 2.0 m)
        obb_px        = float(max(det.width, det.height)) if max(det.width, det.height) > 0 else 1.0
        alt_from_obb  = self.usv_length * self.fx / obb_px
        alt_from_odom = cam_world[2]

        # ── publish ──────────────────────────────────────────────────────────
        pose_msg = PoseStamped()
        pose_msg.header.stamp    = header.stamp
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = rel[0]
        pose_msg.pose.position.y = rel[1]
        pose_msg.pose.position.z = rel[2]
        pose_msg.pose.orientation.w = 1.0
        self.pub_pose.publish(pose_msg)

        dbg = PointStamped()
        dbg.header       = pose_msg.header
        dbg.point.x      = alt_from_obb
        dbg.point.y      = alt_from_odom
        dbg.point.z      = obb_px
        self.pub_debug.publish(dbg)

        rospy.loginfo_throttle(1.0,
            f'[pos_estimator] '
            f'USV rel to cam: ({rel[0]:+.2f}, {rel[1]:+.2f}, {rel[2]:+.2f}) m  '
            f'| alt_odom={alt_from_odom:.2f}  alt_obb={alt_from_obb:.2f}  '
            f'conf={det.confidence:.2f}'
        )


if __name__ == '__main__':
    try:
        PositionEstimator()
    except rospy.ROSInterruptException:
        pass

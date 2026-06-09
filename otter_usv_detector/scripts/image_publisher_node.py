#!/usr/bin/env python3
"""
image_publisher_node.py
───────────────────────
Publishes a single image (or a directory of images) on a ROS camera topic,
simulating a live camera stream for offline testing of the detector.

Useful for verifying the full detection pipeline without running Gazebo.

Parameters
----------
  ~image_path   str    path to a .jpg/.png image OR a directory of images
  ~topic        str    output topic  (default /overhead_cam/image_raw)
  ~rate_hz      float  publish rate  (default 10.0 Hz)
  ~frame_id     str    camera frame  (default overhead_cam_optical)
  ~loop         bool   loop directory images (default True)
"""

import glob
import os
import sys
import time

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class ImagePublisherNode:

    def __init__(self):
        rospy.init_node("image_publisher", anonymous=False)

        image_path = rospy.get_param("~image_path", "")
        topic      = rospy.get_param("~topic",    "/overhead_cam/image_raw")
        rate_hz    = rospy.get_param("~rate_hz",  10.0)
        self.frame_id = rospy.get_param("~frame_id", "overhead_cam_optical")
        self.loop  = rospy.get_param("~loop",     True)

        # ── Collect image file list ───────────────────────────────────────────
        if not image_path:
            rospy.logfatal("[img_pub] ~image_path is not set.")
            sys.exit(1)

        if os.path.isdir(image_path):
            exts   = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
            files  = []
            for ext in exts:
                files.extend(glob.glob(os.path.join(image_path, ext)))
            self.images = sorted(files)
            rospy.loginfo(f"[img_pub] Directory mode — {len(self.images)} images "
                          f"from {image_path}")
        elif os.path.isfile(image_path):
            self.images = [image_path]
            rospy.loginfo(f"[img_pub] Single-image mode — {image_path}")
        else:
            rospy.logfatal(f"[img_pub] image_path not found: {image_path}")
            sys.exit(1)

        if not self.images:
            rospy.logfatal("[img_pub] No images found.")
            sys.exit(1)

        self.bridge = CvBridge()
        self.pub    = rospy.Publisher(topic, Image, queue_size=1)
        self.rate   = rospy.Rate(rate_hz)

        rospy.loginfo(f"[img_pub] Publishing on '{topic}' at {rate_hz} Hz")

    def run(self):
        idx = 0
        n   = len(self.images)

        while not rospy.is_shutdown():
            path = self.images[idx % n]

            bgr = cv2.imread(path)
            if bgr is None:
                rospy.logwarn_throttle(5.0, f"[img_pub] Cannot read: {path}")
            else:
                msg             = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
                msg.header.stamp    = rospy.Time.now()
                msg.header.frame_id = self.frame_id
                self.pub.publish(msg)
                rospy.loginfo_throttle(
                    2.0,
                    f"[img_pub] Published frame {idx % n + 1}/{n}  — {os.path.basename(path)}"
                )

            idx += 1
            if idx >= n and not self.loop:
                rospy.loginfo("[img_pub] All images published. Shutting down.")
                break

            self.rate.sleep()


if __name__ == "__main__":
    try:
        ImagePublisherNode().run()
    except rospy.ROSInterruptException:
        pass

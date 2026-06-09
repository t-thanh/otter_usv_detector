#!/usr/bin/env python3
"""
gimbal_nadir_cmd.py
────────────────────
One-shot node: waits a configurable delay, then sends a nadir (straight-down)
pitch command to the gimbal position controller.

Used by pose_validation.launch so the camera is already pointing at the USV
before the operator arms the UAV and commands a climb.
"""
import rospy
from std_msgs.msg import Float64


def main():
    rospy.init_node('gimbal_nadir_cmd', anonymous=True)

    ns    = rospy.get_param('~ns',      'uav1')
    delay = rospy.get_param('~delay_s', 12.0)   # seconds to wait after node start

    pitch_topic = f'/{ns}/gimbal/position/pitch/command'
    yaw_topic   = f'/{ns}/gimbal/position/yaw/command'

    pub_pitch = rospy.Publisher(pitch_topic, Float64, queue_size=1)
    pub_yaw   = rospy.Publisher(yaw_topic,   Float64, queue_size=1)

    rospy.loginfo(
        f'[gimbal_nadir_cmd] Sleeping {delay:.0f} s before sending nadir command …')
    rospy.sleep(delay)

    if rospy.is_shutdown():
        return

    pub_yaw.publish(Float64(data=0.0))          # gimbal yaw = 0 (forward)
    rospy.sleep(0.2)
    pub_pitch.publish(Float64(data=1.5708))    # pitch = −π/2 = nadir
    rospy.loginfo(
        f'[gimbal_nadir_cmd] Nadir command sent: yaw=0  pitch=−π/2 → {pitch_topic}')


if __name__ == '__main__':
    main()

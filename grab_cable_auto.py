#!/usr/bin/env python3
import numpy as np
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import ros2_numpy as rnp

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_io import RTDEIOInterface as RTDEIO

# ---------- CONFIG ----------
ROBOT_IP = "192.168.1.100"
TRACKDLO_TOPIC = "/trackdlo/results_pc"   # change if needed
# ----------------------------

print("moin")

rtde_c = RTDEControl(ROBOT_IP)
rtde_r = RTDEReceive(ROBOT_IP)
rtde_io = RTDEIO(ROBOT_IP)

mat = np.array([
    [ 0.66488889, -0.68882831, -0.28885694,  0.43563037],
    [-0.57764439, -0.71935776,  0.38581263, -0.88582901],
    [-0.47355014, -0.08966594, -0.87619078,  0.58353029],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])


def ur_upright_x_align_rotvec(v, eps=1e-12):
    """
    Given a 3D vector v (in base frame), return a UR angle-axis rotation vector [rx, ry, rz]
    that keeps the TCP perfectly upright with z pointing DOWN (aligned with -base z),
    and aligns the TCP x-axis with the vector's XY heading.
    """
    v = np.asarray(v, dtype=float).reshape(3)
    vx, vy = v[0], v[1]

    # Handle degenerate XY case: choose ψ = 0 by convention.
    if abs(vx) + abs(vy) < eps:
        psi = 0.0
    else:
        psi = np.arctan2(vy, vx)

    c, s = np.cos(psi), np.sin(psi)

    # Rz(ψ)
    Rz = np.array([[ c, -s, 0.0],
                   [ s,  c, 0.0],
                   [0.0, 0.0, 1.0]])

    # Rx(π) — flips z to -z while keeping x
    Rx_pi = np.array([[1.0,  0.0,  0.0],
                      [0.0, -1.0,  0.0],
                      [0.0,  0.0, -1.0]])

    # Upright, z-down with x heading ψ
    R = Rz @ Rx_pi

    # For angle π, (R + I)/2 = a a^T gives the rotation axis a (unit).
    B = 0.5 * (R + np.eye(3))
    vals, vecs = np.linalg.eigh(B)
    a = vecs[:, np.argmax(vals)]

    # Numerical sign convention
    if a[2] < 0:
        a = -a

    rvec = np.pi * a

    # Clean tiny numerical rz
    if abs(rvec[2]) < 1e-10:
        rvec[2] = 0.0

    return rvec


def pick_up_cable(p1_cam, p2_cam):
    """
    p1_cam, p2_cam: np.array([x, y, z]) in *camera* frame.
    """
    rtde_io.setStandardDigitalOut(0, True)  # gripper open

    # hom coords
    p1 = mat @ np.array(p1_cam.tolist() + [1.0])
    p2 = mat @ np.array(p2_cam.tolist() + [1.0])

    direction = p2 - p1
    rvec = ur_upright_x_align_rotvec(direction[:3])

    # approach above cable
    pose = [p1[0], p1[1], 0.02, rvec[0], rvec[1], rvec[2]]
    rtde_c.moveL(pose)

    # go down to cable
    pose = [p1[0], p1[1], 0.0, rvec[0], rvec[1], rvec[2]]
    print("Move down pose:", pose)
    rtde_c.moveL(pose, speed=0.03)
    print("Moved to position")

    rtde_io.setStandardDigitalOut(0, False)  # gripper close

    # lift again
    pose = [p1[0], p1[1], 0.02, rvec[0], rvec[1], rvec[2]]
    rtde_c.moveL(pose, speed=0.06)


def drive_to_point(point):
    p_transformed = mat @ np.array(point.tolist() + [1.0])
    pose = [p_transformed[0], p_transformed[1], 0.02, 3.14159, 0.0, 0.0]
    rtde_c.moveL(pose)

    pose = [p_transformed[0], p_transformed[1], 0.0, 3.14159, 0.0, 0.0]
    print(pose)
    rtde_c.moveL(pose, speed=0.03)


class TrackDLOPickupNode(Node):
    def __init__(self):
        super().__init__('trackdlo_pickup_node')
        self.get_logger().info(f"Subscribing to TrackDLO topic: {TRACKDLO_TOPIC}")
        self.sub = self.create_subscription(
            PointCloud2,
            TRACKDLO_TOPIC,
            self.cloud_callback,
            10
        )
        self.already_picked = False

    def cloud_callback(self, msg: PointCloud2):
        if self.already_picked:
            return

        self.get_logger().info("Received TrackDLO point cloud, extracting first 2 points...")

        pts = rnp.numpify(msg)  # structured array with fields 'x', 'y', 'z', ...
        # convert to Nx3
        xyz = np.vstack((pts['x'], pts['y'], pts['z'])).T
        if xyz.shape[0] < 2:
            self.get_logger().warn(f"Point cloud has only {xyz.shape[0]} points, waiting for another message...")
            return

        p1_cam = xyz[0]
        p2_cam = xyz[1]

        self.get_logger().info(f"Using p1_cam={p1_cam}, p2_cam={p2_cam} for pickup")

        try:
            self.already_picked = True
            pick_up_cable(p1_cam, p2_cam)
            self.get_logger().info("Cable pickup sequence finished.")
        except Exception as e:
            self.get_logger().error(f"Error during pick_up_cable: {e}")
        finally:
            # Shutdown ROS after one pickup
            self.get_logger().info("Shutting down ROS node.")
            rclpy.shutdown()


def main():
    print("Starting TrackDLO pickup node...")
    rclpy.init()
    node = TrackDLOPickupNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        node.destroy_node()
        # Optionally stop robot script
        try:
            rtde_c.stopScript()
        except Exception:
            pass


if __name__ == "__main__":
    main()

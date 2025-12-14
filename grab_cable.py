import numpy as np
import time

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_io import RTDEIOInterface as RTDEIO

print("moin")
rtde_c = RTDEControl("192.168.1.100")
rtde_r = RTDEReceive("192.168.1.100")
rtde_io = RTDEIO("192.168.1.100")

mat = np.array([[ 0.66488889, -0.68882831, -0.28885694,  0.43563037],
 [-0.57764439, -0.71935776,  0.38581263, -0.88582901],
 [-0.47355014, -0.08966594, -0.87619078,  0.58353029],
 [ 0.    ,      0.     ,     0.    ,      1.        ]])

cable_points = np.array([[-0.17928019165992737, 0.10072904825210571, 0.7880000472068787],
 [-0.14412552118301392, 0.1124051883816719, 0.7670000195503235],
 [-0.10788216441869736, 0.1196867823600769, 0.7440000176429749],
 [-0.07162804156541824, 0.1206226721405983, 0.7250000238418579],
 [-0.03537477180361748, 0.11524972319602966, 0.7020000219345093],
 [-0.003620864124968648, 0.10642336308956146, 0.6850000619888306],
 [0.024504348635673523, 0.09242415428161621, 0.6710000038146973],
 [0.04944070056080818, 0.07480384409427643, 0.659000039100647],
 [0.07011006772518158, 0.05365493521094322, 0.6490000486373901],
 [0.08926869928836823, 0.03262195363640785, 0.6430000066757202],
 [0.1040160208940506, 0.007775059901177883, 0.6380000114440918],
 [0.11698397248983383, -0.017465904355049133, 0.6360000371932983],
 [0.12786541879177094, -0.04373685270547867, 0.6310000419616699],
 [0.1350208967924118, -0.07164838165044785, 0.6320000290870667],
 [0.13167735934257507, -0.09731844067573547, 0.6360000371932983]])


def ur_upright_x_align_rotvec(v, eps=1e-12):
    """
    Given a 3D vector v (in base frame), return a UR angle-axis rotation vector [rx, ry, rz]
    that keeps the TCP perfectly upright with z pointing DOWN (aligned with -base z),
    and aligns the TCP x-axis with the vector's XY heading.
    
    Returns an axis-angle with:
        - angle ~ pi
        - axis in the XY plane  -> rz = 0
        - magnitude sqrt(rx^2 + ry^2) ~ pi
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
    # Extract principal eigenvector of B to get axis a (lies in XY plane).
    B = 0.5 * (R + np.eye(3))
    vals, vecs = np.linalg.eigh(B)
    a = vecs[:, np.argmax(vals)]

    # Numerical sign convention: make rz >= 0 in magnitude-min sense (rz ~ 0 anyway).
    if a[2] < 0:
        a = -a

    # Rotation vector: angle * axis, angle = π
    rvec = np.pi * a

    # Clean tiny numerical rz
    if abs(rvec[2]) < 1e-10:
        rvec[2] = 0.0

    return rvec


def pick_up_cable(p1_cam, p2_cam):
    rtde_io.setStandardDigitalOut(0, True)  # gripper open
    # for i in range(len(cable_points)-1):
    p1 = mat @ np.array(p1_cam.tolist() + [1])
    p2 = mat @ np.array(p2_cam.tolist() + [1])
    direction = p2 - p1
    rvec = ur_upright_x_align_rotvec(direction[:3])
    pose = [p1[0], p1[1], 0.02, rvec[0], rvec[1], rvec[2]]
    rtde_c.moveL(pose)

    pose = [p1[0], p1[1], -0.031, rvec[0], rvec[1], rvec[2]]
    print(pose)
    rtde_c.moveL(pose, speed=0.03)
        # print(f"Moving to point {i}: {pose}")
        # rtde_c.moveL([p1[0], p1[1], 0.07, 2.03, 2.39, 0.0])
    print("Moved to position")
    rtde_io.setStandardDigitalOut(0, False)  # gripper close

    pose = [p1[0], p1[1], 0.02, rvec[0], rvec[1], rvec[2]]
    rtde_c.moveL(pose, speed=0.06)

def drive_to_point(point):
    p_transformed = mat @ np.array(point.tolist() + [1])
    pose = [p_transformed[0], p_transformed[1], 0.02, 3.14159, 0.0, 0.0]
    rtde_c.moveL(pose)

    pose = [p_transformed[0], p_transformed[1], -0.02, 3.14159, 0.0, 0.0]
    print(pose)
    rtde_c.moveL(pose, speed=0.03)


def main():
    print("debug1")
    #pick_up_cable(x,y)
    p = np.array([0.1584, -0.1375, 0.6294, 1.0])
    print("debug1")
    # drive_to_point(p)
    print(mat @ p)
    print("debug1")
    


if __name__ == "__main__":
    print("debug1")
    main()
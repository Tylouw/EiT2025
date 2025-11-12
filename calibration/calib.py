#!/usr/bin/env python3
"""
Hand–Eye calibration (eye-in-hand) using ChArUco + OpenCV.

Inputs
------
- A folder of images (already undistorted OR raw + intrinsics/distortion)
- A CSV with robot end-effector poses per image
- Camera intrinsics (YAML or NPZ)
- ChArUco board definition (set via CLI flags)

Outputs
-------
- Estimated hand–eye transform (camera->gripper) as 4x4, R|t
- JSON + NPZ files with the result
- Basic stats on how many pairs were used

CSV format (default)
--------------------
Assumes one row per image with columns:
    filename, px, py, pz, qx, qy, qz, qw
where (p*) is position of the gripper in BASE frame (meters) and (q*) is a unit quaternion
representing the orientation of the gripper in BASE frame. If your CSV stores BASE->GRIPPER,
set --robot_frame base_to_gripper (default). If it stores GRIPPER->BASE, set --robot_frame gripper_to_base.

We need GRIPPER->BASE for OpenCV. If you pass base_to_gripper (default), we will invert it.

Usage
-----
python handeye_charuco.py \
  --images /path/to/images \
  --poses /path/to/poses.csv \
  --intrinsics /path/to/camera_intrinsics.yaml \
  --charuco_squares_x 5 --charuco_squares_y 7 \
  --charuco_square_len 0.030 --charuco_marker_len 0.022 \
  --aruco_dict DICT_5X5_100 \
  --robot_frame base_to_gripper

Notes
-----
OpenCV's calibrateHandEye expects:
- R_gripper2base, t_gripper2base  (list)
- R_target2cam,  t_target2cam     (list)
It returns:
- R_cam2gripper, t_cam2gripper

So the final printed T_cam_gripper brings points from camera frame to gripper frame.
"""

import argparse
import glob
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd


# ---------------------------- SE(3) helpers ---------------------------------
def quat_to_R(qx, qy, qz, qw):
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.dot(q, q)
    if n < 1e-12:
        return np.eye(3)
    q = q / np.sqrt(n)
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R

# def Rt_to_T(R, t):
#     import numpy as np
#     T = np.eye(4, dtype=float)
#     T[:3, :3] = np.asarray(R, dtype=float)
#     T[:3, 3]  = np.asarray(t, dtype=float).reshape(3)  # accepts (3,), (3,1), (1,3)
#     return T

def axis_angle_to_rot_matrix(r):
    """
    Convert UR axis-angle vector (rx, ry, rz) to a 3x3 rotation matrix.
    """
    theta = np.linalg.norm(r)
    
    # If angle ~ 0, return identity
    if theta < 1e-8:
        return np.eye(3)

    # Normalized rotation axis
    k = r / theta
    kx, ky, kz = k

    K = np.array([
        [0,   -kz,  ky],
        [kz,   0,  -kx],
        [-ky,  kx,   0]
    ])

    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R



def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return R

def se3_log(T):
    R = T[:3,:3]; t = T[:3,3]
    theta = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
    if theta < 1e-12:
        w = np.zeros(3); v = t
    else:
        w_hat = (R - R.T) * (0.5 / np.sin(theta))
        w = np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]]) * theta
        A = np.sin(theta)/theta
        B = (1-np.cos(theta))/(theta**2)
        V = np.eye(3) + B*(R - R.T)/2 + ((1-A)/(theta**2))*(w[:,None]@w[None,:])
        v = np.linalg.inv(V) @ t
    return w, v

def rot_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1)))

def transform_inv(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4); Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t
    return Ti

def compose(A,B):
    C = np.eye(4)
    C[:3,:3] = A[:3,:3] @ B[:3,:3]
    C[:3,3] = A[:3,:3] @ B[:3,3] + A[:3,3]
    return C

def Rt_to_T(R,t):
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=t
    return T

# ----------------------- Intrinsics I/O helpers ------------------------------
def load_intrinsics(path):
    """
    Optimized for ROS CameraInfo-style YAML:
      K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
      D: [k1, k2, p1, p2, k3]  # 'plumb_bob' (radial-tangential)
    Returns:
      K (3x3 float64), D (Nx1 float64 column vector)
    """
    import yaml
    import numpy as np

    with open(path, "r") as f:
        y = yaml.safe_load(f)

    # Mandatory keys in your file
    if "K" not in y or "D" not in y:
        raise ValueError("YAML must contain 'K' (9 values) and 'D' (distortion coeffs).")

    K_list = y["K"]
    D_list = y["D"]

    if len(K_list) != 9:
        raise ValueError(f"'K' must have 9 elements, got {len(K_list)}.")
    # 'plumb_bob' usually has 5 coeffs (k1,k2,t1,t2,k3), but allow variable length
    if not isinstance(D_list, (list, tuple)) or len(D_list) < 4:
        raise ValueError(f"'D' should be a list of >=4 coeffs, got {len(D_list)}.")

    K = np.array(K_list, dtype=np.float64).reshape(3, 3)
    D = np.array(D_list, dtype=np.float64).reshape(-1, 1)  # column vector (N x 1)

    return K, D


# ---------------------------- ChArUco helpers --------------------------------
ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

def make_charuco(params):
    import cv2

    # dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[params.aruco_dict])

    # board ctor (new API); fall back to legacy factory if needed
    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (params.charuco_squares_x, params.charuco_squares_y),
            params.charuco_square_len,
            params.charuco_marker_len,
            dictionary
        )
    else:
        board = cv2.aruco.CharucoBoard_create(
            params.charuco_squares_x, params.charuco_squares_y,
            params.charuco_square_len, params.charuco_marker_len,
            dictionary
        )

    # detector params (new API classes with legacy _create fallback)
    if hasattr(cv2.aruco, "DetectorParameters"):
        det_params = cv2.aruco.DetectorParameters()
    else:
        det_params = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "CharucoParameters"):
        charuco_params = cv2.aruco.CharucoParameters()
    else:
        # legacy OpenCV didn’t expose this; leave as None
        charuco_params = None

    return dictionary, board, det_params, charuco_params



# ------------------------------- Main logic ----------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Hand–Eye calibration with ChArUco + OpenCV.")
    p.add_argument("--images", required=True, help="Folder with images.")
    p.add_argument("--poses", required=True, help="CSV with robot poses.")
    p.add_argument("--intrinsics", required=True, help="YAML/XML/NPZ intrinsics file.")
    p.add_argument("--aruco_dict", default="DICT_5X5_100", choices=list(ARUCO_DICT_MAP.keys()))
    p.add_argument("--charuco_squares_x", type=int, required=True)
    p.add_argument("--charuco_squares_y", type=int, required=True)
    p.add_argument("--charuco_square_len", type=float, required=True, help="Square side length (meters).")
    p.add_argument("--charuco_marker_len", type=float, required=True, help="Marker side length (meters).")
    p.add_argument("--image_glob", default="*.png", help="Pattern for images (e.g., *.png, *.jpg).")
    p.add_argument("--csv_filename_col", default="filename", help="Column name mapping row to image file.")
    p.add_argument("--csv_px", default="px")
    p.add_argument("--csv_py", default="py")
    p.add_argument("--csv_pz", default="pz")
    p.add_argument("--csv_qx", default="qx")
    p.add_argument("--csv_qy", default="qy")
    p.add_argument("--csv_qz", default="qz")
    p.add_argument("--csv_qw", default="qw")
    p.add_argument("--robot_frame", default="base_to_gripper",
                   choices=["base_to_gripper", "gripper_to_base"],
                   help="Frame of the pose stored in the CSV.")
    p.add_argument("--undistort", action="store_true",
                   help="Undistort images before detection (recommended).")
    p.add_argument("--min_charuco_corners", type=int, default=10,
                   help="Require at least this many interpolated ChArUco corners per image.")
    p.add_argument("--visualize", action="store_true", help="Show detections (press any key to step).")
    p.add_argument("--save_result_prefix", default="handeye_result",
                   help="Prefix for saving results (NPZ + JSON).")
    return p.parse_args()

def load_images(folder, pattern):
    paths = sorted(glob.glob(str(Path(folder) / pattern)))
    return [Path(p) for p in paths]

def read_poses(csv_path, args):
    df = pd.read_csv(csv_path)
    # required = [args.csv_filename_col, args.csv_px, args.csv_py, args.csv_pz,
    #             args.csv_qx, args.csv_qy, args.csv_qz, args.csv_qw]
    required = ["filename", "px", "py", "pz", "rx", "ry", "rz"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")
    return df

def find_row_for_image(df, fname_col, image_path):
    # Match by filename only (without directories). Try exact, then try basename.
    name = Path(image_path).name
    row = df[df[fname_col] == name]
    if len(row) == 0:
        # sometimes CSV holds stem (no extension)
        row = df[df[fname_col] == Path(name).stem]
    if len(row) == 0:
        # try full relative path
        row = df[df[fname_col] == str(image_path)]
    if len(row) == 0:
        return None
    return row.iloc[0]

def detect_charuco_pose(img, board, dictionary, det_params, charuco_params, K, D):
    import cv2

    # ---- Preferred path: OpenCV ≥ 4.7 new API ----
    if hasattr(cv2.aruco, "CharucoDetector"):
        # Build optional refine params if available
        refine_params = None
        if hasattr(cv2.aruco, "RefineParameters"):
            try:
                refine_params = cv2.aruco.RefineParameters()  # defaults are fine
            except Exception:
                refine_params = None

        # IMPORTANT: Do NOT pass `dictionary` here. The board already contains it.
        cd = cv2.aruco.CharucoDetector(
            board,
            charuco_params if charuco_params is not None else None,
            det_params if det_params is not None else None,
            refine_params
        )

        # Returns: charucoCorners, charucoIds, markerCorners, markerIds
        charuco_corners, charuco_ids, marker_corners, marker_ids = cd.detectBoard(img)
        if hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
            ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, board, K, D, None, None
            )
        else:
            ok, rvec, tvec = estimate_charuco_pose_pnp(
                charuco_corners, charuco_ids, board, K, D
            )

        if not ok:
            return None

        return rvec, tvec, marker_corners, marker_ids, charuco_corners, charuco_ids

def estimate_charuco_pose_pnp(charuco_corners, charuco_ids, board, K, D):
    """
    Estimate ChArUco board pose via generic PnP (works across OpenCV versions).
    Uses the board's chessboard corner 3D points (z=0) indexed by the detected charuco IDs.
    Returns (ok, rvec, tvec).
    """
    import numpy as np
    import cv2

    # Need at least 4 corners for a stable PnP
    if charuco_ids is None or len(charuco_ids) < 4:
        return False, None, None

    # --- Get 3D corner template in board frame (z=0) across API variants ---
    if hasattr(board, "chessboardCorners"):
        # Legacy OpenCV: attribute
        all_obj = np.asarray(board.chessboardCorners, dtype=np.float32)
    elif hasattr(board, "getChessboardCorners"):
        # Modern OpenCV: method
        all_obj = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    else:
        raise RuntimeError("This OpenCV build lacks both 'chessboardCorners' and 'getChessboardCorners()'.")

    # Flatten ids to int indices into the template
    sel = charuco_ids.flatten().astype(int)

    # Defensive checks
    max_id = sel.max()
    if max_id >= len(all_obj):
        raise ValueError(
            f"ChArUco ID {max_id} out of range for board corners (len={len(all_obj)}). "
            "Check that your squaresX/squaresY match the printed board."
        )

    obj_pts = all_obj[sel, :]  # (M,3)

    # charuco_corners is (M,1,2) or (M,2). Convert to (M,2) float32.
    img_pts = np.asarray(charuco_corners, dtype=np.float32)
    if img_pts.ndim == 3 and img_pts.shape[1] == 1 and img_pts.shape[2] == 2:
        img_pts = img_pts[:, 0, :]  # (M,2)
    elif img_pts.ndim == 2 and img_pts.shape[1] == 2:
        pass
    else:
        img_pts = img_pts.reshape(-1, 2)

    # Solve PnP (radial-tangential ‘plumb_bob’ works fine here)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE
    )
    return bool(ok), rvec, tvec




def main():
    # print(list(ARUCO_DICT_MAP.keys()))
    args = parse_args()

    K, D = load_intrinsics(args.intrinsics)
    dictionary, board, det_params, charuco_params = make_charuco(args)

    image_paths = load_images(args.images, args.image_glob)
    if len(image_paths) == 0:
        print("No images found. Check --images and --image_glob.", file=sys.stderr)
        sys.exit(1)

    df = read_poses(args.poses, args)

    # Containers for OpenCV calibrateHandEye
    R_g2b_list, t_g2b_list = [], []
    R_t2c_list, t_t2c_list = [], []

    used = 0
    skipped = 0

    for img_path in image_paths:
        row = find_row_for_image(df, args.csv_filename_col, img_path)
        if row is None:
            print(f"[WARN] No pose row for image: {img_path.name}")
            skipped += 1
            continue

        # Robot pose from CSV
        px, py, pz = float(row[args.csv_px]), float(row[args.csv_py]), float(row[args.csv_pz])
        # qx, qy, qz, qw = float(row[args.csv_qx]), float(row[args.csv_qy]), float(row[args.csv_qz]), float(row[args.csv_qw])
        rx, ry, rz = float(row["rx"]), float(row["ry"]), float(row["rz"])
        # R_b2g = quat_to_R(qx, qy, qz, qw)
        R_b2g = axis_angle_to_rot_matrix(np.array([rx, ry, rz], dtype=float))
        t_b2g = np.array([px, py, pz], dtype=float)
        tcp_to_marker = np.array([[0.0, -1.0, 0.0, 0.09703],
                                    [0.0, 0.0, -1.0, 0.01049],
                                    [1.0, 0.0, 0.0, 0.00007],
                                    [0.0, 0.0, 0.0, 1.0]])

        tcp_to_marker = np.linalg.inv(tcp_to_marker)
        # T_b_g = Rt_to_T(R_b2g, t_b2g) @ tcp_to_marker  # BASE->GRIPPER @ TCP->MARKER

        T_b_g = Rt_to_T(R_b2g, t_b2g)

        if args.robot_frame == "base_to_gripper":
            # OpenCV wants GRIPPER->BASE, so invert
            T_g_b = invert_T(T_b_g)
        else:
            # already gripper->base
            T_g_b = T_b_g

        R_g2b = T_g_b[:3, :3]
        t_g2b = T_g_b[:3, 3]

        # Load and (optionally) undistort image
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            skipped += 1
            continue

        if args.undistort:
            img = cv2.undistort(img, K, D)

        det = detect_charuco_pose(img, board, dictionary, det_params, charuco_params, K, D)
        if det is None:
            print(f"[WARN] ChArUco not found in {img_path.name}")
            skipped += 1
            continue

        rvec, tvec, corners, ids, charuco_corners, charuco_ids = det

        if charuco_corners is None or len(charuco_corners) < args.min_charuco_corners:
            print(f"[WARN] Too few ChArUco corners ({len(charuco_corners) if charuco_corners is not None else 0}) in {img_path.name}")
            skipped += 1
            continue

        R_t2c = rodrigues_to_R(rvec)
        t_t2c = tvec.reshape(3)
        # print(t_t2c)

        # Convert from corner frame to marker frame
        # T_t2c = Rt_to_T(R_t2c, t_t2c)
        # # T_t2c = np.linalg.inv(tcp_to_marker) @ T_t2c
        # T_t2c = tcp_to_marker @ T_t2c

        # # Now use these transformed poses
        # R_t2c = T_t2c[:3, :3]
        # t_t2c = T_t2c[:3, 3]
        # # print(t_t2c)

        R_g2b_list.append(R_g2b)
        t_g2b_list.append(t_g2b)
        R_t2c_list.append(R_t2c)
        t_t2c_list.append(t_t2c)

        used += 1

        if args.visualize:
            vis = img.copy()
            # draw detected markers and pose
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.drawFrameAxes(vis, K, D, rvec, tvec, args.charuco_square_len * 2.0)
            cv2.putText(vis, f"{img_path.name}  corners:{len(charuco_corners)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, f"{img_path.name}  corners:{len(charuco_corners)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("ChArUco detection", vis)
            cv2.waitKey(0)

    if args.visualize:
        cv2.destroyAllWindows()

    print(f"\nPairs used: {used}   skipped: {skipped}")
    if used < 3:
        print("Not enough pairs for a stable solution. Collect more data or adjust detection.", file=sys.stderr)
        sys.exit(2)

    # OpenCV calibrateHandEye
    print(t_g2b_list)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_g2b_list,
        t_gripper2base=t_g2b_list,
        R_target2cam=R_t2c_list,
        t_target2cam=t_t2c_list,
        method=cv2.CALIB_HAND_EYE_TSAI  # or CALIB_HAND_EYE_PARK / DANIILIDIS / HORAUD
    )

    # print(t_cam2gripper)

    t_cam2gripper = np.asarray(t_cam2gripper, dtype=float).reshape(3)

    T_cam_gripper = Rt_to_T(R_cam2gripper, t_cam2gripper)

    # Also provide the inverse (gripper->camera) for convenience
    T_gripper_cam = invert_T(T_cam_gripper)

    def mat_to_list(M):
        return [[float(v) for v in row] for row in M]

    print("\n=== Result (camera -> gripper) ===")
    print(T_cam_gripper)

    print("\n=== Inverse (gripper -> camera) ===")
    print(T_gripper_cam)

    # Save results
    prefix = args.save_result_prefix
    np.savez(f"{prefix}.npz",
             R_cam2gripper=R_cam2gripper,
             t_cam2gripper=t_cam2gripper.reshape(3,1),
             T_cam_gripper=T_cam_gripper,
             T_gripper_cam=T_gripper_cam)

    with open(f"{prefix}.json", "w") as f:
        json.dump({
            "T_cam_gripper": mat_to_list(T_cam_gripper),
            "T_gripper_cam": mat_to_list(T_gripper_cam),
            "pairs_used": int(used),
            "pairs_skipped": int(skipped),
            "method": "OpenCV calibrateHandEye (TSAI)",
            "notes": "T_cam_gripper maps points from camera frame to gripper frame."
        }, f, indent=2)

    print(f"\nSaved: {prefix}.npz and {prefix}.json")


    X = T_cam_gripper  # camera -> gripper
    Xinvt = transform_inv(X)

    rot_errs_deg = []
    tran_errs_mm = []

    for Rg2b, tg2b, Rt2c, tt2c in zip(R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list):
        A = Rt_to_T(Rg2b, tg2b)   # gripper->base
        B = Rt_to_T(Rt2c, tt2c)   # target->camera
        E = compose(compose(A, X), transform_inv(compose(X, B)))  # should be I
        rot_errs_deg.append(rot_angle_deg(E[:3,:3]))
        tran_errs_mm.append(1e3*np.linalg.norm(E[:3,3]))

    print("\n[AX≈XB residuals]")
    print(f"  Rotation error:  mean={np.mean(rot_errs_deg):.3f}°  rms={np.sqrt(np.mean(np.square(rot_errs_deg))):.3f}°")
    print(f"  Translation err: mean={np.mean(tran_errs_mm):.2f} mm  rms={np.sqrt(np.mean(np.square(tran_errs_mm))):.2f} mm")

if __name__ == "__main__":
    main()

# python calibration/calib.py --images ./calibration/img/ --poses ./calibration/poses.csv --intrinsics ./calibration/camera_intrinsics.yaml --charuco_squares_x 7 --charuco_squares_y 5 --charuco_square_len 0.037 --charuco_marker_len 0.027 --image_glob *.jpg
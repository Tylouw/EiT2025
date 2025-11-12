import cv2
import numpy as np
import pandas as pd 
import argparse
import yaml

charuco_squares_x = 14
charuco_squares_y = 9
charuco_square_len = 0.02  # meters
charuco_marker_len = 0.014  # meters


def parse_args():
    p = argparse.ArgumentParser(description="Hand–Eye calibration with ChArUco + OpenCV.")
    p.add_argument("--vis", action="store_true", help="Show detections (press any key to step).")
    return p.parse_args()

def read_poses(csv_path, args):
    df = pd.read_csv(csv_path)
    # required = [args.csv_filename_col, args.csv_px, args.csv_py, args.csv_pz,
    #             args.csv_qx, args.csv_qy, args.csv_qz, args.csv_qw]
    required = ["filename", "px", "py", "pz", "rx", "ry", "rz"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")
    return df

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

def Rt_to_T(R,t):
    T = np.eye(4)
    T[:3,:3]=R
    T[:3,3]=t.flatten()  # Flatten the translation vector to ensure it's 1D
    return T

def detect_charuco_pose(img, K, D):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

    board = cv2.aruco.CharucoBoard((charuco_squares_x, charuco_squares_y), charuco_square_len, charuco_marker_len, dictionary)
    
    charuco_params = cv2.aruco.CharucoParameters()
    det_params = cv2.aruco.DetectorParameters()
    refine_params = cv2.aruco.RefineParameters()

    cd = cv2.aruco.CharucoDetector(board, charuco_params, det_params, refine_params)

    # Returns: charucoCorners, charucoIds, markerCorners, markerIds
    charuco_corners, charuco_ids, marker_corners, marker_ids = cd.detectBoard(img)

    

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board, K, D, None, None
    )

    print(ok)

    return ok, rvec, tvec, marker_corners, marker_ids, charuco_corners, charuco_ids


def main():
    args = parse_args()

    with open("./calibration/camera_intrinsics.yaml", "r") as f:
        y = yaml.safe_load(f)

    K = np.array(y["K"], dtype=np.float64).reshape(3, 3)
    D = np.array(y["D"], dtype=np.float64).reshape(-1, 1)

    rotmat_marker_to_base = []
    trans_marker_to_base = []
    rotmat_marker_to_cam = []
    trans_marker_to_cam = []
    for index, row in read_poses("./calibration/cal5/robot_positions.csv", None).iterrows():
        img_path = "./calibration/cal5/" + row["filename"] + ".jpg"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        ok, rvec, tvec, marker_corners, marker_ids, charuco_corners, charuco_ids = detect_charuco_pose(img, K, D)
        
        if not ok:
            print(f"Pose detection failed for image: {img_path}")
            continue

        px, py, pz = float(row["px"]), float(row["py"]), float(row["pz"])
        rx, ry, rz = float(row["rx"]), float(row["ry"]), float(row["rz"])
        tcp_to_marker = np.array([[0.0, -1.0, 0.0, 0.09],
                                  [-1.0, 0.0, 0.0, 0.14],
                                  [0.0, 0.0, -1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
        
        R_base_to_tcp = axis_angle_to_rot_matrix(np.array([rx, ry, rz], dtype=float))
        t_base_to_tcp = np.array([[px], [py], [pz]], dtype=float)
        T_base_to_marker = Rt_to_T(R_base_to_tcp, t_base_to_tcp) @ tcp_to_marker
        marker_to_base = np.linalg.inv(T_base_to_marker)

        rotmat_marker_to_base.append(marker_to_base[:3, :3])
        trans_marker_to_base.append(marker_to_base[:3, 3])


        cam_to_marker = Rt_to_T(cv2.Rodrigues(rvec)[0], tvec) @ tcp_to_marker
        marker_to_cam = np.linalg.inv(cam_to_marker)

        rotmat_marker_to_cam.append(cam_to_marker[:3, :3])
        trans_marker_to_cam.append(cam_to_marker[:3, 3])

        # print("marker pose:")
        # print(base_to_tcp @ tcp_to_marker @ np.linalg.inv(cam_to_marker))
        if args.vis and ok:
            vis = img.copy()

            cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
            cv2.drawFrameAxes(vis, K, D, rvec, tvec, charuco_square_len * 2.0)
            # cv2.putText(vis, f"{img_path.name}  corners:{len(charuco_corners)}",
            #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
            # cv2.putText(vis, f"{img_path.name}  corners:{len(charuco_corners)}",
            #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow("img", vis)
            cv2.waitKey(0)

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(rotmat_marker_to_base, trans_marker_to_base, rotmat_marker_to_cam, trans_marker_to_cam, cv2.CALIB_HAND_EYE_TSAI)

    final_mat = Rt_to_T(R_cam2gripper, t_cam2gripper)
    print("Final hand-eye calibration matrix (camera to gripper):")
    print(final_mat)

    point = np.array([ 0.133493572473526, -0.086402028799057, 0.6330000162124634, 1.0])#

    res = final_mat @ point
    print(res )

if __name__ == "__main__":
    main()
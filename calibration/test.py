import rtde_receive
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.100")  # robot IP
print("TCP pose:", rtde_r.getActualTCPPose())
print("Joint positions:", rtde_r.getActualQ())
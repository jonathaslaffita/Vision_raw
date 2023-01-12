import cv2 as cv
import cv2
from cv2 import aruco
import numpy as np
import time
time1 = 0.0
time2 = 0.0
calib_data = np.load("calibration.npz")
# print(calib_data.files)

cam_mat = calib_data["cameraMatrix"]
dist_coef = calib_data["dist"]
r_vectors = calib_data["rvecs"]
t_vectors = calib_data["tvecs"]

MARKER_SIZE = np.array([15.0 , 5.0 ,5.0 ,5.0, 5.0, 2.5, 2.5])  # centimeters
Marker_Object_corners = np.array([[[0.0, 6.0, 0],[15.0, 6.0, 0],[15.0, 21.0 ,0],[0.0, 21.0,0]],[[0.0, 0.0,0],[5.0, 0.0,0],[5.0, 5.0,0],[0.0, 5.0,0]],[[10.0, 0.0,0],[15.0, 0.0,0],[15.0, 5.0,0],[10.0, 5.0,0]],[[0.0, 22.0,0],[5.0, 22.0,0],[5.0, 27.0,0],[0.0, 27.0,0]],[[10.0, 22.0,0],[15.0, 22.0,0],[15.0, 27.0,0],[10.0, 27.0,0]],[[6.25, 1.25,0],[8.75, 1.25,0],[8.75, 3.625,0],[6.25, 3.625,0]],[[6.25, 23.25,0],[8.75, 23.25,0],[8.75, 25.75,0],[6.25, 25.75,0]]])

marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)

param_markers = aruco.DetectorParameters_create()

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    (marker_corners, marker_IDs, reject) = aruco.detectMarkers(
        gray_frame, marker_dict, parameters = param_markers
    )
    
    if marker_corners:
        object_points = Marker_Object_corners[marker_IDs]
        # rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
        #     marker_corners, MARKER_SIZE, cam_mat, dist_coef
        # )
        # for marker_id, marker_cornerss in zip(marker_IDs, marker_corners):
        # Print the ID and corner locations of the current marker
        
        if len(marker_IDs) == 1:
            image_point = marker_corners[0]
            
        else:
            image_point = np.concatenate(marker_corners, axis=0)
            image_point = np.concatenate(image_point, axis=0)
        objectpoint = np.concatenate(object_points, axis=0)
        objectpoint = np.concatenate(objectpoint, axis=0)

        print(image_point)
        print(objectpoint)
        _ ,rVec, tVec = cv2.solvePnP(objectpoint, image_point, cam_mat, dist_coef)
            # Since there was mistake in calculating the distance approach point-outed in the Video Tutorial's comment
            # so I have rectified that mistake, I have test that out it increase the accuracy overall.
            # Calculating the distance
        distance = np.sqrt(
            tVec[2] ** 2 + tVec[0] ** 2 + tVec[1] ** 2
        )
        
        # Draw the pose of the marker
        point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec, tVec, 4, 4)
        cv.putText(
            frame,
            f"id: {1} Dist: {distance}",
            (10, 0),
            cv.FONT_HERSHEY_PLAIN,
            1.3,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            frame,
            f"x:{(tVec[0],1)} y: {(tVec[1],1)} ",
            (-10,0),
            cv.FONT_HERSHEY_PLAIN,
            1.0,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )
        print(len(marker_IDs))
        print(distance)
        time1 = time2
        time2 = time.time()
        # print(time.time())
        # print((time2 - time1))
        
    cv.imshow("frame", frame) 
    key = cv.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv.destroyAllWindows()

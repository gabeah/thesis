# Function that should? put it all together

import cv2 as cv
import glob
import numpy as np
import sys
from scipy import linalg
import yaml
import os
import dmx
from dmx.colour import RED, GREEN, BLUE
import calib_cam as c

#This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}

def main():
    if len(sys.argv) != 2:
        print('Call with settings filename: "python3 calibrate.py calibration_settings.yaml"')
        quit()

    print("Light Tracking Software: Version 0 (Prototype)")

    print("Beginning with Camera Calibration")

    #Open and parse the settings file
    c.parse_calibration_settings_file(sys.argv[1])


    """Step1. Save calibration frames for single cameras"""
    c.save_frames_single_camera('camera0') #save frames for camera0
    c.save_frames_single_camera('camera1') #save frames for camera1


    """Step2. Obtain camera intrinsic matrices and save them"""
    #camera0 intrinsics
    images_prefix = os.path.join('frames', 'camera0*')
    cmtx0, dist0 = c.calibrate_camera_for_intrinsic_parameters(images_prefix) 
    c.save_camera_intrinsics(cmtx0, dist0, 'camera0') #this will write cmtx and dist to disk
    #camera1 intrinsics
    images_prefix = os.path.join('frames', 'camera1*')
    cmtx1, dist1 = c.calibrate_camera_for_intrinsic_parameters(images_prefix)
    c.save_camera_intrinsics(cmtx1, dist1, 'camera1') #this will write cmtx and dist to disk


    """Step3. Save calibration frames for both cameras simultaneously"""
    c.save_frames_two_cams('camera0', 'camera1') #save simultaneous frames


    """Step4. Use paired calibration pattern frames to obtain camera0 to camera1 rotation and translation"""
    frames_prefix_c0 = os.path.join('frames_pair', 'camera0*')
    frames_prefix_c1 = os.path.join('frames_pair', 'camera1*')
    R, T = c.stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)


    """Step5. Open both camera feeds and modify the masking for the feeds to detect the LED well."""
    # Added to the process!!
    upper_blue, lower_blue = c.calib_mask('camera0', 'camera1')

    """Step6. Save calibration data where camera0 defines the world space origin."""
    #camera0 rotation and translation is identity matrix and zeros vector
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))

    c.save_extrinsic_calibration_parameters(R0, T0, R, T, upper_blue, lower_blue) #this will write R and T to disk
    R1 = R; T1 = T #to avoid confusion, camera1 R and T are labeled R1 and T1
    #check your calibration makes sense
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R1, T1]
    c.check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift = 60.)

    print("Calibration complete, ready for loop!")

    with DMXInterface("AVRDMX") as interface:

        universe = DMXUniverse()

        # Build the lighting profile for the ROBE Esprite (TODO TOMORROW)
        lights = []

        esprite = DMXLight3Slot(1) # Placeholder Address... Add 14 to account for Esprite addr.
        #(We will add an esprite profile later....)

        cap0 = cv.VideoCapture(calibration_settings[camera0_name])
        cap1 = cv.VideoCapture(calibration_settings[camera1_name])

        if not cap0.isOpened():
            print("ERR CAP 1")
            exit()
        if not cap1.isOpened():
            print("ERR CAP 2")
            exit()

        #RT matrix for C1 is identity.
        RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
        P1 = cmtx1 @ RT1 #projection matrix for C1
    
        #RT matrix for C2 is the R and T obtained from stereo calibration.
        RT2 = np.concatenate([R, T], axis = -1)
        P2 = cmtx2 @ RT2 #projection matrix for C2

        while True:

            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()

            if not ret0 or ret1:
                print("Error.. Cannot recieve frame.. Exiting")
                quit()
            
            hsv0 = cv.cvtColor(frame0, cv.COLOR_BGR2HSV)
            hsv1 = cv.cvtColor(frame1, cv.COLOR_BGR2HSV)

            mask0 = cv.inRange(hsv0, lowerb=lower_blue, upperb=upper_blue)
            mask1 = cv.inRange(hsv1, lowerb=lower_blue, upperb=upper_blue)

            params = cv.SimpleBlobDetector_Params()

            params.filterByArea = True
            params.minArea = 100
            params.filterByCircularity = False
            params.filterByConvexity = False
            params.filterByInertia = True

            detector = cv.SimpleBlobDetector_create(params)

            keypoints = detector.detect(mask1)
            keypoints1 = detector.detect(mask2)
            pts0 = np.array([key_point.pt for key_point in keypoints])
            pts1 = np.array([key_point.pt for key_point in keypoints1])

            pt0 = [sum(x)/len(x) for x in zip(*pts0)]
            pt1 = [sum(y)/len(y) for y in zip(*pts1)]
            
            # If light is detected in both keypoints, send update \o/
            if pt0 and pt1:
                esprite.set_colour(GREEN)
            else:
                esprite.set_colour(RED)

            interface.set_frame(universe.serialise())
            interface.send_update()

            # This is going to be worked on tomorrow
            # DLT(P1, P2, pt0, pt1) # ADD STUFF HERE...

if __name__ == '__main__':

    main()
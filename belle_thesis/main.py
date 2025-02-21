# Function that should? put it all together

import cv2 as cv
import glob
import numpy as np
import sys
from scipy import linalg
import yaml
import os
# from dmx import DMXInterface, DMXUniverse, DMXLight
# from dmx.colour import RED, GREEN, BLUE
import calib_cam as cc
import calib_lamp as cl
import pyenttec as ent
from esprite_profile import Esprite

#This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}
cam0_settings = {}
cam1_settings = {}

universe = ent.DMXConnection("/dev/ttyUSB0")

# Build the lighting profile for the ROBE Esprite (TODO TOMORROW)
lights = []


title_window = "testing world"
class trackbar_var(object):
    def __init__(self, val=0):
        self.val = val
    def change(self, val):
        print("CHANGING")
        self.val = val

def write_dmx():
    for light in lights:
        #print(light.serialise_pydmx())
        for i, chan in enumerate(light.serialise_pydmx()):
            universe.set_channel(i, chan)
    universe.render()

def calibrate_cam_settings():
    print("The big kahuna")
    #Open and parse the settings file
    cc.parse_calibration_settings_file(sys.argv[1])


    """Step1. Save calibration frames for single cameras"""
    cc.save_frames_single_camera('camera0') #save frames for camera0
    cc.save_frames_single_camera('camera1') #save frames for camera1


    """Step2. Obtain camera intrinsic matrices and save them"""
    #camera0 intrinsics
    images_prefix = os.path.join('frames', 'camera0*')
    cmtx0, dist0 = cc.calibrate_camera_for_intrinsic_parameters(images_prefix) 
    cc.save_camera_intrinsics(cmtx0, dist0, 'camera0') #this will write cmtx and dist to disk
    #camera1 intrinsics
    images_prefix = os.path.join('frames', 'camera1*')
    cmtx1, dist1 = cc.calibrate_camera_for_intrinsic_parameters(images_prefix)
    cc.save_camera_intrinsics(cmtx1, dist1, 'camera1') #this will write cmtx and dist to disk


    """Step3. Save calibration frames for both cameras simultaneously"""
    cc.save_frames_two_cams('camera0', 'camera1') #save simultaneous frames


    """Step4. Use paired calibration pattern frames to obtain camera0 to camera1 rotation and translation"""
    frames_prefix_c0 = os.path.join('frames_pair', 'camera0*')
    frames_prefix_c1 = os.path.join('frames_pair', 'camera1*')
    R, T = cc.stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)


    """Step5. Open both camera feeds and modify the masking for the feeds to detect the LED well."""
    # Added to the process!!
    upper_blue, lower_blue = c.calib_mask('camera0', 'camera1')

    """Step6. Save calibration data where camera0 defines the world space origin."""
    #camera0 rotation and translation is identity matrix and zeros vector
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))

    cc.save_extrinsic_calibration_parameters(R0, T0, R, T, upper_blue, lower_blue) #this will write R and T to disk
    R1 = R; T1 = T #to avoid confusion, camera1 R and T are labeled R1 and T1
    #check your calibration makes sense
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R1, T1]
    cc.check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift = 60.)

def calibrate_instrument():
    print("TODO")

def main():
    if len(sys.argv) != 2:
        print('Call with settings filename: "python3 calibrate.py calibration_settings.yaml"')
        quit()

    print("Light Tracking Software: Version 0 (Prototype)")
    print("Beginning with Camera Calibration")
    
    try:
        with open("./camera_parameters/camera_mask_results.dat", "r") as blus:
            blu_types = blus.readlines()
            print(blu_types)
            calibration_settings["upper blue"] = np.array(blu_types[0][1:12].strip().split(), dtype=np.uint8)
            calibration_settings["lower blue"] = np.array(blu_types[1][1:12].strip().split(), dtype=np.uint8)

            print(calibration_settings)

        with open("./camera_parameters/camera0_intrinsics.dat", "r") as cam0_int:
            lines = cam0_int.readlines()
            cam0_settings[lines[0][:-1]] = np.array([lines[1].strip().split(),lines[2].strip().split(), lines[3].strip().split()], dtype=np.float32)
            cam0_settings[lines[4][:-1]] = np.array(lines[5].strip().split(), dtype=np.float32)

        with open("./camera_parameters/camera1_intrinsics.dat", "r") as cam1_int:
            lines = cam1_int.readlines()
            cam1_settings[lines[0][:-1]] = np.array([lines[1].strip().split(),lines[2].strip().split(), lines[3].strip().split()], dtype=np.float32)
            cam1_settings[lines[4][:-1]] = np.array(lines[5].strip().split(), dtype=np.float32)

        with open("./camera_parameters/camera0_rot_trans.dat", "r") as cam0_rt:
            lines = cam0_rt.readlines()
            cam0_settings[lines[0][:-1]] = np.array([lines[1].strip().split(),lines[2].strip().split(), lines[3].strip().split()], dtype=np.float32)
            cam0_settings[lines[4][:-1]] = np.array([lines[5].strip().split(),lines[6].strip().split(), lines[7].strip().split()], dtype=np.float32)

        with open("./camera_parameters/camera1_rot_trans.dat", "r") as cam1_rt:
            lines = cam1_rt.readlines()
            cam1_settings[lines[0][:-1]] = np.array([lines[1].strip().split(),lines[2].strip().split(), lines[3].strip().split()], dtype=np.float32)
            cam1_settings[lines[4][:-1]] = np.array([lines[5].strip().split(),lines[6].strip().split(), lines[7].strip().split()], dtype=np.float32)

        RT0 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis=-1)
        P0 = cam0_settings["intrinsic:"] @ RT0

        RT1 = np.concatenate([cam1_settings["R:"],cam1_settings["T:"]], axis= -1)
        P1 = cam1_settings["intrinsic:"] @ RT1
    except Exception as e:
        print(e)
        print("Error in loading in settings, calibration may be in order")

    esprite = Esprite(1)


    menu = True
    while menu:
        choice = input("Input Option (type help for menu): ").lower()
        print(choice)
        if choice == "help" or choice == "h":
            print("Options: \n\t 1. calib_cam (or cc) \n\t2. calib_lamp (or cl) \n\t3. send_it (or go)")
        if choice == "calib_cam" or choice == "cc":
            calibrate_cam_settings()
        # ADD CALIB LAMP WHEN READY
        if choice == "calib_lamp" or choice == "cl":
            cl.calib_lamp("camera0", "camera1", calibration_settings["upper blue"], calibration_settings["lower blue"], P0, P1, esprite)
        if choice == "send_it" or choice == "go":
            print("SENDING IT")
            menu = False
        if choice == "quit" or choice == "q":
            quit()
    print("Calibration complete, ready for loop!")
    print("Opening generated parameters, dont rename them!")

    # TODO ADD INTRENSIC AND ROT AND TRANS LATER

    

    print(cam0_settings)
    print(cam1_settings)

    cv.VideoCapture(200).release()
    cv.VideoCapture(204).release()

    cap0 = cv.VideoCapture(200)
    cap1 = cv.VideoCapture(204)

    cv.resizeWindow

    if not cap0.isOpened():
        print("ERR CAP 1")
        exit()
    if not cap1.isOpened():
        print("ERR CAP 2")
        exit()

    # #RT matrix for C1 is identity.
    # RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    # P1 = cmtx1 @ RT1 #projection matrix for C1

    # #RT matrix for C2 is the R and T obtained from stereo calibration.
    # RT2 = np.concatenate([R, T], axis = -1)
    # P2 = cmtx2 @ RT2 #projection matrix for C2

    hue_max = 169
    sat_max = 255
    val_max = 255
    trackbar_uhue = trackbar_var(0)
    trackbar_usat = trackbar_var(0)
    trackbar_uval = trackbar_var(0)
    trackbar_lhue = trackbar_var(0)
    trackbar_lsat = trackbar_var(0)
    trackbar_lval = trackbar_var(0)

    cv.namedWindow(title_window)
    cv.createTrackbar("uhue", title_window , 0, hue_max, trackbar_uhue.change)
    cv.createTrackbar("usat", title_window , 0, sat_max, trackbar_usat.change)
    cv.createTrackbar("uval", title_window , 0, val_max, trackbar_uval.change)

    cv.createTrackbar("lhue", title_window , 0, hue_max, trackbar_lhue.change)
    cv.createTrackbar("lsat", title_window , 0, sat_max, trackbar_lsat.change)
    cv.createTrackbar("lval", title_window , 0, val_max, trackbar_lval.change)

    # upper_blue = calibration_settings["upper blue"]
    # lower_blue = calibration_settings["lower blue"]

    lights.append(esprite)
    esprite.set_intensity(255)
    esprite.go_home()

    while True:

        # Temporary, while i pretend not to bash my own brain out
        lower_blue = np.array([trackbar_lhue.val, trackbar_lsat.val, trackbar_lval.val])
        upper_blue = np.array([trackbar_uhue.val, trackbar_usat.val, trackbar_uval.val])

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print(f"Error.. Cannot recieve frame.. Exiting {ret0} {ret1}")
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
        keypoints1 = detector.detect(mask0)
        pts0 = np.array([key_point.pt for key_point in keypoints])
        pts1 = np.array([key_point.pt for key_point in keypoints1])

        print(f"keypoints detected: {pts0} and {pts1}")

        pt0 = [sum(x)/len(x) for x in zip(*pts0)]
        pt1 = [sum(y)/len(y) for y in zip(*pts1)]
        
        # If light is detected in both keypoints, send update \o/
        if pt0 and pt1:
            print("YES")
            esprite.set_color([255,0,255])
            write_dmx()
        else:
            print("NO")
            esprite.set_color([0,255,255])
            write_dmx()

        print("passed if")
        #print(universe.dmx_frame)
        cv.imshow("mask0", mask0)

        cv.imshow("cam0 hsv", hsv0)
        cv.imshow("cam0 reg", frame0)

        # This is going to be worked on tomorrow
        DLT(P1, P2, pt0, pt1) # ADD STUFF HERE...

        k = cv.waitKey(1)

        if k == 27:
            print("cancelling")
            esprite.go_home()
            esprite.set_intensity(0)
            cap0.release()
            cap1.release()
            write_dmx()
            cv.destroyAllWindows()
            quit()

if __name__ == '__main__':

    main()
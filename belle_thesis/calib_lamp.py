import cv2 as cv
import numpy as np
from calib_cam import DLT, parse_calibration_settings_file
import yaml
import os
from scipy import linalg
from typing import List
from main import write_dmx
import pyenttec as ent
from esprite_profile import Esprite
import keyboard

#This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}
lights = []
universe = ent.DMXConnection("/dev/ttyUSB0")

cam0_settings = {}
cam1_settings = {}

robe = Esprite(1)
lights.append(robe)

class trackbar_var(object):
    def __init__(self, val=0):
        self.val = val
    def change(self, val):
        print("CHANGING")
        self.val = val


# Calib is currently set for the esprite, but it will be generalized down the line
def calib_lamp(camera0_name, camera1_name, upper_blue, lower_blue, P0, P1, esprite):

    title_window = "calib_cam"

    # Surprise Dictionaries that will help us later
    saved_data = {}
    lamp_information = {}

    # Iterate and save information in the `saved_data dict`
    things_to_capture = ["home", "point1", "point2", "point3", "point4"]
    c_index = 0

    print(camera0_name, camera1_name)

    cam0 = 200
    cam1 = 204 # change to be parameters later TODO:

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


    # Prep CV detector
    params = cv.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 100
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = True
    
    detector = cv.SimpleBlobDetector_create(params)

    # restore esprite to home position
    esprite.go_home()
    write_dmx()

    # Everyone's favorite video capture loop
    cap0 = cv.VideoCapture(cam0)
    cap1 = cv.VideoCapture(cam1)

    if not cap0.isOpened():
        print("ERR CAP 1")
        exit()
    if not cap1.isOpened():
        print("ERR CAP 2")
        exit()

    while True:

        lower_blue1 = np.array([trackbar_lhue.val, trackbar_lsat.val, trackbar_lval.val])
        upper_blue1 = np.array([trackbar_uhue.val, trackbar_usat.val, trackbar_uval.val])

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print("Error.. Cannot recieve frame.. Exiting ")
            quit()
        
        hsv0 = cv.cvtColor(frame0, cv.COLOR_BGR2HSV)
        hsv1 = cv.cvtColor(frame1, cv.COLOR_BGR2HSV)

        mask0 = cv.inRange(hsv0, lowerb=lower_blue1, upperb=upper_blue1)
        mask1 = cv.inRange(hsv1, lowerb=lower_blue1, upperb=upper_blue1)

        keypoints = detector.detect(mask0)
        keypoints1 = detector.detect(mask1)
        pts0 = np.array([key_point.pt for key_point in keypoints])
        pts1 = np.array([key_point.pt for key_point in keypoints1])

        pt0 = [sum(x)/len(x) for x in zip(*pts0)]
        pt1 = [sum(y)/len(y) for y in zip(*pts1)]


        cv.imshow(title_window, frame0)
        cv.imshow("mask 0", mask0)

        if pt0 and pt1:
            center_pix = (np.rint(pt0)).astype(int)
            print(f"good track on both {pt0}, {pt1}")
            cv.circle(frame0, center_pix, 50, [255,255,255], 10)

            # TODO: Light trigger

        k = cv.waitKey(1)
        if k == 27:
            #if ESC is pressed at any time, the program will exit.
            quit()
        #if k == 109:
        if keyboard.is_pressed("m"):
            # if `m` is pressed at any time, switch to moving mover
            move_mover(esprite, universe)
        if k == 32:
            #Press spacebar to start data collection
            track_loc = DLT(P0, P1, pt0, pt1)
            saved_data[things_to_capture[c_index]] = track_loc
            c_index += 1

        if c_index >= len(things_to_capture):
            # Calculate something
            print(saved_data)
            return saved_data

        """ TODO: ADD CALIBRATION PROCESS:
            Need to add the following:
                1. A dict that stores all XYZ data from cams & pan/tilt data for lamp
                2. A list that indicates what we are capturing
                3. Ability to pan/tilt with arrow keys
        """

def move_mover(light, universe):
    print("move")

    """have the ability to move the mover through the arrow keys
        up & down control tilt
        left & right control pan
        currently should(?) have ability to take input information
    """
    while True:
        try:
            if keyboard.is_pressed('p'):
                light.set_pan(int(input("Input pan between -270 and 270")))
            if keyboard.is_pressed('t'):
                light.set_pan(int(input("Input tilt between -135 and 135")))
            if keyboard.is_pressed('left'): # left
                light.set_pan(light.pan() + 1)
            if keyboard.is_pressed('up'): # up
                light.set_tilt(light.tilt() + 1)
            if keyboard.is_pressed('down'): # down
                light.set_tilt(light.tilt() - 1)
            if keyboard.is_pressed('right'): # right
                lirhg.set_pan(light.pan() - 1)
            if keyboard.is_pressed("q"): # quit
                return
            write_dmx()
        except: print("something is unhappy")
    return                

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Call with settings filename: "python3 calibrate.py calibration_settings.yaml"')
        quit()

    parse_calibration_settings_file(sys.argv[1])

    calib_lamp('camera0', 'camera1', ... )


    
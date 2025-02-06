from dmx import DMXUniverse, DMXInterface
import cv2 as cv
import numpy
from calib_cam import DLT, parse_calibration_settings_file
import yaml
import os
from scipy import linalg
from typing import List

#This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}

# Calib is currently set for the esprite, but it will be generalized down the line
def calib_lamp(camera0_name, camera1_name, upper_blue, lower_blue, P0, P1, esprite: dmx.DMXLight):

    # Surprise Dictionaries that will help us later
    saved_data = {}
    lamp_information = {}

    # Iterate and save information in the `saved_data dict`
    things_to_capture = ["home", "point1", "point2", "point3", "point4"]

    assert light.pan_tilt # Make sure we have pan_tilt
    cam0 = calibration_settings[camera0_name]
    cam1 = calibration_settings[camera1_name]

    # Prep CV detector
    params = cv.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 100
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = True
    
    detector = cv.SimpleBlobDetector_create(params)

    with DMXInterface('AVRDMX') as interface:

        # Initialize the universe
        universe = DMXUniverse()
        universe.add_light(light)

        # restore esprite to home position
        esprite.go_home()
        interface.set_frame(universe.serialise())
        interface.send_update()

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
    
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()

            if not ret0 or ret1:
                print("Error.. Cannot recieve frame.. Exiting")
                quit()
            
            hsv0 = cv.cvtColor(frame0, cv.COLOR_BGR2HSV)
            hsv1 = cv.cvtColor(frame1, cv.COLOR_BGR2HSV)

            mask0 = cv.inRange(hsv0, lowerb=lower_blue, upperb=upper_blue)
            mask1 = cv.inRange(hsv1, lowerb=lower_blue, upperb=upper_blue)

            keypoints = detector.detect(mask1)
            keypoints1 = detector.detect(mask2)
            pts0 = np.array([key_point.pt for key_point in keypoints])
            pts1 = np.array([key_point.pt for key_point in keypoints1])

            pt0 = [sum(x)/len(x) for x in zip(*pts0)]
            pt1 = [sum(y)/len(y) for y in zip(*pts1)]

            if pt0 and pt1:
                print("good track on both")
                # TODO: Light trigger

            k = cv.waitKey(1)
            if k == 27:
                #if ESC is pressed at any time, the program will exit.
                quit()
            if k == 109:
                # if `m` is pressed at any time, switch to moving mover
                move_mover(interface, esprite, universe)
            if k == 32:
                #Press spacebar to start data collection
                track_loc = DLT(P0, P1, pt0, pt1)

            """ TODO: ADD CALIBRATION PROCESS:
                Need to add the following:
                    1. A dict that stores all XYZ data from cams & pan/tilt data for lamp
                    2. A list that indicates what we are capturing
                    3. Ability to pan/tilt with arrow keys
            """

def move_mover(interface, light, universe):
    print("move")

    """have the ability to move the mover through the arrow keys
        up & down control tilt
        left & right control pan
        currently should(?) have ability to take input information
    """
    k = cv.waitKey(1)
    if k == 'p':
        light.set_pan(int(input("Input pan between -270 and 270")))
    if k == 't':
        light.set_pan(int(input("Input tilt between -135 and 135")))

    interface.set_frame(universe.serialise())
    interface.send_update()
                

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Call with settings filename: "python3 calibrate.py calibration_settings.yaml"')
        quit()

    parse_calibration_settings_file(sys.argv[1])

    calib_lamp('camera0', 'camera1', ... )


    
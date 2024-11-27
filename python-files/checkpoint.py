# Gabriel Howland
# Thesis 470 Checkpoint
# Computer Science & Theatre

# --------- Imported modules
import numpy as np
import argparse
import cv2 as cv
import time
import matplotlib.pyplot as plt
import math
from cv2_enumerate_cameras import enumerate_cameras 
import graphics as g

# --------- Constants
EPSILON = 0.05

# --------- Class Definitions
class trackbar_var(object):
    """
    trackbar_var class to store values for HSV masks
    """
    def __init__(self, val=0):
        self.val = val
    def change(self, val):
        #print("CHANGING")
        self.val = val

# --------- Blob Detect --------- #
def blob_dect(cam0_ID=2, cam1_ID=206):
    """
    Blob Detect: A function that takes multiple camera streams, applies an HSV
    mask, and runs a blob detection algorithm on the masked result. The output
    of this is multiple window pop-ups showcasing various stages.

    Blobs that are detected are shown with drawn circles.

    Blob detect uses the blob detection algorithm provided by OpenCV

    Inputs: cam0_ID - Device ID shown by cv2_enumerate_cameras
            cam1_ID - Device ID shown by cv2_enumerate_cameras

            Inputs default to my local machine's IDs

    Outputs: None
    """
    # Try to get a video captured
    title_window = "mask result"

    # the cap and cap2 objects have methods like read() to extract frames from input streams
    print("opening cap 1")
    cap = cv.VideoCapture(cam0_ID)
    print("opening cap 2")
    cap2 = cv.VideoCapture(cam1_ID)
    print("setup cap, starting loop")

    # This checks if the video capture opened properly. Sometimes a capture won't
    # initialize properly.
    if not cap.isOpened():
        print("ERR CAP 1")
        exit()
    if not cap2.isOpened():
        print("ERR CAP 2")
        exit()

    # Trackbar settings:
    # Initialize trackbars for HSV masking control
    # there are 6 initialized bars, three for the upper limit and three for the lower limit
    hue_max = 169
    sat_max = 255
    val_max = 255
    trackbar_uhue = trackbar_var(0)
    trackbar_usat = trackbar_var(0)
    trackbar_uval = trackbar_var(0)
    trackbar_lhue = trackbar_var(0)
    trackbar_lsat = trackbar_var(0)
    trackbar_lval = trackbar_var(0)

    # this creates the trackbar on the titled window which is a window with
    # the masked result
    cv.namedWindow(title_window)
    cv.createTrackbar("uhue", title_window , 0, hue_max, trackbar_uhue.change)
    cv.createTrackbar("usat", title_window , 0, sat_max, trackbar_usat.change)
    cv.createTrackbar("uval", title_window , 0, val_max, trackbar_uval.change)

    cv.createTrackbar("lhue", title_window , 0, hue_max, trackbar_lhue.change)
    cv.createTrackbar("lsat", title_window , 0, sat_max, trackbar_lsat.change)
    cv.createTrackbar("lval", title_window , 0, val_max, trackbar_lval.change)

    # There is a double loop which ensures the program runs only if both
    # video captures are opened and able to read frames 
    while cap.isOpened():
        while cap2.isOpened():

            # read in frames, ret & ret2 are bools that report if a frame read was
            # successful. frames are numpy matrices of numpy 3-vectors holding GRB Data
            ret2, frame2 = cap2.read()
            ret, frame1 = cap.read()

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # cvtColor converts the image arrays from BGR to HSV which allows for better
            # image masking
            hsv1 = cv.cvtColor(frame1, cv.COLOR_BGR2HSV)
            hsv2 = cv.cvtColor(frame2, cv.COLOR_BGR2HSV)

            # establish HSV ranges defined by the trackbars
            lower_blue = np.array([trackbar_lhue.val, trackbar_lsat.val, trackbar_lval.val])
            upper_blue = np.array([trackbar_uhue.val, trackbar_usat.val, trackbar_uval.val])

            # inRange takes a frame and returns a black & white image where all pixels fall between
            # the two ranges
            mask1 = cv.inRange(hsv1, lowerb=lower_blue, upperb=upper_blue)
            mask2 = cv.inRange(hsv2, lowerb=lower_blue, upperb=upper_blue)

            # this returns a color image that applies the mask to a frame (allows us to see the colored
            # result of the mask)
            masked1 = cv.bitwise_and(hsv1, hsv1, mask=mask1)
            masked2 = cv.bitwise_and(hsv2, hsv2, mask=mask2)

            # create a parameter-type object to define how the blob detection operates
            params = cv.SimpleBlobDetector_Params()

            params.filterByArea = True
            params.minArea = 100
            params.filterByCircularity = False
            params.filterByConvexity = False
            params.filterByInertia = True

            # create a blob detector based off of parameters
            detector = cv.SimpleBlobDetector_create(params)

            # keypoints and keypoints1 are all the detected blobs in a given mask
            keypoints = detector.detect(mask1)
            keypoints1 = detector.detect(mask2)

            # this takes the keypoints and puts them within an array
            points = np.array([key_point.pt for key_point in keypoints])
            points1 = np.array([key_point.pt for key_point in keypoints1])

            # this draws circles around all detected blobs in an image
            for point in points:
                cv.circle(mask1, (int(point[0]),int(point[1])), 63, (255,255,255), 10)
            for point in points1:
                cv.circle(mask2, (int(point[0]),int(point[1])), 63, (255,255,255), 10)

            #frame_w_keypoints = cv.drawKeypoints(mask, keypoints, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # open multiple windows showcasing the multiple states
            
            # original frames
            cv.imshow("frame2", frame2)
            cv.imshow("frame1", frame1)

            # B&W mask
            cv.imshow("mask1", mask1)
            cv.imshow("mask2", mask2)
            
            # masked frames
            cv.imshow(title_window, masked1)
            cv.imshow("masked 2", masked2)

            # if the letter 'q' is pressed at any point, exit the program
            if cv.waitKey(1) == ord('q'):
                break
        break
    # return the HSV values that were last used in the program for easy restart
    print(f"Lower Blue Final Value: {lower_blue}\nUpper Blue Final Value: {upper_blue}")

    # release all the cameras and close all windows
    cap.release()
    cv.destroyAllWindows()
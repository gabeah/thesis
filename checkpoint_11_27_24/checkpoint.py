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

def camera_dots_to_world(px1,py1,
                        px2,py2):
    """
    camera_dots_to_world: Takes the input x,y of a blob in each camera and calculates the real-world
    position of an object. This is used in conjunction with `blob_detect()`.

    Inputs:
    px1 - x-position of a blob detected in camera 1
    py1 - y-position of a blob detected in camera 1
    px2 - x-position ""                 "" camera 2
    py2 - y-position ""                 "" camera 2

    created in collaboration with Jim Fix
    """
    DISTANCE_BETWEEN_CAMERAS = 2.0  # Meters between cameras
    CAMERA_HEIGHT = 1.5             # Height of both cameras
    MAX_X = 1920                    # MAX_X and MAX_Y are the max frame size of an image
    MAX_Y = 1080
    assert(MAX_X >= MAX_Y)
    RATIO = MAX_X / MAX_Y           # Aspect Ratio

    FOV = 80.0 * 2.0 * math.pi / 180.0  # FOV taken from the cameras that are used in this thesis
    FOV_RATIO = math.tan(FOV/2.0)


    # The following lines takes a pixel location and converts it to a metric location, essentially locating the blob on a
    # 1m x 16/9m screen, located 1m away from the cameras. It also reestablishes the blobs to be in relation to where the centerpoint of a camera is located

    # Takes the x_position of the blob in the first camera and converts it from a range of -1.0 to 1.0, then multiplies by
    # both the aspect ratio and the fov ratio to account for warping and the aspect ratio
    x1 = (2.0 * px1 / MAX_X - 1.0) * RATIO / FOV_RATIO 
    # we do not multiply the y_values by the ratios because we want to standardize for all y_values to be between -1.0 and 1.0
    y1 = (2.0 * py1 / MAX_Y - 1.0)

    # Same calculation but for the blob in the second window
    x2 = (2.0 * px2 / MAX_X - 1.0) * RATIO / FOV_RATIO
    y2 = (2.0 * py2 / MAX_Y - 1.0)

    print(x1,y1)

    # Now that we have physical locations, we run frame2vector_cal to find where vectors that intersect at x,y locations on each
    # plane would intersect in 3-space
    x_y_z = frame2vector_cal(DISTANCE_BETWEEN_CAMERAS, x1, x2, y1, y2)
    
    # Finally, we need to do a quick modification to adjust for camera height
    x = x_y_z[0]
    y = x_y_z[1] + CAMERA_HEIGHT
    z = x_y_z[2]

    return np.array([x,y,z])

def frame2vector_cal(D, x1, x2, y1, y2):
    """
    frame2vector_cal: A function that takes the distance between 2 values, and the location in meters on where blobs
    are located on a theoretical screen 1m away.

    Inputs:
    # D is distance between cameras
    # x1 is meters from camera1_center on the 1 meter away screen
    # x2 is meters from camera1_center on the 1 meter away screen but offset by D (which should be camera2_center)
    # y1 "" y SHOULD be equal across both screens
    # y2 "" 

    Outputs:
    (x,y,z) numpy array giving the actual location
    Written in collaboration with Jim Fix
    """
    x = x1 * D / (x1-x2)
    assert(abs(y1-y2) <= EPSILON)
    y = y1 * D / (x1 - x2)
    z = x / x1

    return np.array([x,y,z])

    # By Similar Triangles, x / x1 = z
    #                       (D - x) / z == -x2
    # Solving for z we get
    #              (D - x) / (x / x1) == -x2
    # Which means that
    #               x = (D * x1) / (x1 - x2)
    # as we wrote.

    # This also means that
    #               z = D / (x1 - x2)
    # as written

    # Note also that y / y1 = z
    # That means that
    #               y = z * y1


    # Screen is 1m away, infinite projection screen
    

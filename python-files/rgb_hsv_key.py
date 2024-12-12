import numpy as np
import argparse
import cv2 as cv
import time
import matplotlib.pyplot as plt
import math
import graphics as g

def main(args):
    
    picInput = "/home/gabeh/thesis/default_screenshot_14.10.2024.png"

    hsv_rgb_key(picInput)


class trackbar_var(object):
    def __init__(self, val=0):
        self.val = val
    def change(self, val):
        #print("CHANGING")
        self.val = val

def hsv_rgb_key(pic):
    # Try to get a video captured
    hsv_window = "hsv mask"
    rgb_window = "rgb window"
    #fstream = in_stream
    print("opening cap 1")
    cap = cv.imread(pic, cv.IMREAD_UNCHANGED)
    print("opening cap 2")
    cap2 = cv.imread(pic, cv.IMREAD_UNCHANGED)
    print("setup cap, starting loop")


# Trackbar settings:
    hue_max = 169
    sat_max = 255
    val_max = 255

    RGB_MAX = 255
    trackbar_ur = trackbar_var(0)
    trackbar_ug = trackbar_var(0)
    trackbar_ub = trackbar_var(0)
    trackbar_lr = trackbar_var(0)
    trackbar_lg = trackbar_var(0)
    trackbar_lb = trackbar_var(0)

    trackbar_uhue = trackbar_var(0)
    trackbar_usat = trackbar_var(0)
    trackbar_uval = trackbar_var(0)
    trackbar_lhue = trackbar_var(0)
    trackbar_lsat = trackbar_var(0)
    trackbar_lval = trackbar_var(0)

    cv.namedWindow(hsv_window)
    cv.namedWindow(rgb_window)
    cv.createTrackbar("uhue", hsv_window , 0, hue_max, trackbar_uhue.change)
    cv.createTrackbar("usat", hsv_window , 0, sat_max, trackbar_usat.change)
    cv.createTrackbar("uval", hsv_window , 0, val_max, trackbar_uval.change)

    cv.createTrackbar("lhue", hsv_window , 0, hue_max, trackbar_lhue.change)
    cv.createTrackbar("lsat", hsv_window , 0, sat_max, trackbar_lsat.change)
    cv.createTrackbar("lval", hsv_window , 0, val_max, trackbar_lval.change)

    cv.createTrackbar("ured", rgb_window , 0, RGB_MAX, trackbar_ur.change)
    cv.createTrackbar("ugrn", rgb_window , 0, RGB_MAX, trackbar_ug.change)
    cv.createTrackbar("ublu", rgb_window , 0, RGB_MAX, trackbar_ub.change)

    cv.createTrackbar("lred", rgb_window , 0, RGB_MAX, trackbar_lr.change)
    cv.createTrackbar("lgrn", rgb_window , 0, RGB_MAX, trackbar_lg.change)
    cv.createTrackbar("lblu", rgb_window , 0, RGB_MAX, trackbar_lb.change)

    while True:


        #print(ret)

        # frame = cv.flip(frame, 0)
        # frame2 = cv.flip(frame2, 0)
        # if frame is read correctly ret is True
        print(cap)
        print(cap2)

        hsv = cv.cvtColor(cap, cv.COLOR_RGB2HSV)
        #rgb = cv.cvtColor(cap2, cv.COLOR_BGR2RGB)
        rgb = cap2
        lower_blue = np.array([trackbar_lhue.val, trackbar_lsat.val, trackbar_lval.val])
        upper_blue = np.array([trackbar_uhue.val, trackbar_usat.val, trackbar_uval.val])

        lower_rgb = np.array([trackbar_lr.val, trackbar_lg.val, trackbar_lb.val])
        upper_rgb = np.array([trackbar_ur.val, trackbar_ug.val, trackbar_ub.val])

        print(lower_rgb.dtype)

        #rgb = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        mask1 = cv.inRange(hsv, lowerb=lower_blue, upperb=upper_blue)
        mask2 = cv.inRange(rgb, lowerb=lower_rgb, upperb=upper_rgb)

        masked1 = cv.bitwise_and(cap, hsv, mask=mask1)
        masked2 = cv.bitwise_and(rgb, rgb, mask=mask2)

        
        cv.imshow("input rgb", cap2)
        cv.imshow("input hsv", cap)

        cv.imshow("mask hsv", mask1)
        cv.imshow("mask rgb", mask2)

        cv.imshow(hsv_window, masked1)
        cv.imshow(rgb_window, masked2)

        if cv.waitKey(1) == ord('q'):
            break
    
    cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='?', help="input images")
    args = parser.parse_args()
    main(args)

import numpy as np
import argparse
import cv2 as cv
import time
import matplotlib.pyplot as plt

class trackbar_var(object):
    def __init__(self, val=0):
        self.val = val
    def change(self, val):
        print("CHANGING")
        self.val = val

def main(args):
    
    print(args.input)

    print(f"testing various streams")
    img_mask(args)
    # for stream in args.input:
    #     print(f"testing stream through video {stream}")
    #     mask(stream)

def img_mask(args):
    cap = cv.imread(cv.samples.findFile(args.input[0]))

    title_window = 'HSV Mask Test'

    if cap is None:
        print('Could not open or find the image: ', args.input1)
        exit(0)

    # Create the trackbar
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

    while True:

        hsv_cap = cv.cvtColor(cap, cv.COLOR_RGB2HSV)

        upper_msk = np.array([trackbar_uhue.val,trackbar_usat.val,trackbar_uval.val])
        lower_msk = np.array([trackbar_lhue.val,trackbar_lsat.val,trackbar_lval.val])

        hsv_mask = cv.inRange(hsv_cap, lower_msk, upper_msk)

        hsv_masked = cv.bitwise_and(hsv_cap, hsv_cap, mask=hsv_mask)

        cv.imshow(title_window, hsv_masked)
        cv.imshow("mask", hsv_mask)
        cv.imshow("default", hsv_cap)

        key = cv.waitKey(1)
        if key == ord('q'):
                break
        elif key == ord('p'):
            print("PAUSE")
            cv.waitKey(0)
    cap.release()
    cv.destroyAllWindows()
        

# Function for masking
def mask(fstream):
    # Try to get a video captured
    cap = cv.VideoCapture(fstream)
    print("setup cap, starting loop")

    if not cap.isOpened():
        print("ERR")
        #exit()

    while cap.isOpened():

        # This is in BGR color!! Note this!!
        ret, frame = cap.read()
        
        if not ret:
            break
        frame = cv.flip(frame, 0)
        print(frame)

        cv.imshow("default", frame)

        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        lower_bl_hsv = np.array([110,0,50])
        upper_bl_hsv = np.array([130, 255, 255])

        mask_hsv = cv.inRange(frame_hsv, lower_bl_hsv, upper_bl_hsv)
        masked_hsv = cv.bitwise_and(frame, frame, mask=mask_hsv)

        upper_bl = np.array([255,255,255])
        lower_bl = np.array([200,100,100])

        mask_bgr = cv.inRange(frame, lower_bl, upper_bl)
        masked_bgr = cv.bitwise_and(frame, frame, mask=mask_bgr)

        cv.imshow("mask_bgr", mask_bgr)
        cv.imshow("masked_hsv", masked_bgr)

        cv.imshow("mask_hsv", mask_hsv)
        cv.imshow("masked_hsv", masked_hsv)

        key = cv.waitKey(1)
        if key == ord('q'):
                break
        elif key == ord('p'):
            print("PAUSE")
            cv.waitKey(0)
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', help="input video/stream(s)")
    args = parser.parse_args()
    main(args)
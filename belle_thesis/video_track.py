import numpy as np
import argparse
import cv2 as cv
import time
import matplotlib.pyplot as plt
import math

def blob_dect(in_stream=None):
    # Try to get a video captured
    title_window = "mask result"
    #fstream = in_stream
    print("opening cap 1")
    cap = cv.VideoCapture(2)
    print("opening cap 2")
    cap2 = cv.VideoCapture(206)
    print("setup cap, starting loop")

    if not cap.isOpened():
        print("ERR CAP 1")
        exit()
    if not cap2.isOpened():
        print("ERR CAP 2")
        exit()

# Trackbar settings:
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

    while cap.isOpened():
        while cap2.isOpened():

            ret2, frame2 = cap2.read()
            ret, frame1 = cap.read()
            #print(ret)

            # frame = cv.flip(frame, 0)
            # frame2 = cv.flip(frame2, 0)
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            hsv1 = cv.cvtColor(frame1, cv.COLOR_BGR2HSV)
            hsv2 = cv.cvtColor(frame2, cv.COLOR_BGR2HSV)

            lower_blue = np.array([trackbar_lhue.val, trackbar_lsat.val, trackbar_lval.val])
            upper_blue = np.array([trackbar_uhue.val, trackbar_usat.val, trackbar_uval.val])

            #rgb = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            mask1 = cv.inRange(hsv1, lowerb=lower_blue, upperb=upper_blue)
            mask2 = cv.inRange(hsv2, lowerb=lower_blue, upperb=upper_blue)

            masked1 = cv.bitwise_and(hsv1, hsv1, mask=mask1)
            masked2 = cv.bitwise_and(hsv2, hsv2, mask=mask2)

            params = cv.SimpleBlobDetector_Params()

            params.filterByArea = True
            params.minArea = 100
            params.filterByCircularity = False
            params.filterByConvexity = False
            params.filterByInertia = True

            detector = cv.SimpleBlobDetector_create(params)

            keypoints = detector.detect(mask1)
            keypoints1 = detector.detect(mask2)
            points = np.array([key_point.pt for key_point in keypoints])
            points1 = np.array([key_point.pt for key_point in keypoints1])

            for point in points:
                cv.circle(mask1, (int(point[0]),int(point[1])), 63, (255,255,255), 10)
            for point in points1:
                cv.circle(mask2, (int(point[0]),int(point[1])), 63, (255,255,255), 10)

            #frame_w_keypoints = cv.drawKeypoints(mask, keypoints, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv.imshow("frame2", frame2)
            cv.imshow("frame1", frame1)
            cv.imshow("mask1", mask1)
            cv.imshow("mask2", mask2)
            #cv.imshow("Blob Detection", frame_w_keypoints)
            #cv.imshow("rgb", rgb)
            #cv.imshow("hsv", hsv)
            cv.imshow(title_window, masked1)
            cv.imshow("masked 2", masked2)

            if cv.waitKey(1) == ord('q'):
                break
        break
    print(f"Lower Blue Final Value: {lower_blue}\nUpper Blue Final Value: {upper_blue}")

    cap.release()
    cv.destroyAllWindows()

def main(args):
    blob_dect(args.input)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='?', help="input video/stream(s)")
    args = parser.parse_args()
    main(args)
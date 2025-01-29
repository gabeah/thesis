import numpy as np
import argparse
import cv2 as cv
import time
import matplotlib.pyplot as plt
import math
from cv2_enumerate_cameras import enumerate_cameras 
import graphics as g

class trackbar_var(object):
    def __init__(self, val=0):
        self.val = val
    def change(self, val):
        #print("CHANGING")
        self.val = val

def main(args):
    
    for camera_info in enumerate_cameras():
        print(f'{camera_info.index}: {camera_info.name}')
    print(args.input)
    lb = np.array([104,77,108])
    ub = np.array([124,255,255])
    blob_dect(lower_blu=lb, upper_blu=ub)
    #blob_dect()

def blob_dect(upper_blu=None, lower_blu=None):
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

    if upper_blu.any()==None or lower_blu.any()==None:
        print("No blue, making bars")
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

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    x_locs = [0]
    y_locs = [0]
    z_locs = [0]

    ax.plot(0,0,"bo")
    plt.show(block=False)



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

            if lower_blu.any()==None:
                lower_blue = np.array([trackbar_lhue.val, trackbar_lsat.val, trackbar_lval.val])
            else:
                lower_blue = lower_blu
            if upper_blu.any()==None:
                upper_blue = np.array([trackbar_uhue.val, trackbar_usat.val, trackbar_uval.val])
            else:
                upper_blue = upper_blu

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
 # pull the first point of second cam and see what happens

            for point in points:
                cv.circle(mask1, (int(point[0]),int(point[1])), 63, (255,255,255), 10)
            for point in points1:
                cv.circle(mask2, (int(point[0]),int(point[1])), 63, (255,255,255), 10)

            #frame_w_keypoints = cv.drawKeypoints(mask, keypoints, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv.imshow("frame2", frame2)
            cv.imshow("frame1", frame1)
            cv.imshow("mask1", mask1)
            cv.imshow("mask2", mask2)
            
            cv.imshow(title_window, masked1)
            cv.imshow("masked 2", masked2)

            try:
                p1 = points[0]
                p2 = points1[0]
                intersect = camera_dots_to_world(p1[0], p1[1],p2[0],p2[1])
                print("got through intersect")
                print(intersect)
                print("clearing")
                ax.plot(intersect[0],intersect[2], "r+")
                # x_locs.append(intersect[0])
                # y_locs.append(intersect[1])
                # z_locs.append(intersect[2])
                fig.canvas.draw()
                print(f"updated?")
            except:
                print("bad")
            if cv.waitKey(1) == ord('q'):
                break
        break
    print(f"Lower Blue Final Value: {lower_blue}\nUpper Blue Final Value: {upper_blue}")

    cap.release()
    cv.destroyAllWindows()

def camera_dots_to_world(px1,py1,
                        px2,py2):
    DISTANCE_BETWEEN_CAMERAS = 2.0  # Meters between cameras
    CAMERA_HEIGHT = 1.5             # Height of both cameras
    MAX_X = 1920
    MAX_Y = 1080
    assert(MAX_X >= MAX_Y)
    RATIO = MAX_X / MAX_Y

    FOV = 80.0 * math.pi / 180.0
    FOV_RATIO = math.tan(FOV/2.0)

    x1 = (2.0 * px1 / MAX_X - 1.0) * RATIO / FOV_RATIO
    y1 = (2.0 * py1 / MAX_Y - 1.0)
    x2 = (2.0 * px2 / MAX_X - 1.0) * RATIO / FOV_RATIO
    y2 = (2.0 * py2 / MAX_Y - 1.0)

    print(f"{x1},{y1} starting dist cal")

    x_y_z = frame2vector_cal(DISTANCE_BETWEEN_CAMERAS, x1, x2, y1, y2)
    x = x_y_z[0]
    y = x_y_z[1] + CAMERA_HEIGHT
    z = x_y_z[2]

    return np.array([x,y,z])

def frame2vector_cal(D, x1, x2, y1, y2):

    # D is distance between cameras
    # x1 is meters from center on the 1 meter away screen
    # x2 """" but offset by D
    # y1 "" y SHOULD be equal across both screens
    # y2 "" 
    print("distcal")
    x = x1 * D / (x1-x2)
    print("x worked")
    #assert(abs(y1-y2) <= EPSILON, "no similar y")
    print("assert worked")
    y = y1 * D / (x1 - x2)
    z = x / x1

    return np.array([x,y,z])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='?', help="input video/stream(s)")
    args = parser.parse_args()
    main(args)
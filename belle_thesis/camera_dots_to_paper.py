import numpy as np
import argparse
import cv2 as cv
import time
import matplotlib.pyplot as plt
import math
import yaml
import matplotlib
from matplotlib.animation import FuncAnimation
from calib_cam import DLT
import csv
matplotlib.use("QtAgg")

def blob_dect(camera_data):
    
    print(camera_data)


    print("opening cap 1")
    cap = cv.VideoCapture(camera_data["camera0"]["cam_id"])
    print("opening cap 2")
    cap2 = cv.VideoCapture(camera_data["camera1"]["cam_id"])
    print("setup cap, starting loop")

    if not cap.isOpened():
        print("ERR CAP 1")
        exit()
    if not cap2.isOpened():
        print("ERR CAP 2")
        exit()

    print("Displaying input data")
    print(camera_data)

    T0 = np.array(camera_data["camera0"]["trans_mat"])      # Translation Matrix to self
    T1 = np.array(camera_data["camera1"]["trans_mat"])      # Translation Matrix to Cam0
    R0 = np.array(camera_data["camera0"]["rot_mat"])        # Rotation Matrix to self
    R1 = np.array(camera_data["camera1"]["rot_mat"])        # Rotation Matrix to Cam0
    mtx0 = np.array(camera_data["camera0"]["intr_mat"])   # Intrinsic Matrix
    mtx1 = np.array(camera_data["camera1"]["intr_mat"])   # Intrinsic Matrix
    C0_dist = np.array(camera_data["camera0"]["dist_mat"])  # Distortion Matrix
    C1_dist = np.array(camera_data["camera1"]["dist_mat"])  # Distortion Matrix
    lower_blue = np.array(camera_data["lower_blue"])
    upper_blue = np.array(camera_data["upper_blue"])

    #RT matrix for C1 is identity.
    RT0 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P0 = mtx1 @ RT0 #projection matrix for C1
    
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT1 = np.concatenate([R1, T1], axis = -1)
    P1 = mtx1 @ RT1 #projection matrix for C2

    params = cv.SimpleBlobDetector_Params()

    params.filterByArea = False
    #params.minArea = 100
    params.filterByCircularity = False
    params.filterByConvexity = True
    params.filterByInertia = True

    detector = cv.SimpleBlobDetector_create(params)

    x = np.array([0.0])
    y = np.array([0.0])

    fig = plt.figure()

    ax = fig.add_subplot()
    plt.ylim(-20,20)
    plt.xlim(-20,20)

    led_pos = ax.plot([], [], c="b")[0]

    bg_ax = fig.canvas.copy_from_bbox(ax.bbox)

    fig.canvas.draw()

    buf = fig.canvas.buffer_rgba()
    plot = np.asarray(buf)
    plot = cv.cvtColor(plot, cv.COLOR_RGB2BGR)

    while cap.isOpened():
        while cap2.isOpened():

            ret2, frame2 = cap2.read()
            ret, frame1 = cap.read()
            
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            """Original Detector, looking to try something faster/different"""
            hsv1 = cv.cvtColor(frame1, cv.COLOR_BGR2HSV)
            hsv2 = cv.cvtColor(frame2, cv.COLOR_BGR2HSV)

            mask1 = cv.inRange(hsv1, lowerb=lower_blue, upperb=upper_blue)
            mask2 = cv.inRange(hsv2, lowerb=lower_blue, upperb=upper_blue)

            masked1 = cv.bitwise_and(hsv1, hsv1, mask=mask1)
            masked2 = cv.bitwise_and(hsv2, hsv2, mask=mask2)

            # keypoints = detector.detect(mask1)
            # keypoints1 = detector.detect(mask2)
            # pts0 = np.array([key_point.pt for key_point in keypoints])
            # pts1 = np.array([key_point.pt for key_point in keypoints1])

            # for point in pts0:
            #     cv.circle(mask1, (int(point[0]),int(point[1])), 63, (255,255,255), 10)
            # for point in pts1:
            #     cv.circle(mask2, (int(point[0]),int(point[1])), 63, (255,255,255), 10)

            # pt0 = [sum(x)/len(x) for x in zip(*pts0)]
            # pt1 = [sum(y)/len(y) for y in zip(*pts1)]

            """Testing using minmaxloc (cause why not)"""

            mml_pass0 = cv.cvtColor(masked1, cv.COLOR_BGR2GRAY)
            mml_pass1 = cv.cvtColor(masked2, cv.COLOR_BGR2GRAY)

            min_val0, max_val0, min_loc0, pt0 = cv.minMaxLoc(mml_pass1)
            min_val1, max_val1, min_loc1, pt1 = cv.minMaxLoc(mml_pass1)


            if len(pt0) > 0 and len(pt1) > 0:
                dlt_out = DLT(P0, P1, pt0, pt1)
                print(f"DLT result: {dlt_out[0]} {dlt_out[1]}")
                x = np.append(x, dlt_out[0])
                y = np.append(y, dlt_out[1])

            else:
                print("not enough")

            led_pos.set_data(x,y)

            fig.canvas.restore_region(bg_ax)
            ax.draw_artist(led_pos)
            fig.canvas.blit(ax.bbox)
            plt.pause(0.05)

            # We can assume the CV is functioning normally
            cv.imshow("frame2", frame2)
            cv.imshow("frame1", frame1)
            cv.imshow("mask1", mask1)
            cv.imshow("mask2", mask2)
            cv.imshow("Plotted", plot)

            if cv.waitKey(1) == ord('q'):
                print("Printing Values:")
                with open('saved_locations', 'w') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    wr.writerow(x)
                    wr.writerow(y)
                break
        break

    print(x)
    print(y)
    cap.release()
    cv.destroyAllWindows()

def main(args):
    if args.calib_results:
        print("reading data calibrated already")
        with open(args.calib_results, 'r') as settings_yaml:
            try:
                # Converts yaml document to python object
                d=yaml.safe_load(settings_yaml)

                # Printing dictionary
                print(d)
                blob_dect(d)
            except yaml.YAMLError as e:
                print(e)
                
    else:
        print("please calibrate and give file")
        return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='?', help="input video/stream(s)")
    parser.add_argument('--calib_results', help="input calibration file")
    args = parser.parse_args()
    main(args)


def read_cam_params(filename):
    print(filename)
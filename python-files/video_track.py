# Gabe Howland
# Video tracking test
import numpy as np
import argparse
import cv2 as cv
import time
import matplotlib.pyplot as plt


def main(args):
    
    print(args.input)

    print(f"testing various streams")
    for stream in args.input:
        print(f"testing stream through video {stream}")
        #capVid(stream)
        blob_dect(stream)

def blob_dect(in_stream):
    # Try to get a video captured
    fstream = in_stream
    cap = cv.VideoCapture(fstream)
    print("setup cap, starting loop")

    if not cap.isOpened():
        print("ERR")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        print(ret)

        frame = cv.flip(frame, 0)
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)


        # Initializing parameter setting using cv.SimpleBlobDetector function
        params = cv.SimpleBlobDetector_Params()
        
        # Filter by area (value for area here defines the pixel value)
        params.filterByArea = True
        params.minArea = 100
        
        # Filter by circularity
        params.filterByCircularity = True
        params.minCircularity = 0.75
        
        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = 0.2
            
        # Filter by inertia ratio
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        
        # Creating a blob detector using the defined parameters
        detector = cv.SimpleBlobDetector_create(params)
            
        # Detecting the blobs in the image
        keypoints = detector.detect(rgb)
        
        # Drawing the blobs that have been filtered with blue on the image
        blank = np.zeros((1, 1))
        blobs = cv.drawKeypoints(rgb, keypoints, blank, (0, 0, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        cv.imshow("original", rgb)
        cv.imshow("blobs", blobs)
        
        # Setting the grid size
        plt.figure(figsize=(20,20))
        
        # Displaying the image
        plt.subplot(121)
        plt.title('Original')
        plt.imshow(rgb, cmap='gray')
        
        plt.subplot(122)
        plt.title('Blobs')
        plt.imshow(blobs)
        
        plt.show()


def capVid(in_stream):
    # Try to get a video captured
    fstream = in_stream
    cap = cv.VideoCapture(fstream)
    print("setup cap, starting loop")

    if not cap.isOpened():
        print("ERR")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()

        frame = cv.flip(frame, 0)
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        

        # Seeing if I can do pixel analysis

        # Example of applying Image Thresholding on a grayscale picture.
        threshold = 140
        assignvalue = 255 # Value to assign the pixel if the threshold is met
        threshold_method = cv.THRESH_BINARY

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        lower_blue = np.array([0,0,255])
        upper_blue = np.array([255,255,255])

        mask = cv.inRange(hsv, lower_blue, upper_blue)
        _, result = cv.threshold(frame,threshold,assignvalue,threshold_method)

        result2 = cv.bitwise_and(result, result, mask = mask)
        mask2 = np.array(mask)
        result_blur = cv.GaussianBlur(result2, (7,7), 10)

        img_array = np.array(result2)

        (minVal, maxVal, minLoc,  maxLoc) = cv.minMaxLoc(mask2)

        # Stuck trying to get a location of the bright-spot
        #minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(img_array)

        cv.circle(result2, maxLoc, 100, (255,234,132))
        cv.circle(mask2, maxLoc, 100, (255,234,132))
        cv.circle(result_blur, maxLoc, 100, (255,234,132))


        cv.imshow('frame', frame)
        cv.imshow('mask', mask2)
        cv.imshow('result', result2)
        cv.imshow('blur', result_blur)
        time.sleep(0.1)
        #gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        print(frame)
        print(type(frame))

        # Display the results
        #cv.imshow('frame',result)

        if cv.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', help="input video/stream(s)")
    args = parser.parse_args()
    main(args)
# Gabe Howland
# Video tracking test
import numpy as np
import argparse
import cv2 as cv
import time

def main(args):
    
    print(args.input)

    print(f"testing various streams")
    for stream in args.input:
        print(f"testing stream through video {stream}")
        capVid(stream)

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

        img_array = np.array(result2)

        (minVal, maxVal, minLoc,  maxLoc) = cv.minMaxLoc(mask2)

        # Stuck trying to get a location of the bright-spot
        #minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(img_array)

        cv.circle(result2, maxLoc, 100, (255,234,132))
        cv.circle(mask2, maxLoc, 100, (255,234,132))


        cv.imshow('frame', frame)
        cv.imshow('mask', mask2)
        cv.imshow('result', result2)
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
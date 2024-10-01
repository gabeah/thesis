# Gabe Howland
# Video tracking test
import numpy as np
import cv2 as cv
import time

# Try to get a video captured
cap = cv.VideoCapture('../Videos/IMG_3958.mp4')
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

    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('result', result2)
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    
    #(minVal, maxVal, minLoc,  maxLoc) = cv.minMaxLoc(frame)

    print(frame)
    print(type(frame))

    # Display the results
    #cv.imshow('frame',result)

    if cv.waitKey(1) == ord('q'):
        break
 
cap.release()
cv.destroyAllWindows()
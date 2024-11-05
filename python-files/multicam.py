import numpy as np
import argparse
import cv2 as cv
import time
import matplotlib.pyplot as plt
import threading

class camThread(threading.Thread):

    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    
    def run(self):
        print(f"Starting thread: {self.previewName}")
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):

    cv.namedWindow(previewName)
    cam = cv.VideoCapture(camID)
    if cam.isOpened():
        read, frame = cam.read()
    else:
        read = False

    while read:
        cv.imshow(previewName, frame)
        read, frame = cam.read()
        key = cv.waitKey(0)
        if key == ord('q'):
            break
    cv.destroyAllWindows(previewName)

def main(args):
    print("hello world")
    thread0 = camThread("laptop-cam", 201)
    thread1 = camThread("webcam 1", 2)
    thread0.start()
    thread1.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', help="input video/stream(s)")
    args = parser.parse_args()
    main(args)

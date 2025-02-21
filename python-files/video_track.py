# Gabe Howland
# Video tracking test
import numpy as np
import argparse
import cv2 as cv
import time
import matplotlib.pyplot as plt
import math
from cv2_enumerate_cameras import enumerate_cameras 
import graphics as g

EPSILON = 0.05

class Scene(object):

    def __init__(self, camera1, camera2):
        self.cam1 = camera1
        self.cam2 = camera2

    def frame2vec(self, pixel_loc: [int, int]) -> [float, float, float]:
        """
        frame2vec:  take a pixel location and return a vector coming from the camera object
                    and intersecting the image plane at the pixel location
        
        params:
        @self = camera_object,
        @pixel_loc = (x,y) location of the pixel
        """

        theta = self.FOV / 2
        towards = np.array([0,0,self.FOCAL_LENGTH])
        hframe_mm = 2 * (np.tan(theta) * self.FOCAL_LENGTH)
        pixel_size = hframe_mm / self.RESOLUTION[0]
        vframe_mm = pixel_size * self.RESOLUTION[1]


class Camera(object):

    placeholder_val = None

    def __init__(self, 
                    id: int,            # camera id as assigned from enumerate_cameras()
                    foc_len: float,     # float of focal_length in mm
                    fov: int,           # int value of the horizontal FOV
                    pos: (int, int, int),    # X,Y,Z position of the camera in mm
                    resolution: (int, int) = (1920,1080),   # Set camera resolution
                    origin: bool = False):                  # Is this camera the origin?
        self.ID = id
        self.POS = np.array(pos)
        self.ORIGIN = origin
        self.FOV = fov
        self.FOCAL_LENGTH = foc_len
        self.RESOLUTION = resolution
        
        # Create information for projection frame, to be calculated later
        self.center = Camera.placeholder_val
        self.right  = Camera.placeholder_val
        self.up     = Camera.placeholder_val
        self.into   = Camera.placeholder_val
    
    def create_scene_frame(self):

        vRange = np.array[-1,1]
        hRange = np.array[-16/9, 16/9]
        center_pixel = [self.RESOLUTION[0]/2,self.RESOLUTION[1]/2]
        

        # self.center = self.POS * (1/self.POS)
        # self.into = self.center

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

    print(x1,y1)

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
    
def graphics_loop():
    win = g.GraphWin("Graphic Demo", 1500, 1500)

    pixel_to_meter_ratio = 0.00131109375 # The size of a pixel in M

    camera_pix_dist = 1525

    # Back of the napkin calculations:
    # A screen placed a meter away the size of 720x1280 is about 1.67m (jim as +/- 1.49m)
    # cameras that are placed 2m apart should be ~1525px apart

    MAX_Y_RES = 720
    MAX_X_RES = MAX_Y_RES * 16//9

    cam1_window = "left window"
    cam2_window = "right side"

    cv_bg = np.ones((MAX_Y_RES,int(MAX_X_RES),3), dtype=np.uint8)
    cv_mult = np.array([255,255,255], dtype=np.uint8)

    cv_blk = cv_mult * cv_bg

    cv.namedWindow(cam1_window, cv.WINDOW_AUTOSIZE)
    cv.namedWindow(cam2_window, cv.WINDOW_AUTOSIZE)

    lcx = trackbar_var(int(MAX_X_RES//2)-1)
    rcx = trackbar_var(int(MAX_X_RES//2)+1)
    cv.createTrackbar("LCX", cam1_window, 0, MAX_X_RES, lcx.change)
    cv.createTrackbar("RCX", cam2_window, 0, MAX_X_RES, rcx.change)

    while True:

        cv_mult = np.array([0,0,0], dtype=np.uint8)
        left = cv_mult * cv_bg
        right = cv_mult * cv_bg

        # Visualize the circles
        cv.circle(left, (lcx.val, MAX_Y_RES//2), 30, (255,98,115), 15)
        cv.circle(right, (rcx.val, MAX_Y_RES//2), 30, (100,85,255), 15)

        cv.imshow(cam1_window, left)
        cv.imshow(cam2_window, right)

        cam1_loc = (0,1500)
        cam2_loc = (cam1_loc[0] + camera_pix_dist,1500)

        cam1 = g.Circle(g.Point(*cam1_loc), 100)
        cam2 = g.Circle(g.Point(*cam2_loc), 100)
        cam1.draw(win)
        cam2.draw(win)

        print(f"Looking for intersection between {lcx.val} and {rcx.val}")

        intersect_loc = frame2vector_cal(2,lcx.val, rcx.val, 540, 540)
        print(intersect_loc)
        
        intersect = g.Circle(g.Point(intersect_loc[0]/pixel_to_meter_ratio, intersect_loc[2]/pixel_to_meter_ratio), 50)

        cam1_proj = g.Vec2D(g.Point(*cam1_loc), g.Point(lcx.val, 540))
        cam2_proj = g.Vec2D(g.Point(*cam2_loc), g.Point(rcx.val, 540))
        cam2_proj.undraw()
        cam2_proj.draw(win)
        cam1_proj.undraw()
        cam1_proj.draw(win)

        if cv.waitKey(1) == ord('q'):
                break        

    #win.getMouse() # pause for click in window

    cv.destroyAllWindows()

    win.close()


def main(args):
    
    for camera_info in enumerate_cameras():
        print(f'{camera_info.index}: {camera_info.name}')
    print(args.input)

    #color_test()

    
    #graphics_loop()

    print(f"testing various streams")
    
    if args.input:
        for stream in args.input:
            print(f"testing stream through video {stream}")
        
    # # capVid(stream)
            blob_dect(stream)
    else:
        blob_dect()


class trackbar_var(object):
    def __init__(self, val=0):
        self.val = val
    def change(self, val):
        #print("CHANGING")
        self.val = val

def blob_dect(in_stream=None):
    # Try to get a video captured
    title_window = "mask result"
    #fstream = in_stream
    print("opening cap 1")
    cap = cv.VideoCapture(200)
    print("opening cap 2")
    cap2 = cv.VideoCapture(204)
    print("setup cap, starting loop")

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    cap2.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap2.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

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

def triangulate(cam1, cam2, distance):
    # Add some stuff for the cameras, cam1 and cam2 are cam.type() objects
    # All measurements will be in metric
    # nexigo cam focal_length = 4.35mm
    print(f"triangulating with {cam1.id} and {cam2.id}")
    if cam1.origin == True:
        assert cam1.pos == [0,0], "Not origin value"
        origin = cam1
    elif cam2.origin == True:
        assert cam2.pos == [0,0], "Not origin value"
        origin = cam2
    else: assert False, "Mark a camera as an origin!"

    assert cam1.focal_length == cam2.focal_length, "mismatch in focal_length"

    D = distance






def capVid(in_stream):
    # Try to get a video captured
    fstream = in_stream
    print(fstream)
    cap = cv.VideoCapture(0)
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
    parser.add_argument('--input', nargs='?', help="input video/stream(s)")
    args = parser.parse_args()
    main(args)
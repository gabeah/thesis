def graphics_loop():
    win = g.GraphWin("Graphic Demo", 1500, 1500, autoflush=True)

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

    cv_mult = np.array([0,0,0], dtype=np.uint8)
    left = cv_mult * cv_bg
    right = cv_mult * cv_bg

    cam1_loc = (0,1500)
    cam2_loc = (cam1_loc[0] + camera_pix_dist,1500)

    cam1 = g.Circle(g.Point(*cam1_loc), 100)
    cam2 = g.Circle(g.Point(*cam2_loc), 100)
    cam1.draw(win)
    cam2.draw(win)

    cam1_proj = g.Vec2D(g.Point(*cam1_loc), g.Point(lcx.val, 540))
    cam2_proj = g.Vec2D(g.Point(*cam2_loc), g.Point(rcx.val, 540))
    cam2_proj.draw(win)
    cam1_proj.draw(win)

    while True:

        print(f"Looking for intersection between {lcx.val} and {rcx.val}")
        intersect_loc = frame2vector_cal(2,lcx.val, rcx.val, 540, 540)
        print(intersect_loc)
        
        intersect = g.Circle(g.Point(intersect_loc[0]/pixel_to_meter_ratio, intersect_loc[2]/pixel_to_meter_ratio), 50)

        # Visualize the circles
        cv.circle(left, (lcx.val, MAX_Y_RES//2), 30, (255,98,115), 15)
        cv.circle(right, (rcx.val, MAX_Y_RES//2), 30, (100,85,255), 15)

        cv.imshow(cam1_window, left)
        cv.imshow(cam2_window, right)

        dx1 = cam1_proj.p1.x - lcx.val
        dx2 = cam2_proj.p2.x - rcx.val

        cam1_proj.move(dx1, 0)
        cam2_proj.move(dx2, 0)
        win.update()

        if cv.waitKey(1) == ord('q'):
                break        

    #win.getMouse() # pause for click in window

    cv.destroyAllWindows()

    win.close()

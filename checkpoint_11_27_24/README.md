# Checkpoint: Thesis 470
## Gabriel Howland

This README follows the current status of code for my working project. It is a few snippets that are used for stereo-vision tracking. There are three key functions that will be combined in the near future. `blob_detect()` which takes the hardware ID values of two cameras, and opens up a multitude of windows showcasing the cameras' feed, with the ability to modify the HSV mask actively. When a mask is dialed-in, the `cv2.blob_detection()` algorithm kicks in and draws circles around each blob seen in a specific camera.

Once I am finished calibrating the cameras to recognize a specific LED color, I will then implement the `camera_dots_to_world()` which takes the position of blobs from each camera, and returns a real-world x,y,z position in meters where the stereoscopic cameras would theoretically locate an LED. It relies on a `frame2vector_cal()` helper function to do it's calculations.

Also included in this checkpoint is a PDF of my introduction to the thesis. Since my thesis is an interdiciplinary one, the introduction discusses the concepts I will be exploring throughout my thesis project. It is an initial draft.
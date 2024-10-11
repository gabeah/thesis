[[Thesis Menu]]
[[Thesis Meetings]]
Below is a log of progress made on the thesis. It serves to remind myself on all that I accomplished over the year. Here's hoping I continue to update it!!

## 10/6/24 - Some tracking works I guess?
Finally got the minMaxLoc working. It's loving the static that exists. Some finetuning should be done to get rid of static bits/find blobs in the video. I'm going to look into blob detection and then try to follow up with multiple camera streams.

## 9/30/24 - Made a Video Mask
In the evening before I am supposed to show Jim progress, I have worked on making a video mask to then do work on. I was able to do this by creating a mask using `cv.inRange()` function. This progress bit is saved at commit `e10dd71`. I am continuing to see if I can locate the brightest spot in the frame. If so, then I will try to draw a circle around it.
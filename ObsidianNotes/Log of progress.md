[[Thesis Menu]]
[[Thesis Meetings]]
Below is a log of progress made on the thesis. It serves to remind myself on all that I accomplished over the year. Here's hoping I continue to update it!!

## 2/21/25 - Long Update!!

[[belle thesis update]]

## 10/31/24 - 2 Cameras done! Now for some math stuff!!!
Right before the break, in a burst of efficiency, we got two cameras to work at the same time. I have moved to start doing math (boo!). I also know I need to do more writing. Going to add some more below after I do more reading/writing

## 10/11/24 - Tracking is improving, now to just try to get more cameras working
Interesting discoveries today. Looks like running more than one camera is pretty intensive on a USB hub. Looks like I need more than 1 USB controller for each. Luckily my laptop is capable of that.

In other news, I worked on some progress ticking away towards chapter 1. Started looking at Light Fantastic to get a rough outline of the history of lighting control. Looks like it halts around 1950 (probably because changes were less significant? Not sure yet).

Also, Blob tracking works! Exciting! We also need to try and figure out what the threshold function is doing. Its confusing the hell out of me. If I can lock down the blob tracking, that would be wonderful. I think I need to lock down the live feed first? I can also do some live tracking of camera 1 and not worry about a second feed just yet??? It feels stressful that we inch along slowly. Till next time. Commit hash tonight is `5b193f2`

## 10/6/24 - Some tracking works I guess?
Finally got the minMaxLoc working. It's loving the static that exists. Some finetuning should be done to get rid of static bits/find blobs in the video. I'm going to look into blob detection and then try to follow up with multiple camera streams.

## 9/30/24 - Made a Video Mask
In the evening before I am supposed to show Jim progress, I have worked on making a video mask to then do work on. I was able to do this by creating a mask using `cv.inRange()` function. This progress bit is saved at commit `e10dd71`. I am continuing to see if I can locate the brightest spot in the frame. If so, then I will try to draw a circle around it.
[[Thesis Menu]]
[[Log of progress]]

This chronicles the progress (and lack of) towards Belle's thesis.

The original idea was to demo the tracking software in Belle's thesis  "Chronicle of a Planet". The show explores Belle's relationship with her and the aspects of mother in her life. Those aspects were her own mother, her motherland as a student from China, and her mother tongue of Mandarin.

The original goal was to include the tracker in the chapter exploring motherland. In that chapter, Belle moves around a white umbrella. The intended effect was to track the umbrella in certain sections when Belle was criticizing the CCP most, to serve as the "watchful eye of a surveillance state". The show was set to open on February 14th.

### Where we started:

I had ended the semester feeling a bit disheartened. At that moment I was hoping for a fully functional demo of the tracking software ready to plug into a moving light. I realize now it was a pretty lofty task. Over that winter break, I spent some time digging around the internet to see if someone had done work with stereo vision to see what I was missing; this was when I discovered the work by [Temuge Batpurevl,](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html) a researcher with a PhD in Computer Science from Osaka University. Dr. Batpurev had spent time working with stereo vision, and had developed a script to calibrate stereo cameras. The issue with my project so far was that even if I had gotten a good track with the stereo cameras, the data coming out with my back-of-the-napkin math made no sense for the current context. That was because of two things: I hadn't calibrated for distortion by a single camera, and that I tried to rely on perfect placement of the two cameras with a measured distance instead of calibrating the two. Dr. Batpurev's script handled both of those problems.

I begun the semester integrating and testing Dr. Batpurev's code. His code relies on calibrating through using a checkerboard with a specific size and row/column count. In fact, opencv provides infrastructure for detecting a checkerboard for this exact purpose (!!!). The integration proved to be fairly straightforward. The next challenge was to send instructions over DMX.

#### DMX "Hell"

DMX is an interesting protocol << Describe aspects of DMX and how it as a serial protocol>>. While I had done some work in networking & protocols, my knowledge of implementing it in python wasn't the greatest. Thankfully, through the magic of open-source software, there were multiple people who had created python modules designed to work in DMX transmission. One of the defining characteristics of DMX was that every packet of data to be transmitted *had* to be 513 bytes; the first one is the **start code** and it denotes the way the signal should be interpreted (usually 0x00). The other 512 bytes communicate the behavior of the following 512 addresses, with one byte assigned to each address. Because of this, the majority of open-source software I encountered would have me either create the entire 513 byte packet, or to define what addresses were needed for the light (and what data it used).

The light for this was a ROBE Esprite, provided by Outlaw Lighting--A local supplier of lighting gear. It required, on average, 49 addresses to run, to control various aspects of the light. That made for an interesting evening of starting at datasheets provided by ROBE, and using the magic of object orientation to create a consistent way to *serialize* all the attributes I needed. I ended up getting it working on it's own, but I wasn't let off that easy. (Talk about the issues with USB ports & bus space and such).

#### So how did you get so close?

I felt really good going into the final week before Tech (the period where all technical aspects of a show are combined). I had a way to communicate with the light, and I had a way to track the LED. Sure, I needed to still find a way to communicate the position of the LED to the Esprite, but even if I could just start to feed junk data, and just see what happens, it would have been awesome. Wishful thinking.

In order to try and speed up the tracking to be as close to real-time as possible, I downgraded the resolution from 1920x1080 to 640x480. Thats a difference of 2457600 pixels to analyze to 307200 pixels (all at a hopeful speed of 20-30fps). The tracking works great when close up, but I was hoping for Belle to be dancing with the umbrella more than 15 feet from the camera, and with that resolution, the cameras had a *bit* of trouble even putting the pinprick of light that was the LED on screen. Oops.

However, if I tried to upscale the resolution, it would scale the calculations by a factor of at minimum 8. I am not the most efficient programmer, and some of the algorithms I used absolutely did not run in constant time (maybe elaborate on wtf that means). Even when I tried that, the computer struggled to keep pace (and also the LED did appear at a far distance very well).

Unfortunately, at that point, tech arrived and I resolved to put the project on hold while I finished designing for Belle's thesis.

#### Where does that put us now:

Well, it was certainly a step back, but the encouraging thing is that there is a pipeline that exists. After discussing with my advisors, the best step forward was to try to work on the pipeline in sections to figure out the capacity, and to generate demos for each particular step. That way, even if the final, full pipeline never came to completion, I would still be able to showcase my work in the various sections. The three demos Jim and I came up with were as follows:

1. [ ] A working demo showing my ability to write programmed sequences for a moving light to follow. This showcases my ability to just write onto DMX from my laptop.
2. [ ] A working demo tracking an LED with stereo camera, and displaying all the points (or points found every second) on a graph. Likely using matplotlib
3. [ ] A working demo that takes a generated graph (like one created through the previous demo), and sends it to the light.

If I can get all three working, then the pipeline should be trivial. This still means I need to work through calibrating the light, but I have some ideas on how to accomplish that. Although we are facing obstacles, this is a two step forward, one step back process.

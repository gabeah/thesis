[[Thesis Menu]] <- Go up

# Peter Meeting 10/11
Next week is a demo! Show Peter & Jim something!

# Jim Meeting 10/8
Blob recognition works!!

Need to figure out how OpenCV uses color, and how those translations work for when I eventually have to recalibrate the system

Next steps:
- 2 cameras?
- Live Feed?
- Can i show the xy values

When it comes to comparing two feeds. Pretend you are shooting a ray through each camera screen, and look for where the distance between the two rays in 3-space is minimized.

Note, focal distance will have to be calibrated. Look for it from the webcam manufacturer, or calculate it yourself with Trig :{

# Peter Meeting 10/4
You don't need 60fps, and you don't need 1080p. Compress the video/reduce the amount of data. Maybe use Davinci Resolve??

See if you can make a bounding box (pixels that are never going to be tracked into because of heights in theatrical spaces.) Parameterize!!!
	Then just calibrate with a person standing in space, maybe running a live feed and creating a bounding box.

# Jim Meeting 10/1
Jim is slacking me a document about learning opencv
https://learnopencv.com/introduction-to-epipolar-geometry-and-stereo-vision/
https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf
https://medium.com/@dc.aihub/3d-reconstruction-with-stereo-images-part-3-epipolar-geometry-98b75e40f59d
Mask looks cool, lets see if we can get some data output from the openCV stuff
	Maybe even another camera????

Also, look at trying to get the orangepi running to read data from DMX512
Also, start figuring out how to control the icue (or other light)

I should start working on this in two prongs, one dedicated to the furthering of OpenCV information, the other focusing on controlling a light

# Peter Meeting 9/27
Go talk to Jay Ewing about light filters and manufacturing the video system.

Need to do a more specialized search for *what books are helpful for this thesis?*
- Books on Lighting Control
- Manuals for lighting consoles through the years
- History of performance technology

USE ARCHIVE.ORG!! They have stuff!

Maybe reach out to Mark Coniglio

Come back with an outline for what the first chapter would contain.
# Jim Meeting 9/24
OpenCV looks like the library of choice to do work in

Need to start learning how it works, come back with a video

# Peter Meeting 9/20
Grant is done and submitted!

Looking for feedback on the thesis proposal:
- KB Had some books to look at! (Peter is sending to email)
- Clarify what responsive means for a light, how does it restore an actor/performer's agency
- Why should this be used in Belle's thesis (back up the claim)
- Text says I don't have a thesis advisor
- Pick a citation style (likely chicago), and STICK WITH IT
	- Also learn how Chicago style citations work
- Read through and really clarify what the topic and rationale are!!

Remember the convoluted idea of design: (Find a way to have an answer for each line here)
A specification of an object
manifested by and agent
intended to accomplish goals
in a particular environment
using a set of primitive components
satisfying a set of requirements
and subject to constraints

ToDo:
Make a list of the design requirements (ideals, required) [[Design Requirements]]
- How big is the gear?
- How advanced of the tech?
- What designs choices do I need to make?
- What are the limits?
Start thinking of the workflow/pass through of data (maybe use isadora to generate dummy)
Look for cheap 3Pin XLR Cables to start reading data from DMX chain
**CREATE THE DOCUMENT**
Start looking through the books and taking notes on what's in them!!

Look at URS Electronics and 
# Jim & Peter Meeting 9/17
Looks like Max and I will get the blackbox booth as laboratory space for a computer/interface and the blackbox will be a place to test stuff

Note on deadlines: There *should* be a prototype for Belle's thesis, but the prototype does not need to be fully functional. If the software isn't ready for the piece, I will still work on the show and look to make some demo later the semester.

**Specific Deadlines**: 
Deliver something CS-related by thanksgiving
1st Chapter due end of semester (with mini-orals during finals week)
DEMO FOR SYSTEM DURING MINI ORALS

If the demo (getting cameras, and moving a light) is working by the end of the semester, the spring focuses more on adding the bells and whistles of networking (which Charlie could help out with more).

I am allowed to use existing libraries to do some analysis on camera feeds (you can't use fancy commercial tech, but you also don't have to initially code C++ libraries)

Writing stuff: More theatre-heavy, dialogue about the history of lighting and how the more advanced control has departed from the reactive-live design.

Chapter layout:
1. History of lighting/lighting control
2. Lighting principles behind the project
3. CS principles behind the project
4. Working prototype/end result and use in a project

**NEEDS**:
* Apply for Initiative grant for tomorrow!
	* Send Jim email detailing specifics of the project for the grant
# Initial Meeting
Project looks and sounds good! Now I need to formalize it!

Research question ideas:
* How to we create a responsive, lightweight, cheap, software that re-injects the "liveness" to lighting design.
* What does that liveness bring back to a show

Look at Alex Swann's thesis for ideas on how to format it.
Max Keller's book about lighting (FIND)
	Go to the library and just start grabbing books!
	Archive.org
	jstor
	"Dance and media technologies" - paper (FIND)
More research in Troika-ranch!!!
Research in Performance and Technology (Josephine Machon)
Steve Dixon, Digital Performance
**LOOK IN THE BIBLIOGRAPHIES OF THESE BOOKS**
Most of these books from MIT Press

Departmental proposal needs the most detail, divisional needs less detail.
Include:
- What problems do you forsee?
- What materials/spaces do you need?
- Why this project, why now?

## What do i need by next time:
Thesis proposal
What materials do I need? (and when?)
* arduinos and such
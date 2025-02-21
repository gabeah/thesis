[[Thesis Menu]] <- Go up

# Peter 2/14
The progress is okay. Put the dang things to paper. Write about the progress and such

# Jim 1/29
We're back!!

Keep the software simple!
Get writing!! There are a lot of different things you have the ability to 
# Jim 12/3

Talk about 3 technical specifics for CS-component
- What is an image/how do we threshold an image
- What is blob detection/ how er we use it
- How do we locate someone in a space/what is the math for it
Write 3-5 paragraphs and add pseudocode and images for each and send to Jim. That can easily take up multiple pages
# Jim 11/19:

Chapter 2 - CS Lit / Preliminaries
- There is a chapter 4 problem that needs to be discussed
	- Covering the basic CS concepts of blob detection / image processing / stereo
# Peter 11/15:

Initial thoughts on the writings:
- Appia's thoughts around dance & movement cam serve as a good anchor
- Also check out Joseph Svoboda
- CHAPTER 1 IS DUE 12/6 AT NOON!!!! DON'T BE LATE!!!!
- Thesis mini-oral is on 12/13 at 9am!!!
# jim
- Plenoptic modeling
- Get the cameras set up
- also know that in the spring, I will have to actually care about the literature for the CS components, likely looking at how to generalize the program

# Meeting with Peter 11/1:
### What do I need to submit in writing in 2 weeks?
- Introduction
- Flushed out outline for Chapter 1.

#### Structure of the thesis:
- Introduction - adapted version of the thesis proposal (See guide)
	- What is the motivation
	- What is the scope
	- What is the process
	- Why is it important
	- How will you approach it
	- Maybe dive into the creation of the vari-light (how did the creation of the light drive the creation of other things)
	- etc.
- Chapter 1 (theatre history chapter) - background research / literature review
	- I am making an argument that "liveness" has been taken away as lighting becomes more automated and timecoded
	- Reverse centaur problems
	- How did we get here?
- Chapter 2 (the computer science i am engaging with)
	- What is the CS problems
	- How did I accomplish it/solve the problems?
- Chapter 3 (the results and outcome???)
	- How did this engage with the previous chapters

By the end of the semester, I should submit about 20-25 pages.

# Caleah Meeting

## Various pathways to research
* Psych of light/color, look outside of just light in live performance, and look at how light affects the body (SAD, how light creates an environment)
* Look at interviews of lighting designers (check archive.org)
* Does current lighting technology take away the "human element of a show"
* History of lighting design for live music/rock performances (look around 1970's with performances around KISS and David Bowie)
* History of the costs of theatrical equipment, and how it falls onto the consumer of art
* Effect of this software on jobs.
* Lighting in circus!! How do they track their people!?
	* Connecticut catastrophe, how did lighting technology change in circus performance
	* Hartford Circus Fire (Ringling Brothers) (1944)
* Differences between how indoor and outdoor venues are lit
## Research tips:
- If you have a few pathways, start a google doc for each pathway and put all the information that you currently know.
- Search library/websites for resources on that specific pathway, and compile little bibliographies for each pathway
- Find what pathways are the most resource-rich, and focus further there.
- Look for umbrella topics
- USE https://libguides.reed.edu/theatre!!!
- USE https://libguides.reed.edu/theatre/primary_historical_sources!!!
- USE THE LIBGUIDES
## Other notes:
- Maybe reach out to Hopscotch artists and interview them
- NYPL (look at their blog)
# Peter Meeting & Jim Meeting
Catch up on info
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
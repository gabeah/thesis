#### Remember the convoluted idea of design: (Find a way to have an answer for each line here)

**A specification of an object
manifested by and agent
intended to accomplish goals
in a particular environment
using a set of primitive components
satisfying a set of requirements
and subject to constraints**

**A specification of an object** - A software for reactive lighting design
**Manifested by an agent** - Me, Gabe Howland
**Intended to accomplish goals** - Tracking a performer in with a light
**In a particular environment** - A performance space that utilizes moving lights
**Using a set of primitive components** - Off-the-shelf hardware (webcams, raspberry pi, etc.)
**Satisfying a set of requirements** -  Open source, resource light, cheap to implement
**Subject to constraints** - time, budget, complexity

## Design Specifics:

#### How big is the gear?
The gear should make up the following:
- A laptop
- 2+ webcams & tripods
- Some Raspberry Pis
Overall, it should be able to be carried in a backpack

#### How advanced is the tech?
The technology should be off the shelf, thus accessible to the average consumer. Thus webcams are limited to HD Webcams (1080p).

For laptops, the project will be written around the current resources of my own laptop (which is above average for the average consumer). If it can run okay on the department's Mac Minis, I will consider that a win. Hopefully most productions have that computing power.

#### Visible Spectrum or No?
A current debate is being had about whether or not I should work in the visible spectrum or not. Cursory research has been done to look at NIR cameras. After a discussion with Jim, there are concerns that an IR diode may be difficult to parse from other video feeds.

The current strategy is to develop around a visible beacon, so it is easier to debug and setup, and then shift to NIR once I have the experience with working in OpenCV.

#### What libraries will you use?
Currently, I am looking at the OpenCV library. It is a library of C++ and Python functions designed for all manner of computer vision projects.


***More Design Notes will be added as the design evolves***
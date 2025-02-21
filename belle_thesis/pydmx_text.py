#!/usr/bin/env python3
"""Test script to show how to use module."""

from sys import exit as sys_exit
from time import sleep
import pyenttec as ent
from dmx import Colour, DMXInterface, DMXLight3Slot, DMXUniverse
from esprite_profile import Esprite

def write_dmx(universe,light):
    for i, chan in enumerate(light.serialise_pydmx()):
        print(i+1, chan)
        universe.set_channel(i, chan)
        #robe.set_color([i,0,i])
    universe.render()

def main():
    """Entry point function."""
    universe = ent.DMXConnection("/dev/ttyUSB0")
    robe = Esprite()
    robe.set_intensity(0)
    robe.set_color([0,0,0])

    for i in range(50):
        robe.set_color([0,0,4*i])
        robe.set_tilt(i)
        write_dmx(universe,robe)
        sleep(0.1)
    robe.go_home()
    write_dmx(universe, robe)
        

    

    return 0


if __name__ == "__main__":
    sys_exit(main())
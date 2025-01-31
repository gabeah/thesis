# Main Software for Belle's Thesis
# Utilizing both the PyDMX and python_stereo_camera_calibrate created by Jacob Allen and Temuge Batpurev

from calib import *
from dmx import Colour, DMXInterface, DMXLight3Slot, DMXUniverse
from video_track import blob_dect
from curses import wrapper

class Menu():
    def __init__(self, top):
        self.menu_top = top
    
    def add_item(self, item_name):
        crawler = self.menu_top
        
        while crawler.dn != None:
            crawler = crawler.dn
        
        new = MenuItem(item_name, crawler, None)
        crawler.dn = new

class MenuItem():
    def __init__(self, name, above=None, below=None):
        self.item_name = name
        self.up = above
        self.dn = below
    

def main(stdscr):
    # Clear screen
    stdscr.clear()

    while True:
        stdscr.refresh()
        keypress = stdscr.getkey()

    stdscr.refresh()
    stdscr.getkey()



wrapper(main)
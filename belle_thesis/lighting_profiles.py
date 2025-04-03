from dmx import DMXLight
#from dmx.constants import DMX_MAX_ADDRESS, DMX_MIN_ADDRESS
from typing import List
import abc

DMX_MAX_ADDRESS = 512
DMX_MIN_ADDRESS = 1

class Esprite():

    # The class representing the Robe Esprite. Data is based on datasheet
    # For simpicity's sake, we are using the Esprite in mode 1

    def __init__(self, address: int = 1):
        # All values here must be between 0-255
        self._address = address
        self._pan = 128
        self._tilt = 128
        self._color = [0,0,0]
        self._intensity = 0
        self._shutters = {1: (0,128), 2: (0,128), 3: (0,128), 4: (0,128)}
        self._zoom = 128
        self._iris = 128
        self._focus = 128

    @property
    def start_address(self) -> int:
        return self._address
    
    @property
    def end_address(self) -> int:
        """End address (inclusive) of the light."""
        end_address = self._address + self.slot_count - 1
        if end_address > DMX_MAX_ADDRESS or end_address < DMX_MIN_ADDRESS:
            return ((end_address - DMX_MIN_ADDRESS) % DMX_MAX_ADDRESS) + DMX_MIN_ADDRESS
        return end_address

    @property
    def highest_address(self) -> int:
        """Highest address used by this light."""
        if self.end_address < self.start_address:
            return DMX_MAX_ADDRESS
        return self.end_address

    @property
    def slot_count(self) -> int:
        return 49

    @property
    def color(self):
        return self._color
    
    @property
    def pan_tilt(self):
        return [self._pan, self._tilt]
    
    @property
    def intensity(self) -> int:
        return self._intensity

    @property
    def shutters(self) -> dict():
        return self._shutters

    @property
    def zoom(self) -> int:
        return self._zoom
    
    @property
    def iris(self) -> int:
        return self._iris
    
    @property
    def focus(self) -> int:
        return self._focus

    def set_zoom(self, zoom: int):
        self._zoom = zoom
    
    def set_intensity(self, intensity: int):
        self._intensity = intensity

    def set_iris(self, iris: int):
        self._iris = iris

    def set_focus(self, focus: int):
        self._focus = focus

    def serial_shutters(self):
        shutters_out = []
        for shutter in self._shutters.keys():
            shutters_out.append(self._shutters[shutter][0])
            shutters_out.append(self._shutters[shutter][1])
        return shutters_out

    def set_shutter(self, shutter: int, pos: int, rot: int):
        if pos < 0 or pos > 255:
            print("Error: Position out of range")
            return
        if rot < 0 or rot > 255:
            print("Error: Rotation out of range")
            return
        if shutter > 4 or shutter < 0:
            print("Error: Invalid shutter index")
            return
        self._shutters[shutter] = (pos, rot)
        return

    def set_pan_tilt(self, p: int, t: int):
    # Take pan value between 0 - 540 degrees, convert to range between 0-255
    # Take tilt value between 0 - 265, convert to range between 0 - 255
        self.set_pan(p)
        self.set_tilt(t)
    
    def set_pan(self, p: int):
        if -270 < p < 270:
            self._pan = round(((p+270)/540) * 255)
        else:
            print(f"Error: Value {p} out of pan range")
    
    def set_tilt(self, t: int):
        if -135 < t < 135:
            self._tilt = round((t+135)/270 * 255)
        else:
            print(f"Error: Value {t} out of tilt range")

    def set_color(self, rgb):
        self._color = rgb

    def go_home(self):
        self.set_color([0,0,0])
        self.set_pan_tilt(0,0)
    
    def serialise_pydmx(self):
        #print("serializing")
        """
        Order of address transmission:
            Pan
            Pan-fine (not needed)
            Tilt
            Tilt-fine (not needed)
            Power (not needed)
            LED Frequency (not needed, default 10)
            LED Freq Fine (not needed, default 128)
            Max Light Intensity (not needed)
            Color wheel 1 (not needed for now)
            Color wheel 1 fine (not needed)
            Color wheel 2 (not)
            Color wheel 2 fine (what do you think)
            Cyan
            Magenta
            Yellow
            CTO#
            channels 18-22 not needed (value 0 default)
            chan 23 default to 128
            24-26 default to 0
            27 default to 128
            28,29 default to 0
            30 default to 128
            31-33 default to 0
            34 Zoom default 128
            35 zoom fine (default 0)
            36 focus default 128
            37 focus fine default 0
            38 framing shutters mode (not needed)
            39 shutter 1 pos default 0
            40 shutter 1 rot default 128
            41 shutter 2 pos default 0
            42 shutter 2 rot default 128
            43 shutter 3 pos default 0
            44 shutter 3 rot default 128
            45 shutter 4 pos default 0
            46 shutter 4 rot default 0
            47 shutter strobe def 32
            48 intensity def 0
            49 intensity fine 0
        """
        unneded_chunk = [0,0,0,0,0,128,0,0,0,128,0,0,128,0]
        shutters = self.serial_shutters()
        #print(shutters)
        serial_out = [self._pan] + [0] + [self._tilt] + [0,0,0,10,128,0,0,0,0,0] + self._color + [0] + unneded_chunk + [self._iris] + [0] + [self._zoom] + [0] + [self._focus] + [0,128] + shutters + [32] + [self._intensity] + [0]
        intensity_only = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,self._intensity,0]
        #print(len(intensity_only))
        #print(serial_out)
        #print(len(serial_out))
        return serial_out

class ICueAndIris():
    
    # The class representing the Rosco ICue w/ Iris attachment (assuming DMX footprint is the same)

    def __init__(self, address: int = 1, mode: int = 1):
        # All values here must be between 0-255
        self._address = address
        self._pan = 128
        self._tilt = 128
        self._iris = 128
        self._mode = mode
        self._slot_count = 2

    @property
    def mode(self) -> int:
        return self._mode

    @property
    def iris(self) -> int:
        return self._iris

    @property
    def pan_tilt(self):
        return [self._pan, self._tilt]

    @property
    def start_address(self) -> int:
        return self._address
    
    @property
    def end_address(self) -> int:
        """End address (inclusive) of the light."""
        end_address = self._address + self.slot_count - 1
        if end_address > DMX_MAX_ADDRESS or end_address < DMX_MIN_ADDRESS:
            return ((end_address - DMX_MIN_ADDRESS) % DMX_MAX_ADDRESS) + DMX_MIN_ADDRESS
        return end_address

    @property
    def highest_address(self) -> int:
        """Highest address used by this light."""
        if self.end_address < self.start_address:
            return DMX_MAX_ADDRESS
        return self.end_address

    def set_pan_tilt(self, p: int, t: int):
        # Take pan value between 0 - 540 degrees, convert to range between 0-255
        # Take tilt value between 0 - 265, convert to range between 0 - 255
        self.set_pan(p)
        self.set_tilt(t)
    
    def set_pan(self, p: int):
        if -270 < p < 270:
            self._pan = round(((p+270)/540) * 255)
        else:
            print(f"Error: Value {p} out of pan range")
    
    def set_tilt(self, t: int):
        if -135 < t < 135:
            self._tilt = round((t+135)/270 * 255)
        else:
            print(f"Error: Value {t} out of tilt range")
    
    def serialise_pydmx(self):
        serial_out = pan_tilt() + [iris()]
        return serial_out
    

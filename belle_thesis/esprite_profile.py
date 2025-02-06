from dmx import light

class esprite(DMXLight):

    # The class representing the Robe Esprite. Data is based on datasheet
    # For simpicity's sake, we are using the Esprite in mode 1

    def __init__(self, address: int = 1):
        # All values here must be between 0-255
        self._address = address
        self._pan = 128
        self._tilt = 128
        self._color = [255,255,255]

    @property
    def start_address(self) -> int:
        return self._address
    
    @property
    def slot_count(self) -> int:
        return 49

    @property
    def color(self) -> List(int):
        return self._color
    
    @property
    def pan_tilt(self) -> List(int):
        return [self._pan, self._tilt]

    def set_pan_tilt(self, p: int, t: int):
    # Take pan value between 0 - 540 degrees, convert to range between 0-255
    # Take tilt value between 0 - 265, convert to range between 0 - 255

        if -270 < p < 270:
            self._pan = round(((p+270)/540) * 255)
        else:
            print(f"Error: Value {p} out of pan range")

        if -135 < t < 135:
            self._tilt = round((t+135)/270 * 255)
        else:
            print(f"Error: Value {t} out of tilt range")
    
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

    def set_color(self, rgb: List(int)):
        self._color = rgb

    def go_home(self):

        set_color([0,0,0])
        set_pan_tilt(0,0)
    
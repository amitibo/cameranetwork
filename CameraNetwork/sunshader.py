"""Control the camera sunshader
"""
from __future__ import division
try:
    import pyfirmata
    from pyfirmata import ArduinoDue, util
except:
    import warnings
    warnings.warn("Failed loading pyfirmata. Possibly working locally.")

class SunShader(object):
    """Control the camera sunshader.
    """

    def __init__(self, com='/dev/ttyACM0', pin=5, servo_range=(0, 180)):

        try:
            self.board = ArduinoDue(com)
        except Exception as e:
            self.board = None
            return

        self.board.digital[pin].mode = pyfirmata.SERVO
        self.pin = pin
        self.servo_range = servo_range
        self._angle = 0
        
        #
        # Initialize to center (not to load the servo).
        #
        self.setAngle(90)

    def __del__(self):
        if self.board is not None:
            self.board.exit()

    def setAngle(self, angle):
        """Set the angle of the sunshader."""
        
        if self.board is None:
            raise Exception('No Arduino board available.')

        #
        # Round the angle
        #
        angle = int(angle + 0.5)
        
        if angle < self.servo_range[0] or angle > self.servo_range[1]:
            raise Exception(
                'Angle out of range: angle={} range={}.'.format(angle, self.servo_range)
            )

        self._angle = angle
        self.board.digital[self.pin].write(angle)

    def getAngle(self):
        """Get the last set angle of the sunshader."""
        
        return self._angle


if __name__ == '__main__':
    main()

"""Control the camera sunshader, sprinkler and more through the Arduino.
"""
from __future__ import division
try:
    import pyfirmata
    from pyfirmata import ArduinoDue, util
except:
    import warnings
    warnings.warn("Failed loading pyfirmata. Possibly working locally.")


class ArduinoAPI(object):
    """Control the camera sunshader, water sprinkler and more.

    Args:
        com (str, optional): Com port of the arduino.
        sunshader_pin (int, optional): Pin number of the sunshader servo.
        servo_range (tuple, optional): (min, max) angles of the sunshader servo.
        sprinkler_ping (int, optional): Pin number of the sprinkler relay.

    """

    def __init__(
            self,
            com='/dev/ttyACM0',
            sunshader_pin=5,
            servo_range=(0, 180),
            sprinkler_pin=6):

        try:
            self.board = ArduinoDue(com)
        except Exception as e:
            self.board = None
            return

        #
        #
        # Setup the sunshader pin.
        #
        self.board.digital[sunshader_pin].mode = pyfirmata.SERVO
        self.sunshader_pin = sunshader_pin
        self.servo_range = servo_range
        self._angle = 0

        #
        # Setup the sprinkler pin.
        #
        self.board.digital[sprinkler_pin].mode = pyfirmata.DIGITAL
        self.sprinkler_pin = sprinkler_pin

        #
        # Initialize to center (not to load the servo).
        #
        self.setAngle(90)

    def __del__(self):
        if self.board is not None:
            self.board.exit()

    def setAngle(self, angle):
        """Set the angle of the sunshader.

        Args:
            angle (int): Requested angle of the sunshader.

        """

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
        self.board.digital[self.sunshader_pin].write(angle)

    def getAngle(self):
        """Get the last set angle of the sunshader.

        Returns:
            The last set angle of the sunshader.

        """

        return self._angle

    def setSprinkler(self, state):
        """Set the state of the sprinkler.

        Args:
            state (bool): True to turn on the sprinkler, otherwise turn off.

        """

        self.board.digital[self.sprinkler_pin].write(1 if state else 0)


if __name__ == '__main__':
    main()

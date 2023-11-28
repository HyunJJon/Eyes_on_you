from dataclasses import dataclass
import glob
import sys
from typing import List
import serial
import time

mode: dict = {"bracket": 1, "rail": 2}


@dataclass
class ArduinoController:
    width: float = 640
    height: float = 480
    send_delay: float = 0.01  # Delay between sends
    baudrate: int = 9600
    port: str = "/dev/ttyACM0"

    def possible_ports(self) -> List[str]:
        """Returns a list of possible ports for the Arduino."""
        if sys.platform.startswith("win"):
            ports = ["COM%s" % (i + 1) for i in range(256)]
        elif sys.platform.startswith("linux") or sys.platform.startswith(
            "cygwin"
        ):
            ports = glob.glob("/dev/tty[A-Za-z]*")
        elif sys.platform.startswith("darwin"):
            ports = glob.glob("/dev/tty.*")
        else:
            raise EnvironmentError("Unsupported platform")
        return ports

    def __post_init__(self) -> None:
        success = False
        for port in [self.port] + self.possible_ports():
            try:
                self.ser = serial.Serial(
                    port, self.baudrate, timeout=1, write_timeout=1
                )
                print(f"[ARDUINO] Connected to {port}")
                success = True
                break
            except (OSError, serial.SerialException):
                print(f"[ARDUINO] Failed to connect to {port}")
        time.sleep(2)  # Wait for Arduino to boot
        if not success:
            raise ConnectionError("Failed to connect to Arduino")
        print("[ARDUINO] Arduino ready!")

    def __del__(self) -> None:
        self.ser.close()

    def read(self):
        if self.ser.in_waiting > 0:
            data = self.ser.readline().decode("utf-8").strip()
            print(f"[ARDUINO]: read '{data}")

    def reset(self) -> None:
        # reset all the servos to 90 degrees, and stepper to 0
        string = "0,0,3\n"
        self.ser.write(string.encode())
        print(f"[ARDUINO]: send reset '{string}'")
        time.sleep(self.send_delay)

    def bracket(self, delta_x: float, delta_y: float) -> None:
        string = f"{int(delta_x)},{int(delta_y)},{int(mode['bracket'])}\n"
        # input = delta_x, delta_y, 1
        self.ser.write(string.encode())
        print(f"[ARDUINO]: send x and y '{string}'")
        time.sleep(self.send_delay)  # Delay between sends

    def rail(self, distance: float, velocity_percentage: float) -> None:
        string = f"{int(distance)},{int(velocity_percentage)},{int(mode['rail'])}\n"  # noqa: E501
        # input = direction, velocity_percentage, 2
        self.ser.write(string.encode())
        print(f"[ARDUINO]: send distance and velocity percentage '{string}'")
        time.sleep(self.send_delay)  # Delay between sends

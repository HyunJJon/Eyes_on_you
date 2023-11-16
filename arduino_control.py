from dataclasses import dataclass
import glob
import sys
from typing import List
import serial
import time


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
        for port in [self.port] + self.possible_ports():
            try:
                self.ser = serial.Serial(
                    port, self.baudrate, timeout=1, write_timeout=1
                )
                print(f"[ARDUINO] Connected to {port}")
                break
            except (OSError, serial.SerialException):
                print(f"[ARDUINO] Failed to connect to {port}")
        time.sleep(2)  # Wait for Arduino to boot
        print("[ARDUINO] Arduino ready!")

    def __del__(self) -> None:
        self.ser.close()

    def stop(self) -> None:
        # move to forward view and stop
        encoded = "-2, -2".encode()
        self.ser.write(encoded)

    def send_x_y(self, x: float, y: float) -> None:
        string = f"{x:.2f},{y:.2f}\n"
        self.ser.write(string.encode())
        print(f"[ARDUINO]: send x and y '{string}'")
        time.sleep(self.send_delay)  # Delay between sends

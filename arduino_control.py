from dataclasses import dataclass
import serial
import time


@dataclass
class ArduinoController:
    width: float = 100
    height: float = 60
    time_step: float = 0.1
    baudrate: int = 9600
    port: str = "COM3"

    def __post_init__(self) -> None:
        self.ser = serial.Serial(self.port, self.baudrate)
        time.sleep(2)

    def __del__(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass

    def send_x_y(self, x: float, y: float) -> None:
        x = x / 10
        y = y / 10
        encoded = f"{x:.2f},{y:.2f}\n".encode()
        print(encoded)
        self.ser.write(encoded)
        time.sleep(self.time_step)  # Delay between sends

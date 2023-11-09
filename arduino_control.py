from dataclasses import dataclass
import serial
import time


@dataclass
class ArduinoController:
    width: float = 640
    height: float = 480
    time_step: float = 0.01
    baudrate: int = 9600
    port: str = "/dev/ttyACM0"

    def __post_init__(self) -> None:
        self.ser = serial.Serial(self.port, self.baudrate)
        time.sleep(2)

    def __del__(self) -> None:
        self.ser.close()

    # def send_x_y(self, x: float, y: float) -> None:
    #     x = x * 10 / self.width
    #     y = y * 10 / self.height
    #     encoded = f"{x:.2f},{y:.2f}\n".encode()
    #     print(encoded)
    #     self.ser.write(encoded)
    #     time.sleep(self.time_step)  # Delay between sends

    def stop(self) -> None:
        # move to forward view and stop
        encoded = "-2, -2".encode()
        self.ser.write(encoded)

    def send_x_y(self, x: float, y: float, mode: int) -> None:
        if mode == 1:
            x_dir = x
            y_dir = y
        else:
            x_dir = 0
            y_dir = 0
        encoded = f"{x_dir:.2f},{y_dir:.2f}\n".encode()
        print(encoded)
        self.ser.write(encoded)
        time.sleep(self.time_step)  # Delay between sends

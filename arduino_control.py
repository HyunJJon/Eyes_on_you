import serial
import time


class ArduinoController:
    width: float = 960
    height: float = 720
    time_step: float = 0.1
    baudrate: int = 9600
    port: str = "COM3"

    def __init__(self) -> None:
        self.ser = serial.Serial(self.port, self.baudrate)
        time.sleep(2)

    def __del__(self) -> None:
        self.ser.close()

    def send_x_y(self, x: float, y: float) -> None:
        x = x / self.width
        y = y / self.height
        encoded = f"{x:.2f},{y:.2f}\n".encode()
        print(encoded)
        self.ser.write(encoded)
        time.sleep(self.time_step)  # Delay between sends

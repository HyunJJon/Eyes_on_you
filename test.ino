#include <Servo.h> 

Servo servo1;  
Servo servo2;  

void setup() 
{ 
  Serial.begin(9600);  // start serial communication at 9600bps
  servo1.attach(5);  // attaches the servo on pin 5 to the servo object 
  servo2.attach(3);  // attaches the servo on pin 3 to the servo object 
} 

void loop() 
{ 
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');  // read the incoming data as string until newline character
    int commaIndex = data.indexOf(',');  // find the index of the comma separator
    String x_str = data.substring(0, commaIndex);  // extract the x value substring
    String y_str = data.substring(commaIndex + 1);  // extract the y value substring
    float x = x_str.toFloat();  // convert the x value to float
    float y = y_str.toFloat();  // convert the y value to float

    int angle1 = x * 180;  // map from 0-1 to 0-180 degrees
    int angle2 = y * 180;  // map from 0-1 to 0-180 degrees

    servo1.write(angle1);  // set the servo position
    servo2.write(angle2);  // set the servo position
  }
}

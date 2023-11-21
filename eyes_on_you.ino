#include <Servo.h>

// Constants
const int BAUD_RATE = 9600;
const int SERVO_PIN_1 = 5;
const int SERVO_PIN_2 = 3;
const int STEP_PIN = 9;
const int DIR_PIN = 8;
const int ENABLE_PIN = 10;
const int MIN_STEP_DELAY = 100;
const int MAX_STEP_DELAY = 500;
const int DEFAULT_STEPS = 100;
const int RESET_ANGLE = 90;
const int SERVO1_MIN_ANGLE = 0;
const int SERVO1_MAX_ANGLE = 95;
const int SERVO2_MIN_ANGLE = 0;
const int SERVO2_MAX_ANGLE = 180;
const int SERIAL_TIMEOUT = 1000;
const int MAX_STEPS = 20000;

// Servo motors
Servo servo1;
Servo servo2;
int angle1 = RESET_ANGLE;
int angle2 = RESET_ANGLE;

// Stepper motor control variables
int currentSteps = 0;

// Command structure
struct Command {
  int x = 0;
  int y = 0;
  int mode = 0;
} command;

void setup() {
  // Setup serial communication and pins
  Serial.begin(BAUD_RATE);
  Serial.setTimeout(SERIAL_TIMEOUT);

  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  servo1.attach(SERVO_PIN_1);
  servo2.attach(SERVO_PIN_2);

  digitalWrite(ENABLE_PIN, LOW);
}

Command parseCommand() {
  // Parse command from serial input
  String data = Serial.readStringUntil('\n');
  Command parsedCommand;
  int firstComma = data.indexOf(',');
  int secondComma = data.indexOf(',', firstComma + 1);

  // Check for valid data format
  if (data.length() == 0 || firstComma == -1 || secondComma == -1) {
    Serial.println("Error: Invalid command format");
    return {0, 0, 0}; // Return a default error command
  }

  parsedCommand.x = data.substring(0, firstComma).toInt();
  parsedCommand.y = data.substring(firstComma + 1, secondComma).toInt();
  parsedCommand.mode = data.substring(secondComma + 1).toInt();

  return parsedCommand;
}

void moveBracket(int deltaX, int deltaY) {
  // Move servo motors
  // Constrain angles to prevent damage to servos
  angle1 = min(max(angle1 + deltaX, SERVO1_MIN_ANGLE), SERVO1_MAX_ANGLE);
  angle2 = min(max(angle2 + deltaY, SERVO2_MIN_ANGLE), SERVO2_MAX_ANGLE);
  servo1.write(angle1);
  servo2.write(angle2);
}

void moveRail(int distance, int velocityPercent) {
  // Move stepper motor
  int dir = distance > 0 ? 1 : -1;
  int delayPerStep =
      map(velocityPercent, 0, 100, MAX_STEP_DELAY, MIN_STEP_DELAY);
  int nextSteps = currentSteps + distance;
  if (nextSteps < 0 || nextSteps > MAX_STEPS) {
    // If the stepper motor will exceed its range, do nothing
    return;
  }
  digitalWrite(DIR_PIN, distance > 0 ? HIGH : LOW);
  for (; currentSteps != nextSteps; currentSteps += dir) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(delayPerStep);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(delayPerStep);
  }
}

void resetAll() {
  // Reset servo angles and stepper position
  moveBracket(RESET_ANGLE - angle1, RESET_ANGLE - angle2);
  moveRail(-currentSteps, 100);
}

void executeCommand(Command command) {
  switch (command.mode) {
  case 1: // Servo adjustment mode
    moveBracket(command.x, command.y);
    break;
  case 2: // Stepper movement mode
    moveRail(command.x, command.y);
    break;
  default: // Reset servo angles and stepper position
    resetAll();
    break;
  }
}

void clearBuffer() {
  // Clear serial input buffer to prevent overflow
  while (Serial.available() > 0) {
    Serial.read();
  }
}

void loop() {
  // Read serial input and execute command
  if (Serial.available() > 0) {
    // If a command is received, parse and execute it
    command = parseCommand();
    Serial.println("x: " + String(command.x) + " y: " + String(command.y) +
                   " mode: " + String(command.mode));
    executeCommand(command);
    clearBuffer();
    return;
  }

  // If no command is received, use the last command
  executeCommand(command);
}

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#define SAMPLE_RATE_HZ  50 
#define WINDOW_SIZE     60 

const char GESTURE_NAMES[6][12] = {
  "idle", "up", "down", "circle_cw", "shake", "tap"
};

Adafruit_MPU6050 mpu;
uint8_t current_gesture = 0;
bool is_recording = false;
uint16_t sample_count = 0;
uint32_t start_time = 0;

void setup() {
  Serial.begin(9600);
  delay(100);
  
  // Инициализация MPU6050
  if (!mpu.begin()) {
    Serial.println("ERROR: MPU6050_INIT_FAILED");
    while (1); 
  }
  
  // Настройка датчика 
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_5_HZ);
  
  printInstructions();
}

void loop() {
  // Проверка команды от пользователя
  if (Serial.available() > 0) {
    processCommand(Serial.read());
  }
  
  // Запись данных
  if (is_recording) {
    recordSample();
  }
}

void processCommand(char cmd) {
  switch(cmd) {
    case '0'...'5':
      startRecording(cmd - '0');
      break;
    case '\n': 
    case '\r':
      break;
    default:
      Serial.print("ERROR: UNKNOWN_CMD:");
      Serial.println(cmd);
  }
}

void startRecording(uint8_t gesture_id) {
  if (gesture_id > 5) {
    Serial.println("ERROR: INVALID_GESTURE_ID");
    return;
  }
  
  current_gesture = gesture_id;
  is_recording = true;
  sample_count = 0;
  start_time = millis();

  delay(1000);
}

// Запись одного сэмпла данных
void recordSample() {
  static uint32_t last_sample_time = 0;
  uint32_t current_time = millis();
  
  if (current_time - last_sample_time < (1000 / SAMPLE_RATE_HZ)) {
    return;
  }
  last_sample_time = current_time;
  
  // Чтение данных с датчика
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  
  // Вывод данных в формате CSV
  Serial.print(current_time);
  Serial.print(",");
  Serial.print(a.acceleration.x, 15); 
  Serial.print(",");
  Serial.print(a.acceleration.y, 15);
  Serial.print(",");
  Serial.print(a.acceleration.z, 15);
  Serial.print(",");
  Serial.println(current_gesture);
  
  sample_count++;
  if (sample_count >= WINDOW_SIZE) {
    stopRecording();
  }
}

void stopRecording() {
  is_recording = false;
}

void printInstructions() {
  Serial.println(); 
  Serial.println("Commands: 0-5=record gesture, s=status, h=help");
  Serial.println("Gestures: 0:idle, 1:up, 2:down, 3:circle_cw, 4:shake, 5:tap");
  Serial.println("timestamp,accel_x,accel_y,accel_z,gesture_label");
}

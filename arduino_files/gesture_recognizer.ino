#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

#include "gesture_model.h"

// Параметры модели 
constexpr int WINDOW_SIZE = 60;  
constexpr int NUM_AXES = 3; 
constexpr int NUM_CLASSES = 6; 
constexpr float SAMPLE_RATE = 50.0f; 
constexpr int SAMPLE_INTERVAL = 20; 

// Параметры скользящего окна
constexpr int OVERLAP = 30; 
constexpr int NEW_SAMPLES_NEEDED = 30; 

// Параметры квантования модели 
constexpr float INPUT_SCALE = 0.0078125f; 
constexpr int INPUT_ZERO_POINT = 0; 

const char* GESTURE_NAMES[] = {
  "IDLE (неподвижность)",
  "UP (взмах вверх)",
  "DOWN (взмах вниз)",
  "CIRCLE_CW (круг по часовой)",
  "SHAKE (стряхивание)",
  "TAP (постукивание)"
};

// Параметры фильтра низких частот 
constexpr float ALPHA = 0.4f; 

Adafruit_MPU6050 mpu;

// Буфер данных 
float data_buffer[WINDOW_SIZE][NUM_AXES] = {0};
int buffer_index = 0;
int samples_collected = 0;

// Отфильтрованные значения 
float filtered_x = 0, filtered_y = 0, filtered_z = 0;

// TFLite Micro объекты
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

constexpr int kTensorArenaSize = 16 * 1024; 
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

unsigned long last_sample_time = 0;
unsigned long last_prediction_time = 0;
int last_predicted_gesture = -1;
int prediction_count = 0;
const int PREDICTION_THRESHOLD = 2; 

bool initTFLiteModel() {
  tfl_model = tflite::GetModel(gesture_model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Версия модели не совпадает: ");
    Serial.print(tfl_model->version());
    Serial.print(" != ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    return false;
  }
  
  static tflite::AllOpsResolver resolver;
  
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("ERROR: Ошибка выделения памяти для тензоров");
    return false;
  }
  
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  
  if (input_tensor->dims->size != 3 || 
      input_tensor->dims->data[0] != 1 ||
      input_tensor->dims->data[1] != WINDOW_SIZE ||
      input_tensor->dims->data[2] != NUM_AXES) {
    Serial.println("ERROR: Неверная форма входного тензора");
    return false;
  }
  
  // Вывод информации о модели
  Serial.println("Модель успешно инициализирована:");
  Serial.print("Входной тензор: [");
  Serial.print(input_tensor->dims->data[0]);
  Serial.print(", ");
  Serial.print(input_tensor->dims->data[1]);
  Serial.print(", ");
  Serial.print(input_tensor->dims->data[2]);
  Serial.println("]");
  
  Serial.print("Тип входных данных: ");
  switch (input_tensor->type) {
    case kTfLiteFloat32: Serial.println("float32"); break;
    case kTfLiteInt8: Serial.println("int8"); break;
    case kTfLiteUInt8: Serial.println("uint8"); break;
    default: Serial.println("unknown"); break;
  }
  
  // Для квантованных моделей выводим параметры квантования
  if (input_tensor->quantization.type == kTfLiteAffineQuantization) {
    Serial.print("Масштаб квантования: ");
    Serial.println(input_tensor->params.scale, 6);
    Serial.print("Нулевая точка: ");
    Serial.println(input_tensor->params.zero_point);
  }
  
  Serial.print("Размер модели: ");
  Serial.print(sizeof(gesture_model) / 1024.0);
  Serial.println(" KB");
  
  return true;
}

// Предобработка данных 
void preprocessData(float raw_data[WINDOW_SIZE][NUM_AXES], int8_t* quantized_data) {
  float axis_data[WINDOW_SIZE];
  
  for (int axis = 0; axis < NUM_AXES; axis++) {
    for (int i = 0; i < WINDOW_SIZE; i++) {
      axis_data[i] = raw_data[i][axis];
    }
    
    float mean = 0;
    for (int i = 0; i < WINDOW_SIZE; i++) {
      mean += axis_data[i];
    }
    mean /= WINDOW_SIZE;
    
    float std_dev = 0;
    for (int i = 0; i < WINDOW_SIZE; i++) {
      float diff = axis_data[i] - mean;
      std_dev += diff * diff;
    }
    std_dev = sqrt(std_dev / WINDOW_SIZE);
    
    if (std_dev < 0.0001f) {
      std_dev = 0.0001f;
    }
    
    for (int i = 0; i < WINDOW_SIZE; i++) {
      float normalized = (axis_data[i] - mean) / std_dev;
      
      if (normalized > 3.0f) normalized = 3.0f;
      if (normalized < -3.0f) normalized = -3.0f;
      
      int8_t quantized = static_cast<int8_t>(
          roundf(normalized / INPUT_SCALE + INPUT_ZERO_POINT)
      );
      
      if (quantized > 127) quantized = 127;
      if (quantized < -128) quantized = -128;
      
      int index = i * NUM_AXES + axis;
      quantized_data[index] = quantized;
    }
  }
}

// Выполнение инференса (предсказания)
int performInference() {
  static int8_t quantized_buffer[WINDOW_SIZE * NUM_AXES];
  
  preprocessData(data_buffer, quantized_buffer);
  
  int8_t* input_data = reinterpret_cast<int8_t*>(input_tensor->data.data);
  memcpy(input_data, quantized_buffer, sizeof(quantized_buffer));
  
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Ошибка выполнения инференса");
    return -1;
  }
  
  int8_t* output_data = reinterpret_cast<int8_t*>(output_tensor->data.data);
  
  float scores[NUM_CLASSES];
  if (output_tensor->quantization.type == kTfLiteAffineQuantization) {
    float output_scale = output_tensor->params.scale;
    int output_zero_point = output_tensor->params.zero_point;
    
    for (int i = 0; i < NUM_CLASSES; i++) {
      scores[i] = (output_data[i] - output_zero_point) * output_scale;
    }
  } else {
    float* float_output = reinterpret_cast<float*>(output_data);
    for (int i = 0; i < NUM_CLASSES; i++) {
      scores[i] = float_output[i];
    }
  }
  
  int predicted_class = 0;
  float max_score = scores[0];
  
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (scores[i] > max_score) {
      max_score = scores[i];
      predicted_class = i;
    }
  }
  
  if (max_score < 0.3f) {
    return -1;  
  }
  
  return predicted_class;
}

// Фильтр низких частот 
float applyLowPassFilter(float new_value, float filtered_value) {
  return ALPHA * new_value + (1.0f - ALPHA) * filtered_value;
}

// Инициализация датчика MPU6050
bool initMPU6050() {
  
  if (!mpu.begin()) {
    Serial.println("ERROR: Не удалось найти MPU6050");
    return false;
  }
  
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G); 
  mpu.setGyroRange(MPU6050_RANGE_250_DEG); 
  mpu.setFilterBandwidth(MPU6050_BAND_5_HZ); 
  
  Serial.println("MPU6050 успешно инициализирован");
  
  return true;
}

// Калибровка датчика 
void calibrateMPU6050() {
  delay(3000);
  
  const int CALIBRATION_SAMPLES = 150;  
  float sum_x = 0, sum_y = 0, sum_z = 0;
  
  for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    
    sum_x += a.acceleration.x;
    sum_y += a.acceleration.y;
    sum_z += a.acceleration.z;
    
    delay(20);  
    
    if (i % 30 == 0) {
      Serial.print(".");
    }
  }
  
  float offset_x = sum_x / CALIBRATION_SAMPLES;
  float offset_y = sum_y / CALIBRATION_SAMPLES;
  float offset_z = (sum_z / CALIBRATION_SAMPLES) - 9.81f; 
  
  Serial.println();
  Serial.println("Калибровка:");
  Serial.print("Смещение X: "); Serial.print(offset_x, 4); Serial.println(" м/с²");
  Serial.print("Смещение Y: "); Serial.print(offset_y, 4); Serial.println(" м/с²");
  Serial.print("Смещение Z: "); Serial.print(offset_z, 4); Serial.println(" м/с²");
}

// Чтение данных с датчика с фиксированной частотой
bool readSensorData() {
  unsigned long current_time = millis();
  
  if (current_time - last_sample_time < SAMPLE_INTERVAL) {
    return false;
  }
  
  last_sample_time = current_time;
  
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  
  filtered_x = applyLowPassFilter(a.acceleration.x, filtered_x);
  filtered_y = applyLowPassFilter(a.acceleration.y, filtered_y);
  filtered_z = applyLowPassFilter(a.acceleration.z, filtered_z);
  
  data_buffer[buffer_index][0] = filtered_x;
  data_buffer[buffer_index][1] = filtered_y;
  data_buffer[buffer_index][2] = filtered_z;
  
  buffer_index = (buffer_index + 1) % WINDOW_SIZE;
  samples_collected = min(samples_collected + 1, WINDOW_SIZE);
  
  return true;
}

// Настройка системы
void setup() {
  Serial.begin(9600);
  while (!Serial) {
    delay(10); 
  }
  
  Wire.begin();

  if (!initMPU6050()) {
    Serial.println("ERROR: Не удалось инициализировать датчик");
    while (1);  
  }
  
  if (!initTFLiteModel()) {
    Serial.println("ERROR: Не удалось инициализировать модель");
    while (1); 
  }
  
  while (samples_collected < WINDOW_SIZE) {
    if (readSensorData()) {
      if (samples_collected % 10 == 0) {
        Serial.print(".");
      }
    }
  }
  Serial.println("\nБуфер данных заполнен");
  
  Serial.println("Жесты:");
  Serial.println("0: Неподвижность");
  Serial.println("1: Взмах вверх");
  Serial.println("2: Взмах вниз");
  Serial.println("3: Круг по часовой стрелке");
  Serial.println("4: Стряхивание");
  Serial.println("5: Постукивание");
  
  last_sample_time = millis();
  last_prediction_time = millis();
}

void loop() {
  static int consecutive_predictions[NUM_CLASSES] = {0};
  static int prediction_buffer[5] = {-1, -1, -1, -1, -1};
  static int buffer_index_pred = 0;
  
  if (readSensorData()) {
    static int new_samples_count = 0;
    new_samples_count++;
    
    if (new_samples_count >= NEW_SAMPLES_NEEDED) {
      new_samples_count = 0;
      
      int predicted_gesture = performInference();
      
      if (predicted_gesture >= 0) {
        prediction_buffer[buffer_index_pred] = predicted_gesture;
        buffer_index_pred = (buffer_index_pred + 1) % 5;
        
        int counts[NUM_CLASSES] = {0};
        int max_count = 0;
        int final_prediction = -1;
        
        for (int i = 0; i < 5; i++) {
          if (prediction_buffer[i] >= 0) {
            counts[prediction_buffer[i]]++;
            if (counts[prediction_buffer[i]] > max_count) {
              max_count = counts[prediction_buffer[i]];
              final_prediction = prediction_buffer[i];
            }
          }
        }
        
        if (max_count >= 3 && final_prediction != last_predicted_gesture) {
          last_predicted_gesture = final_prediction;
          
          Serial.print("Жест распознан: ");
          Serial.print(final_prediction);
          Serial.print(" - ");
          Serial.println(GESTURE_NAMES[final_prediction]);
          
          if (Serial.available() && Serial.read() == 'd') {
            Serial.print("Сэмплов в буфере: ");
            Serial.println(samples_collected);
            
            Serial.println("Последние значения акселерометра:");
            for (int i = 0; i < 5; i++) {
              int idx = (buffer_index - 1 - i + WINDOW_SIZE) % WINDOW_SIZE;
              Serial.print("  [");
              Serial.print(data_buffer[idx][0], 2);
              Serial.print(", ");
              Serial.print(data_buffer[idx][1], 2);
              Serial.print(", ");
              Serial.print(data_buffer[idx][2], 2);
              Serial.println("]");
            }
          }
        }
      }
      
      static unsigned long last_status_time = 0;
      unsigned long current_time = millis();
      
      if (current_time - last_status_time > 10000) {
        last_status_time = current_time;
        
        float avg_magnitude = 0;
        for (int i = 0; i < WINDOW_SIZE; i++) {
          float mag = sqrt(
            data_buffer[i][0] * data_buffer[i][0] +
            data_buffer[i][1] * data_buffer[i][1] +
            data_buffer[i][2] * data_buffer[i][2]
          );
          avg_magnitude += mag;
        }
        avg_magnitude /= WINDOW_SIZE;
        
        Serial.print("Средняя магнитуда: ");
        Serial.print(avg_magnitude, 2);
        Serial.println(" м/с^2");
        
        if (avg_magnitude < 8.0f || avg_magnitude > 12.0f) {
          Serial.println("Возможна проблема с датчиком");
        }
      }
    }
  }
  
  if (Serial.available()) {
    char command = Serial.read();
    
    switch (command) {
      case 'r': 
        Serial.println("\nПерезапуск");
        delay(100);
        setup();
        break;
        
      case 'c': 
        calibrateMPU6050();
        break;
        
      case 's': 
        Serial.println("\nИнформация о системе:");
        Serial.print("Размер модели: ");
        Serial.print(sizeof(gesture_model) / 1024.0);
        Serial.println(" KB");
        Serial.print("Свободная память: ");
        Serial.print(freeMemory());
        Serial.println(" байт");
        Serial.print("Частота дискретизации: ");
        Serial.print(SAMPLE_RATE);
        Serial.println(" Гц");
        break;
        
      case 'h': 
        Serial.println("\nДоступные команды:");
        Serial.println("r - Перезапуск системы");
        Serial.println("c - Калибровка датчика");
        Serial.println("s - Статус системы");
        Serial.println("h - Помощь");
        Serial.println("d - Отладочная информация");
        break;
    }
  }
}

#ifdef __arv__
extern char __heap_start, *__brkval;
#endif

int freeMemory() {
#ifdef __arv__
  char top;
  return __brkval ? &top - __brkval : &top - &__heap_start;
#endif
}

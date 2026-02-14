/*
 ════════════════════════════════════════════════════════════════
  ADAR V3.0 — ESP32 Multi-Sensor Firmware
  
  All 5 sensors on ONE ESP32, sending data to the Fleet Server.
  
  SENSORS & WIRING:
  ─────────────────────────────────────────────────────────────
  1. MQ-135 (CO₂ Air Quality)
       VCC  → ESP32 VIN (5V)
       GND  → ESP32 GND
       AOUT → ESP32 GPIO 34  (ADC1_CH6)

  2. MQ-3 (Alcohol Detection)
       VCC  → ESP32 VIN (5V)
       GND  → ESP32 GND
       AOUT → ESP32 GPIO 35  (ADC1_CH7)

  3. MPU6050 (Accelerometer + Gyroscope — Speed & G-Force)
       VCC  → ESP32 3.3V
       GND  → ESP32 GND
       SDA  → ESP32 GPIO 21
       SCL  → ESP32 GPIO 22

  4. MAX30100 (Heart Rate + SpO2)
       VCC  → ESP32 3.3V
       GND  → ESP32 GND
       SDA  → ESP32 GPIO 21  (shared I2C bus with MPU6050)
       SCL  → ESP32 GPIO 22  (shared I2C bus with MPU6050)
       INT  → ESP32 GPIO 19  (optional interrupt pin)

  5. C4001 mmWave Human Presence Sensor (24GHz)
       VCC  → ESP32 3.3V  (or 5V — check your module)
       GND  → ESP32 GND
       TX   → ESP32 GPIO 16 (UART2 RX)
       RX   → ESP32 GPIO 17 (UART2 TX)

  NOTE: MPU6050 (addr 0x68) and MAX30100 (addr 0x57) share
        the same I2C bus — no conflict, different addresses.

  Upload via Arduino IDE:
    Board: "ESP32 Dev Module"
    Upload Speed: 921600
    Flash Size: 4MB
    
  Required Libraries (install via Arduino Library Manager):
    - Adafruit MPU6050
    - MAX30100lib (by OXullo Intersecans)
    - Wire (built-in)
    - WiFi (built-in)
    - HTTPClient (built-in)
 ════════════════════════════════════════════════════════════════
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>

// ── Try to include sensor libraries (comment out if not installed) ──
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "MAX30100_PulseOximeter.h"

// ═══════════════════════════════════════════
//  CONFIGURATION — CHANGE THESE VALUES
// ═══════════════════════════════════════════

// WiFi credentials
const char* WIFI_SSID     = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// ADAR Fleet Server address (your laptop's IP on the same WiFi)
// Find your IP: open CMD → type "ipconfig" → look for IPv4 Address
// Example: "http://192.168.1.100:8000"
const char* SERVER_BASE_URL = "http://192.168.1.100:8000";

// Vehicle ID (must match ADAR dashboard)
const char* VEHICLE_ID = "ADAR-01";

// How often to send each sensor type (milliseconds)
const int SEND_INTERVAL_GAS     = 1000;  // MQ-135 + MQ-3: every 1 second
const int SEND_INTERVAL_IMU     = 200;   // MPU6050: 5 times/second
const int SEND_INTERVAL_HEALTH  = 1000;  // MAX30100: every 1 second
const int SEND_INTERVAL_PRESENCE = 500;  // C4001: every 0.5 second

// ═══════════════════════════════════════════
//  PIN DEFINITIONS
// ═══════════════════════════════════════════
const int MQ135_PIN    = 34;  // ADC1_CH6 — CO₂
const int MQ3_PIN      = 35;  // ADC1_CH7 — Alcohol
const int I2C_SDA      = 21;  // Shared I2C SDA
const int I2C_SCL      = 22;  // Shared I2C SCL
const int MAX30100_INT = 19;  // MAX30100 interrupt (optional)
const int C4001_RX     = 16;  // UART2 RX (connects to C4001 TX)
const int C4001_TX     = 17;  // UART2 TX (connects to C4001 RX)

// ═══════════════════════════════════════════
//  MQ SENSOR CALIBRATION
// ═══════════════════════════════════════════
const float MQ135_R0 = 400.0;   // Clean air baseline ADC (calibrate!)
const float MQ3_R0   = 300.0;   // Clean air baseline ADC (calibrate!)

// ═══════════════════════════════════════════
//  SENSOR OBJECTS
// ═══════════════════════════════════════════
Adafruit_MPU6050 mpu;
PulseOximeter pox;

// ═══════════════════════════════════════════
//  SENSOR FLAGS (set to false if sensor not connected)
// ═══════════════════════════════════════════
bool hasMQ135    = true;
bool hasMQ3      = true;
bool hasMPU6050  = false;  // Set in setup() after init
bool hasMAX30100 = false;  // Set in setup() after init
bool hasC4001    = false;  // Set in setup() after init

// ═══════════════════════════════════════════
//  GLOBAL STATE
// ═══════════════════════════════════════════
bool wifiConnected = false;
unsigned long lastGasSend     = 0;
unsigned long lastIMUSend     = 0;
unsigned long lastHealthSend  = 0;
unsigned long lastPresenceSend = 0;

// MQ-135 smoothing
const int SMOOTH_N = 10;
float mq135Buf[SMOOTH_N]; int mq135Idx = 0; float mq135Sum = 0;
float mq3Buf[SMOOTH_N];   int mq3Idx = 0;   float mq3Sum = 0;
int readingCount = 0;

// MPU6050 speed integration
float imuSpeedMps = 0.0;
unsigned long lastIMUTime = 0;
float prevAx = 0, prevAy = 0;

// MAX30100 values
float lastHeartRate = 0;
float lastSpO2 = 0;
unsigned long lastBeatTime = 0;

// C4001 presence
bool humanPresent = false;
float presenceDistance = 0;
int presenceEnergy = 0;

// ═══════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    delay(100);
    
    Serial.println();
    Serial.println("════════════════════════════════════════");
    Serial.println("  ADAR V3.0 — Multi-Sensor Module");
    Serial.println("════════════════════════════════════════");
    
    // ── ADC for MQ sensors ──
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);
    pinMode(MQ135_PIN, INPUT);
    pinMode(MQ3_PIN, INPUT);
    for (int i = 0; i < SMOOTH_N; i++) { mq135Buf[i] = 0; mq3Buf[i] = 0; }
    Serial.println("  [✓] MQ-135 (CO₂) on GPIO 34");
    Serial.println("  [✓] MQ-3  (Alcohol) on GPIO 35");
    
    // ── I2C init ──
    Wire.begin(I2C_SDA, I2C_SCL);
    
    // ── MPU6050 ──
    if (mpu.begin(0x68, &Wire)) {
        hasMPU6050 = true;
        mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
        mpu.setGyroRange(MPU6050_RANGE_500_DEG);
        mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
        lastIMUTime = millis();
        Serial.println("  [✓] MPU6050 (IMU) at 0x68");
    } else {
        Serial.println("  [✗] MPU6050 not found — skipping");
    }
    
    // ── MAX30100 ──
    if (pox.begin()) {
        hasMAX30100 = true;
        pox.setIRLedCurrent(MAX30100_LED_CURR_7_6MA);
        pox.setOnBeatDetectedCallback(onBeatDetected);
        Serial.println("  [✓] MAX30100 (HR+SpO2) at 0x57");
    } else {
        Serial.println("  [✗] MAX30100 not found — skipping");
    }
    
    // ── C4001 mmWave (UART2) ──
    Serial2.begin(115200, SERIAL_8N1, C4001_RX, C4001_TX);
    delay(100);
    // Send a query command to check if sensor responds
    Serial2.write(0xFD);  // Common query byte for mmWave modules
    delay(200);
    if (Serial2.available() > 0) {
        hasC4001 = true;
        // Flush the response
        while (Serial2.available()) Serial2.read();
        Serial.println("  [✓] C4001 mmWave (24GHz) on UART2");
    } else {
        // Try DFRobot C4001 specific init
        Serial2.println("sensorStart");
        delay(300);
        if (Serial2.available() > 0) {
            hasC4001 = true;
            while (Serial2.available()) Serial2.read();
            Serial.println("  [✓] C4001 mmWave (24GHz) on UART2");
        } else {
            Serial.println("  [✗] C4001 mmWave not found — skipping");
        }
    }
    
    Serial.println("════════════════════════════════════════");
    Serial.printf("  Server: %s\n", SERVER_BASE_URL);
    Serial.printf("  Vehicle: %s\n", VEHICLE_ID);
    Serial.println("════════════════════════════════════════");
    Serial.println("  Warming up MQ sensors (2 min)...\n");
    
    // ── WiFi ──
    connectWiFi();
}

// ═══════════════════════════════════════════
//  MAIN LOOP
// ═══════════════════════════════════════════
void loop() {
    unsigned long now = millis();
    
    // Ensure WiFi stays connected
    if (WiFi.status() != WL_CONNECTED) {
        wifiConnected = false;
        connectWiFi();
    }
    
    // ── Always update MAX30100 (needs frequent polling) ──
    if (hasMAX30100) {
        pox.update();
    }
    
    // ── Read MQ sensors continuously for smoothing ──
    readMQSensors();
    
    // ── Read C4001 if data available ──
    if (hasC4001) {
        readC4001();
    }
    
    // ══ SEND: Gas Sensors (MQ-135 + MQ-3) ══
    if (now - lastGasSend >= SEND_INTERVAL_GAS) {
        lastGasSend = now;
        readingCount++;
        
        float co2 = adcToCO2(mq135Sum / SMOOTH_N);
        float alc = adcToAlcohol(mq3Sum / SMOOTH_N);
        
        Serial.printf("[GAS] CO₂=%.0f PPM  Alcohol=%.3f mg/L", co2, alc);
        
        if (wifiConnected) {
            bool ok = sendGasData(co2, mq135Sum / SMOOTH_N, alc, mq3Sum / SMOOTH_N);
            Serial.printf("  → %s\n", ok ? "SENT ✓" : "FAIL ✗");
        } else {
            Serial.println("  → No WiFi");
        }
    }
    
    // ══ SEND: IMU (MPU6050) ══
    if (hasMPU6050 && now - lastIMUSend >= SEND_INTERVAL_IMU) {
        lastIMUSend = now;
        readIMU(now);
    }
    
    // ══ SEND: Health (MAX30100) ══
    if (hasMAX30100 && now - lastHealthSend >= SEND_INTERVAL_HEALTH) {
        lastHealthSend = now;
        
        lastHeartRate = pox.getHeartRate();
        lastSpO2 = pox.getSpO2();
        
        // Only send valid readings
        if (lastHeartRate > 30 && lastHeartRate < 220 && lastSpO2 > 50) {
            Serial.printf("[HEALTH] HR=%.0f bpm  SpO₂=%.0f%%\n", lastHeartRate, lastSpO2);
            if (wifiConnected) sendHealthData(lastHeartRate, lastSpO2);
        }
    }
    
    // ══ SEND: Presence (C4001) ══
    if (hasC4001 && now - lastPresenceSend >= SEND_INTERVAL_PRESENCE) {
        lastPresenceSend = now;
        if (wifiConnected) sendPresenceData();
    }
    
    delay(10);  // Small yield
}

// ═══════════════════════════════════════════
//  MAX30100 BEAT CALLBACK
// ═══════════════════════════════════════════
void onBeatDetected() {
    lastBeatTime = millis();
}

// ═══════════════════════════════════════════
//  MQ SENSOR READING (smoothed)
// ═══════════════════════════════════════════
void readMQSensors() {
    // MQ-135
    float raw135 = analogRead(MQ135_PIN);
    mq135Sum -= mq135Buf[mq135Idx];
    mq135Buf[mq135Idx] = raw135;
    mq135Sum += raw135;
    mq135Idx = (mq135Idx + 1) % SMOOTH_N;
    
    // MQ-3
    float raw3 = analogRead(MQ3_PIN);
    mq3Sum -= mq3Buf[mq3Idx];
    mq3Buf[mq3Idx] = raw3;
    mq3Sum += raw3;
    mq3Idx = (mq3Idx + 1) % SMOOTH_N;
}

// ═══════════════════════════════════════════
//  ADC → PPM CONVERSIONS
// ═══════════════════════════════════════════
float adcToCO2(float adc) {
    if (adc < 10) return 350.0;
    float ratio = adc / MQ135_R0;
    float ppm = 400.0 * ratio * ratio;
    return constrain(ppm, 350.0, 5000.0);
}

float adcToAlcohol(float adc) {
    // MQ-3 alcohol sensor
    // Returns mg/L (BAC approximation)
    // Clean air: ~300 ADC.  Alcohol present: ADC increases.
    // Legal limit in India: 0.03% BAC ≈ 0.15 mg/L breath
    if (adc < 10) return 0.0;
    float ratio = adc / MQ3_R0;
    // Simplified: ratio 1.0 = clean air (0 mg/L)
    // ratio > 1.2 means alcohol detected
    float mgL = max(0.0f, (ratio - 1.0f) * 0.5f);
    return constrain(mgL, 0.0f, 3.0f);  // Cap at 3.0 mg/L
}

// ═══════════════════════════════════════════
//  IMU READING + SPEED INTEGRATION
// ═══════════════════════════════════════════
void readIMU(unsigned long now) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    
    float dt = (now - lastIMUTime) / 1000.0;
    lastIMUTime = now;
    
    // Simple trapezoidal speed integration (forward axis = X)
    // Remove gravity component (assuming sensor is roughly level)
    float ax = a.acceleration.x;
    float ay = a.acceleration.y;
    float az = a.acceleration.z - 9.81;  // Remove gravity on Z
    
    // Only integrate if acceleration is significant (deadband filter)
    float accelMag = sqrt(ax*ax + ay*ay);
    if (accelMag > 0.3) {
        imuSpeedMps += accelMag * dt;
    } else {
        // Decay speed when no acceleration (friction model)
        imuSpeedMps *= 0.98;
    }
    if (imuSpeedMps < 0.1) imuSpeedMps = 0;
    
    float gForce = sqrt(a.acceleration.x*a.acceleration.x + 
                        a.acceleration.y*a.acceleration.y + 
                        a.acceleration.z*a.acceleration.z) / 9.81;
    
    // Send to server
    if (wifiConnected && readingCount % 5 == 0) {  // Send every 5th reading
        sendIMUData(ax, ay, az, g.gyro.x, g.gyro.y, g.gyro.z, 
                    imuSpeedMps * 3.6, gForce);
    }
}

// ═══════════════════════════════════════════
//  C4001 mmWave READING
// ═══════════════════════════════════════════
void readC4001() {
    // DFRobot C4001 sends data frames via UART
    // Protocol varies by firmware — this handles common formats
    while (Serial2.available() >= 4) {
        // Try to read a complete frame
        uint8_t header = Serial2.read();
        
        if (header == 0xAA || header == 0xF4) {
            // Standard mmWave presence frame
            uint8_t buf[32];
            buf[0] = header;
            int len = Serial2.readBytes(&buf[1], min(31, Serial2.available()));
            
            // Parse presence status (simplified — adjust for your C4001 firmware)
            if (len >= 3) {
                humanPresent = (buf[1] & 0x01) != 0;
                presenceDistance = buf[2] * 0.1;  // Distance in meters (approx)
                presenceEnergy = (len >= 4) ? buf[3] : 0;
            }
        } else {
            // Try text-based protocol (some C4001 modules output text)
            String line = String((char)header) + Serial2.readStringUntil('\n');
            line.trim();
            // Parse: "$JYBSS,1,0.5,100" → present, 0.5m, energy 100
            if (line.startsWith("$JYBSS")) {
                int idx1 = line.indexOf(',');
                int idx2 = line.indexOf(',', idx1+1);
                int idx3 = line.indexOf(',', idx2+1);
                if (idx1 > 0 && idx2 > 0) {
                    humanPresent = line.substring(idx1+1, idx2).toInt() == 1;
                    if (idx3 > 0) {
                        presenceDistance = line.substring(idx2+1, idx3).toFloat();
                        presenceEnergy = line.substring(idx3+1).toInt();
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════
//  HTTP SENDERS
// ═══════════════════════════════════════════

bool sendGasData(float co2, float rawCo2Adc, float alcohol, float rawAlcAdc) {
    HTTPClient http;
    String url = String(SERVER_BASE_URL) + "/api/sensor";
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(3000);
    
    String json = "{";
    json += "\"co2_ppm\":" + String(co2, 1) + ",";
    json += "\"raw_adc\":" + String(rawCo2Adc, 0) + ",";
    json += "\"alcohol_mgl\":" + String(alcohol, 3) + ",";
    json += "\"alcohol_raw_adc\":" + String(rawAlcAdc, 0) + ",";
    json += "\"vehicle_id\":\"" + String(VEHICLE_ID) + "\",";
    json += "\"sensor\":\"MQ-135+MQ-3\",";
    json += "\"reading\":" + String(readingCount);
    json += "}";
    
    int code = http.POST(json);
    http.end();
    return (code == 200);
}

bool sendIMUData(float ax, float ay, float az, float gx, float gy, float gz,
                 float speedKmh, float gForce) {
    HTTPClient http;
    String url = String(SERVER_BASE_URL) + "/api/imu";
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(3000);
    
    String json = "{";
    json += "\"ax\":" + String(ax, 2) + ",";
    json += "\"ay\":" + String(ay, 2) + ",";
    json += "\"az\":" + String(az, 2) + ",";
    json += "\"gx\":" + String(gx, 2) + ",";
    json += "\"gy\":" + String(gy, 2) + ",";
    json += "\"gz\":" + String(gz, 2) + ",";
    json += "\"speed_kmh\":" + String(speedKmh, 1) + ",";
    json += "\"g_force\":" + String(gForce, 2) + ",";
    json += "\"vehicle_id\":\"" + String(VEHICLE_ID) + "\"";
    json += "}";
    
    int code = http.POST(json);
    http.end();
    return (code == 200);
}

bool sendHealthData(float heartRate, float spo2) {
    HTTPClient http;
    String url = String(SERVER_BASE_URL) + "/api/health";
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(3000);
    
    String json = "{";
    json += "\"heart_rate\":" + String(heartRate, 0) + ",";
    json += "\"spo2\":" + String(spo2, 0) + ",";
    json += "\"vehicle_id\":\"" + String(VEHICLE_ID) + "\",";
    json += "\"sensor\":\"MAX30100\"";
    json += "}";
    
    int code = http.POST(json);
    http.end();
    return (code == 200);
}

bool sendPresenceData() {
    HTTPClient http;
    String url = String(SERVER_BASE_URL) + "/api/presence";
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(3000);
    
    String json = "{";
    json += "\"present\":" + String(humanPresent ? "true" : "false") + ",";
    json += "\"distance\":" + String(presenceDistance, 2) + ",";
    json += "\"energy\":" + String(presenceEnergy) + ",";
    json += "\"vehicle_id\":\"" + String(VEHICLE_ID) + "\",";
    json += "\"sensor\":\"C4001\"";
    json += "}";
    
    int code = http.POST(json);
    http.end();
    return (code == 200);
}

// ═══════════════════════════════════════════
//  WiFi
// ═══════════════════════════════════════════
void connectWiFi() {
    Serial.printf("\n[WiFi] Connecting to %s", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 40) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        wifiConnected = true;
        Serial.printf("\n[WiFi] Connected! IP: %s\n", WiFi.localIP().toString().c_str());
        Serial.printf("[WiFi] Signal: %d dBm\n", WiFi.RSSI());
    } else {
        wifiConnected = false;
        Serial.println("\n[WiFi] FAILED — will retry...");
    }
}

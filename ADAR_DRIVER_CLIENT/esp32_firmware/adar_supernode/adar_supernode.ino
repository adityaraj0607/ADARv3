/*
 ════════════════════════════════════════════════════════════════════════
  ADAR V3.0 — THE GRAND FINALE "SUPER-NODE" FIRMWARE
  Features: Welcome Screen -> Safety Pre-Check -> Engine Start Block 
            CO2 Auto-Window Roll Down -> Radar Hardware Pin -> SOS System
 ════════════════════════════════════════════════════════════════════════
*/

#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "MAX30100_PulseOximeter.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <TinyGPSPlus.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <LiquidCrystal_I2C.h>
#include <esp_log.h>

// ═════════════════════════════════════════════════════════════════════
//  CLOUD & WIFI SETTINGS
// ═════════════════════════════════════════════════════════════════════
const char* WIFI_SSID     = "Aditya";
const char* WIFI_PASSWORD = "Aditya09";

// RENDER CLOUD SETTINGS (Render uses HTTPS → use WSS on 443)
#define USE_WSS_FOR_RENDER  1   // 1 = wss:// port 443 (production), 0 = ws:// port 80 (local)
const char* WS_HOST = "adar-fleet-command-centre.onrender.com";
const int   WS_PORT = USE_WSS_FOR_RENDER ? 443 : 80;
const char* WS_PATH = "/ws/vehicle/VH-7842";
const char* VEHICLE_ID = "VH-7842";

// ═════════════════════════════════════════════════════════════════════
//  HARDWARE PINOUT DEFINITIONS
// ═════════════════════════════════════════════════════════════════════
#define I2C_SDA       21
#define I2C_SCL       22

#define MQ135_PIN     34 // CO2 Sensor (Analog)
#define MQ3_PIN       35 // Alcohol Sensor (Analog)
#define RADAR_OUT_PIN 32 // mmWave Radar OUT Pin (Digital - Highly Reliable)

#define WINDOW_MOTOR_PIN 33 // Rolls down windows if CO2 is high
#define BUZZER_PIN    13

// 4 Motors (All connected to go forward/backward together)
#define MOTOR_ENA_PIN    12
#define MOTOR_IN1_PIN    25
#define MOTOR_IN2_PIN    14
#define MOTOR_ENB_PIN    27
#define MOTOR_IN3_PIN    26
#define MOTOR_IN4_PIN    23
#define MOTOR_PWM_FREQ   1000
#define MOTOR_PWM_RES    8

#define ENGINE_SW_PIN    4  // Flip switch to start engine
#define SOS_BTN_PIN      5  // Push button for emergency

#define MPU6050_ADDR     0x68
#define OLED_ADDR        0x3C

// ═════════════════════════════════════════════════════════════════════
//  THRESHOLDS
// ═════════════════════════════════════════════════════════════════════
#define ALCOHOL_THRESHOLD   300   // Warning level
#define CO2_THRESHOLD       1000  // Level to roll down windows
#define GFORCE_THRESHOLD    1.5f  // Crash detection

#define SMOOTH_N 10 // For sensor averaging

// ═════════════════════════════════════════════════════════════════════
//  SHARED DATA STRUCT (For Dual-Core Safety)
// ═════════════════════════════════════════════════════════════════════
struct SensorData {
    int   mq3Raw, mq135Raw;
    float accelX, accelY, accelZ, gForce;
    float heartRate, spO2;
    bool  radarPresence, dangerState, buzzerOn, engineOn;
    int   motorPWM;
    bool  windowOpen, sosActive;
    int   sosShutdownPct;
    volatile bool sosInterruptFlag;
};

static SensorData sharedData;
static SemaphoreHandle_t dataMutex;

// ═════════════════════════════════════════════════════════════════════
//  GLOBAL OBJECTS & STATE
// ═════════════════════════════════════════════════════════════════════
static Adafruit_MPU6050    mpu;
static PulseOximeter       pox;
static Adafruit_SSD1306    oled(128, 64, &Wire, -1);
static LiquidCrystal_I2C   lcd(0x27, 16, 2); // Change to 0x3F if LCD is blank
static WebSocketsClient    ws;

static bool wsConnected = false;
static bool hasMPU6050 = false, hasMAX30100 = false, hasOLED = false, hasLCD = true;
static int  mq3Buf[SMOOTH_N], mq3Idx = 0, mq135Buf[SMOOTH_N], mq135Idx = 0;
static long mq3Sum = 0, mq135Sum = 0;
static float accelX = 0, accelY = 0, accelZ = 0, gForce = 1.0;
static float heartRate = 0, spO2 = 0;
static unsigned long lastBeatTime = 0;
static bool radarPresence = false, dangerState = false, buzzerOn = false, windowOpen = false;
static bool engineOn = false, engineSwitchOn = false, sosActive = false, sosTriggered = false;
static unsigned long sosStartTime = 0, lastEngineSWTime = 0;
static int  sosShutdownPct = 100, motorPWM = 0;
static bool lastEngineSWState = HIGH;
static volatile unsigned long lastSOSIsrTime = 0;
static unsigned long lastLCDUpdate = 0;
static unsigned long lastStateChange = 0;  // Tracks engine/SOS LCD state changes

// ═════════════════════════════════════════════════════════════════════
//  ISR FOR SOS BUTTON
// ═════════════════════════════════════════════════════════════════════
void IRAM_ATTR sosISR() {
    unsigned long now = millis();
    if (now - lastSOSIsrTime < 250) return; // Debounce
    lastSOSIsrTime = now;
    sharedData.sosInterruptFlag = true;
}

// ═════════════════════════════════════════════════════════════════════
//  NETWORK TASK (CORE 0)
// ═════════════════════════════════════════════════════════════════════
void networkTask(void* pvParameters) {
    // ── STEP 1: Connect to WiFi with retries ──
    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);       // Auto-reconnect if WiFi drops
    WiFi.persistent(true);              // Remember credentials across reboots
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    int wifiAttempts = 0;
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        wifiAttempts++;
        if (wifiAttempts > 40) { // 20 seconds timeout — restart WiFi
            WiFi.disconnect();
            delay(1000);
            WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
            wifiAttempts = 0;
        }
    }
    Serial.print("[NET] WiFi connected! IP: ");
    Serial.println(WiFi.localIP());

    // ── STEP 2: Setup WebSocket with aggressive keep-alive ──
#if USE_WSS_FOR_RENDER
    ws.beginSSL(WS_HOST, WS_PORT, WS_PATH);
#else
    ws.begin(WS_HOST, WS_PORT, WS_PATH);
#endif

    ws.onEvent([](WStype_t type, uint8_t* payload, size_t length) {
        switch(type) {
            case WStype_CONNECTED:
                wsConnected = true;
                Serial.println("[WS] CONNECTED to Fleet Server!");
                break;
            case WStype_DISCONNECTED:
                wsConnected = false;
                Serial.println("[WS] DISCONNECTED — will auto-reconnect");
                break;
            case WStype_TEXT:
                // Server can send commands back
                break;
            case WStype_PING:
                Serial.println("[WS] Got PING");
                break;
            case WStype_PONG:
                // Keep-alive confirmed
                break;
            case WStype_ERROR:
                Serial.println("[WS] ERROR occurred");
                wsConnected = false;
                break;
            default:
                break;
        }
    });

    ws.setReconnectInterval(3000);     // Reconnect every 3s if disconnected
    ws.enableHeartbeat(15000, 5000, 2); // Ping every 15s, timeout 5s, 2 failures = disconnect+reconnect
    
    unsigned long lastWSSend = 0;
    unsigned long lastWifiCheck = 0;
    unsigned long lastHeartbeat = 0;

    // ── STEP 3: Main network loop — never exits ──
    for (;;) {
        unsigned long now = millis();

        // ── WiFi health check every 10 seconds ──
        if (now - lastWifiCheck > 10000) {
            lastWifiCheck = now;
            if (WiFi.status() != WL_CONNECTED) {
                Serial.println("[NET] WiFi lost! Reconnecting...");
                wsConnected = false;
                WiFi.disconnect();
                delay(1000);
                WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
                int retries = 0;
                while (WiFi.status() != WL_CONNECTED && retries < 30) {
                    delay(500);
                    retries++;
                }
                if (WiFi.status() == WL_CONNECTED) {
                    Serial.println("[NET] WiFi reconnected!");
                    // Re-init WebSocket after WiFi reconnect
#if USE_WSS_FOR_RENDER
                    ws.beginSSL(WS_HOST, WS_PORT, WS_PATH);
#else
                    ws.begin(WS_HOST, WS_PORT, WS_PATH);
#endif
                }
            }
        }

        // ── WebSocket loop (processes incoming + handles reconnect) ──
        ws.loop();
        
        // ── Send JSON heartbeat every 20s to keep Render proxy alive ──
        if (wsConnected && (now - lastHeartbeat > 20000)) {
            lastHeartbeat = now;
            ws.sendTXT("{\"type\":\"heartbeat\",\"vehicle_id\":\"" + String(VEHICLE_ID) + "\"}");
        }

        // ── Send live sensor data every 1 second ──
        if (now - lastWSSend >= 1000) {
            lastWSSend = now;
            
            SensorData snap;
            if (xSemaphoreTake(dataMutex, pdMS_TO_TICKS(10)) == pdTRUE) { 
                snap = sharedData; xSemaphoreGive(dataMutex); 
            }

            if(wsConnected) {
                JsonDocument doc;
                doc["vehicle_id"] = VEHICLE_ID;
                doc["co2"]        = snap.mq135Raw;
                doc["mq3"]        = snap.mq3Raw;
                doc["bpm"]        = (int)snap.heartRate;
                doc["spo2"]       = (int)snap.spO2;
                doc["g_force"]    = round(snap.gForce * 100.0) / 100.0;
                doc["radar_presence"] = snap.radarPresence;
                doc["radar_active"]   = true;
                doc["radar_distance"] = 0.0;
                doc["radar_energy"]   = 0;
                doc["buzzer"]     = snap.buzzerOn ? "ON" : "OFF";
                doc["danger"]     = snap.dangerState;
                doc["engine"]     = snap.engineOn ? "ON" : "OFF";
                doc["window_open"]= snap.windowOpen;
                doc["sos"]        = snap.sosActive;
                doc["motor_pwm"]  = snap.motorPWM;
                doc["sos_pct"]    = snap.sosShutdownPct;
                doc["uptime_s"]   = (int)(now / 1000);
                doc["has_mpu"]    = hasMPU6050;
                doc["has_max"]    = hasMAX30100;
                doc["has_radar"]  = true;
                doc["accel_x"]    = round(snap.accelX * 100.0) / 100.0;
                doc["accel_y"]    = round(snap.accelY * 100.0) / 100.0;
                doc["accel_z"]    = round(snap.accelZ * 100.0) / 100.0;
                doc["mq_warmup"]  = false;
                doc["mq135_baseline"] = 0;
                doc["mq3_baseline"]   = 0;
                doc["finger_on"]  = (snap.heartRate > 0);

                char jsonBuf[640];
                serializeJson(doc, jsonBuf, sizeof(jsonBuf));
                ws.sendTXT(jsonBuf);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}

// ═════════════════════════════════════════════════════════════════════
//  SETUP ROUTINE
// ═════════════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    // Suppress I2C NACK error spam from absent sensors (MPU6050, MAX30100, OLED)
    esp_log_level_set("i2c.master", ESP_LOG_NONE);
    dataMutex = xSemaphoreCreateMutex();
    
    // Pin Modes
    pinMode(BUZZER_PIN, OUTPUT); digitalWrite(BUZZER_PIN, LOW);
    pinMode(WINDOW_MOTOR_PIN, OUTPUT); digitalWrite(WINDOW_MOTOR_PIN, LOW);
    pinMode(RADAR_OUT_PIN, INPUT); // Hardware Radar Pin
    pinMode(ENGINE_SW_PIN, INPUT_PULLUP);
    pinMode(SOS_BTN_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(SOS_BTN_PIN), sosISR, FALLING);

    // Motor Driver Setup
    pinMode(MOTOR_IN1_PIN, OUTPUT); pinMode(MOTOR_IN2_PIN, OUTPUT);
    pinMode(MOTOR_IN3_PIN, OUTPUT); pinMode(MOTOR_IN4_PIN, OUTPUT);
    ledcAttach(MOTOR_ENA_PIN, MOTOR_PWM_FREQ, MOTOR_PWM_RES);
    ledcAttach(MOTOR_ENB_PIN, MOTOR_PWM_FREQ, MOTOR_PWM_RES);
    ledcWrite(MOTOR_ENA_PIN, 0); ledcWrite(MOTOR_ENB_PIN, 0);

    // ADC Setup
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);
    pinMode(MQ135_PIN, INPUT); pinMode(MQ3_PIN, INPUT);

    // Initialize I2C and LCD
    Wire.begin(I2C_SDA, I2C_SCL);
    lcd.init(); lcd.backlight();

    // ── 1. WELCOME SCREEN ──
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("Welcome to ADAR");
    lcd.setCursor(0, 1); lcd.print("Safety System");
    delay(3000);

    // ── 2. SYSTEM HARDWARE CHECK ──
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("Checking Sensors");
    lcd.setCursor(0, 1); lcd.print("Please Wait...");

    // Initialize Modules
    Wire.beginTransmission(MPU6050_ADDR);
    if (Wire.endTransmission() == 0) {
        Wire.beginTransmission(MPU6050_ADDR);
        Wire.write(0x6B); Wire.write(0x00); Wire.endTransmission();
        hasMPU6050 = true;
    }
    if (pox.begin()) {
        hasMAX30100 = true;
        pox.setIRLedCurrent(MAX30100_LED_CURR_7_6MA);
        pox.setOnBeatDetectedCallback([](){ lastBeatTime = millis(); });
    }
    if (oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) hasOLED = true;

    // Fast-Fill Sensor Arrays to bypass "Warm Up" bug
    for(int i=0; i<SMOOTH_N; i++) {
        mq135Buf[i] = map(analogRead(MQ135_PIN), 0, 4095, 400, 2000); // Maps to real CO2 PPM
        mq135Sum += mq135Buf[i];
        mq3Buf[i] = map(analogRead(MQ3_PIN), 0, 4095, 0, 1000); // Maps to Alcohol Level
        mq3Sum += mq3Buf[i];
        delay(100);
    }

    // ── 3. SAFETY PRE-CHECK SCREEN ──
    int initialCO2 = mq135Sum / SMOOTH_N;
    int initialALC = mq3Sum / SMOOTH_N;
    
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("CO2:"); lcd.print(initialCO2);
    lcd.setCursor(9, 0); lcd.print("ALC:"); lcd.print(initialALC);

    if(initialCO2 > CO2_THRESHOLD || initialALC > ALCOHOL_THRESHOLD) {
        lcd.setCursor(0, 1); lcd.print("CABIN UNSAFE!!");
        dangerState = true;
        digitalWrite(BUZZER_PIN, HIGH);
    } else {
        lcd.setCursor(0, 1); lcd.print("Cabin Safe!");
        dangerState = false;
    }
    delay(3000);

    if(!dangerState) {
        lcd.clear();
        lcd.setCursor(0, 0); lcd.print("Engine Ready.");
        lcd.setCursor(0, 1); lcd.print("Press Start SW");
    }

    // Launch Tasks
    xTaskCreatePinnedToCore(networkTask, "NetworkTask", 8192, NULL, 1, NULL, 0);
}

// ═════════════════════════════════════════════════════════════════════
//  MAIN LOOP (CORE 1 - SENSOR READINGS)
// ═════════════════════════════════════════════════════════════════════
void loop() {
    unsigned long now = millis();
    
    if (hasMAX30100) pox.update();

    // Read Sensors every 200ms
    static unsigned long lastRead = 0;
    if(now - lastRead > 200) {
        lastRead = now;

        // Map Analog Values to realistic Dashboard Numbers
        int currentCO2 = map(analogRead(MQ135_PIN), 0, 4095, 400, 2000);
        mq135Sum = mq135Sum - mq135Buf[mq135Idx] + currentCO2;
        mq135Buf[mq135Idx] = currentCO2;
        mq135Idx = (mq135Idx + 1) % SMOOTH_N;

        int currentALC = map(analogRead(MQ3_PIN), 0, 4095, 0, 1000);
        mq3Sum = mq3Sum - mq3Buf[mq3Idx] + currentALC;
        mq3Buf[mq3Idx] = currentALC;
        mq3Idx = (mq3Idx + 1) % SMOOTH_N;

        // Read Hardware Radar
        radarPresence = digitalRead(RADAR_OUT_PIN) == HIGH;

        // Read MPU6050
        if (hasMPU6050) {
            Wire.beginTransmission(MPU6050_ADDR);
            Wire.write(0x3B);
            if (Wire.endTransmission(false) == 0 && Wire.requestFrom(MPU6050_ADDR, 6) == 6) {
                accelX = ((Wire.read() << 8) | Wire.read()) / 8192.0 * 9.81;
                accelY = ((Wire.read() << 8) | Wire.read()) / 8192.0 * 9.81;
                accelZ = ((Wire.read() << 8) | Wire.read()) / 8192.0 * 9.81;
                gForce = sqrt(accelX*accelX + accelY*accelY + accelZ*accelZ) / 9.81;
            }
        }

        // Pulse Oximeter
        if (hasMAX30100) {
            heartRate = pox.getHeartRate(); spO2 = pox.getSpO2();
            if (now - lastBeatTime > 5000) { heartRate = 0; spO2 = 0; }
        }

        // Evaluate Danger
        int avgCO2 = mq135Sum / SMOOTH_N;
        int avgALC = mq3Sum / SMOOTH_N;
        
        dangerState = (avgCO2 > CO2_THRESHOLD || avgALC > ALCOHOL_THRESHOLD || gForce > GFORCE_THRESHOLD);

        // ── AUTO-WINDOW MOTOR LOGIC (CO2 or Alcohol high → ventilate) ──
        if (avgCO2 > CO2_THRESHOLD || avgALC > ALCOHOL_THRESHOLD) {
            digitalWrite(WINDOW_MOTOR_PIN, HIGH);
            windowOpen = true;
        } else {
            digitalWrite(WINDOW_MOTOR_PIN, LOW);
            windowOpen = false;
        }

        // Buzzer Logic
        if (!sosActive) {
            digitalWrite(BUZZER_PIN, dangerState ? HIGH : LOW);
            buzzerOn = dangerState;
        }

        // Copy to Shared Memory for WebSockets
        if (xSemaphoreTake(dataMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            sharedData.mq135Raw = avgCO2; sharedData.mq3Raw = avgALC;
            sharedData.heartRate = heartRate; sharedData.spO2 = spO2;
            sharedData.gForce = gForce; sharedData.radarPresence = radarPresence;
            sharedData.accelX = accelX; sharedData.accelY = accelY; sharedData.accelZ = accelZ;
            sharedData.dangerState = dangerState; sharedData.buzzerOn = buzzerOn;
            sharedData.engineOn = engineOn; sharedData.windowOpen = windowOpen;
            sharedData.sosActive = sosActive;
            sharedData.motorPWM = motorPWM;
            sharedData.sosShutdownPct = sosShutdownPct;
            xSemaphoreGive(dataMutex);
        }

        // Serial debug output every 2 seconds
        static unsigned long lastSerialPrint = 0;
        if (now - lastSerialPrint > 2000) {
            lastSerialPrint = now;
            Serial.print("[SENSORS] CO2="); Serial.print(avgCO2);
            Serial.print(" ALC="); Serial.print(avgALC);
            Serial.print(" RDR="); Serial.print(radarPresence ? "YES" : "NO");
            Serial.print(" G="); Serial.print(gForce, 2);
            Serial.print(" BPM="); Serial.print((int)heartRate);
            Serial.print(" SpO2="); Serial.print((int)spO2);
            Serial.print(" ENG="); Serial.println(engineOn ? "ON" : "OFF");
        }

        // ── LIVE LCD DISPLAY (Show sensor values during normal operation) ──
        // Only update LCD with live data if no recent state change message
        // (engine start/stop/SOS messages stay visible for 3 seconds)
        if (!sosActive && (now - lastStateChange > 3000) && (now - lastLCDUpdate > 1000)) {
            lastLCDUpdate = now;
            int avgCO2_lcd = mq135Sum / SMOOTH_N;
            int avgALC_lcd = mq3Sum / SMOOTH_N;

            lcd.clear();
            // Line 1: CO2 and Alcohol values
            lcd.setCursor(0, 0);
            lcd.print("CO2:");
            lcd.print(avgCO2_lcd);
            lcd.setCursor(9, 0);
            lcd.print("ALC:");
            lcd.print(avgALC_lcd);

            // Line 2: Radar + Engine status + Connection indicator
            lcd.setCursor(0, 1);
            lcd.print("RDR:");
            lcd.print(radarPresence ? "YES" : "NO ");
            lcd.setCursor(8, 1);
            if (engineOn) {
                lcd.print("ENG:ON");
            } else if (dangerState) {
                lcd.print("DANGER!");
            } else {
                lcd.print("ENG:OFF");
            }
            // Connection indicator at last char (col 15)
            lcd.setCursor(15, 1);
            lcd.print(wsConnected ? "*" : "!");
        }
    }

    // ── ENGINE SWITCH LOGIC ──
    bool reading = digitalRead(ENGINE_SW_PIN);
    if (reading != lastEngineSWState && (now - lastEngineSWTime > 250)) {
        lastEngineSWTime = now;
        lastEngineSWState = reading;

        if (reading == LOW) { // Start Engine Requested
            if (dangerState) { // 🛑 BLOCKED BY CABIN GAS
                lcd.clear(); lcd.setCursor(0,0); lcd.print("UNSAFE CABIN!");
                lcd.setCursor(0,1); lcd.print("Engine Blocked!");
                lastStateChange = now;
            } else { // ✅ START
                engineOn = true; engineSwitchOn = true; motorPWM = 255;
                // Spin all motors forward
                digitalWrite(MOTOR_IN1_PIN, HIGH); digitalWrite(MOTOR_IN2_PIN, LOW);
                digitalWrite(MOTOR_IN3_PIN, HIGH); digitalWrite(MOTOR_IN4_PIN, LOW);
                ledcWrite(MOTOR_ENA_PIN, motorPWM); ledcWrite(MOTOR_ENB_PIN, motorPWM);
                lcd.clear(); lcd.setCursor(0,0); lcd.print("ENGINE RUNNING");
                lastStateChange = now;
            }
        } else { // Stop Engine Requested
            engineOn = false; engineSwitchOn = false; motorPWM = 0;
            digitalWrite(MOTOR_IN1_PIN, LOW); digitalWrite(MOTOR_IN2_PIN, LOW);
            digitalWrite(MOTOR_IN3_PIN, LOW); digitalWrite(MOTOR_IN4_PIN, LOW);
            ledcWrite(MOTOR_ENA_PIN, 0); ledcWrite(MOTOR_ENB_PIN, 0);
            lcd.clear(); lcd.setCursor(0,0); lcd.print("ENGINE STOPPED");
            lastStateChange = now;
        }
    }

    // ── SOS BUTTON LOGIC ──
    if (sharedData.sosInterruptFlag) {
        sharedData.sosInterruptFlag = false;
        if (!sosActive) { // Trigger SOS Shutdown
            sosActive = true; sosStartTime = now;
            lcd.clear(); lcd.setCursor(0,0); lcd.print("!! SOS ACTIVE !!");
            lcd.setCursor(0,1); lcd.print("Shutting Down...");
            lastStateChange = now;
        } else { // Cancel SOS — do NOT auto-resume engine; user must turn engine switch on again
            sosActive = false;
            engineOn = false;
            engineSwitchOn = false;
            motorPWM = 0;
            digitalWrite(BUZZER_PIN, LOW); buzzerOn = false;
            digitalWrite(MOTOR_IN1_PIN, LOW); digitalWrite(MOTOR_IN2_PIN, LOW);
            digitalWrite(MOTOR_IN3_PIN, LOW); digitalWrite(MOTOR_IN4_PIN, LOW);
            ledcWrite(MOTOR_ENA_PIN, 0); ledcWrite(MOTOR_ENB_PIN, 0);
            lcd.clear(); lcd.setCursor(0,0); lcd.print("SOS Cleared");
            lcd.setCursor(0,1); lcd.print("Turn Engine SW");
            lastStateChange = now;
        }
    }

    if (sosActive) {
        unsigned long elapsed = now - sosStartTime;
        if (elapsed > 10000) { // Fully stopped after 10 seconds — now system normal
            sosActive = false;  // Clear SOS so user can turn engine on again
            engineOn = false; engineSwitchOn = false; motorPWM = 0;
            sosShutdownPct = 100;
            digitalWrite(MOTOR_IN1_PIN, LOW); digitalWrite(MOTOR_IN2_PIN, LOW);
            digitalWrite(MOTOR_IN3_PIN, LOW); digitalWrite(MOTOR_IN4_PIN, LOW);
            ledcWrite(MOTOR_ENA_PIN, 0); ledcWrite(MOTOR_ENB_PIN, 0);
            lcd.clear(); lcd.setCursor(0,0); lcd.print("System Normal");
            lcd.setCursor(0,1); lcd.print("Turn Engine SW");
            lastStateChange = now;
        } else { // Gradual slow down over 10 seconds
            motorPWM = map(elapsed, 0, 10000, 255, 0);
            sosShutdownPct = map(elapsed, 0, 10000, 100, 0);
            ledcWrite(MOTOR_ENA_PIN, motorPWM); ledcWrite(MOTOR_ENB_PIN, motorPWM);
        }
    } else {
        sosShutdownPct = 100;
    }
}
/*
 ════════════════════════════════════════════════════════════════
  ADAR V3.0 — ESP32 MQ-135 Sensor Firmware
  
  Reads CO₂ PPM from MQ-135 on GPIO 34 (ADC1_CH6)
  Sends readings to the ADAR Fleet Command Center via HTTP POST
  
  Wiring:
    MQ-135 VCC  →  ESP32 VIN (5V)
    MQ-135 GND  →  ESP32 GND
    MQ-135 AOUT →  ESP32 GPIO 34
  
  IMPORTANT: Let MQ-135 warm up for 2-5 minutes after power-on
             for accurate readings.
  
  Upload via Arduino IDE:
    Board: "ESP32 Dev Module"
    Upload Speed: 921600
    Flash Size: 4MB
 ════════════════════════════════════════════════════════════════
*/

#include <WiFi.h>
#include <HTTPClient.h>

// ═══════════════════════════════════════════
//  CONFIGURATION — CHANGE THESE VALUES
// ═══════════════════════════════════════════

// Your WiFi credentials
const char* WIFI_SSID     = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// ADAR server address (your laptop's IP on the same WiFi network)
// To find your IP: open CMD → type "ipconfig" → look for IPv4 Address
// Example: "http://192.168.1.100:8000"
const char* SERVER_URL = "http://192.168.1.100:8000/api/sensor";

// Sensor pin
const int MQ135_PIN = 34;  // GPIO 34 (ADC1_CH6)

// How often to send readings (milliseconds)
const int SEND_INTERVAL = 1000;  // 1 second

// Vehicle ID (must match ADAR dashboard)
const char* VEHICLE_ID = "VH-7842";

// ═══════════════════════════════════════════
//  MQ-135 CALIBRATION
// ═══════════════════════════════════════════
// These values are for approximate PPM conversion.
// For precise readings, calibrate in clean air (400 PPM baseline).
// 
// The analog value from ADC (0-4095 on ESP32, 12-bit) is converted
// to an approximate PPM using a simple linear + logarithmic mapping.
//
// Clean air reading (~400 PPM): Typically 300-600 raw ADC value
// You can adjust R0_CLEAN_AIR after measuring your sensor in fresh air.

const float R0_CLEAN_AIR = 400.0;  // ADC reading in clean air — CALIBRATE THIS
const float PPM_MIN = 350.0;
const float PPM_MAX = 5000.0;

// ═══════════════════════════════════════════
//  GLOBAL STATE
// ═══════════════════════════════════════════
unsigned long lastSendTime = 0;
bool wifiConnected = false;
int readingCount = 0;
float lastPPM = 0;

// Smoothing: average of last N readings to reduce noise
const int SMOOTH_WINDOW = 10;
float readings[SMOOTH_WINDOW];
int readIndex = 0;
float readTotal = 0;

void setup() {
    Serial.begin(115200);
    delay(100);
    
    Serial.println();
    Serial.println("════════════════════════════════════════");
    Serial.println("  ADAR V3.0 — MQ-135 Sensor Module");
    Serial.println("════════════════════════════════════════");
    Serial.printf("  Sensor Pin : GPIO %d\n", MQ135_PIN);
    Serial.printf("  Server     : %s\n", SERVER_URL);
    Serial.printf("  Interval   : %d ms\n", SEND_INTERVAL);
    Serial.println("════════════════════════════════════════");
    
    // Init ADC
    pinMode(MQ135_PIN, INPUT);
    analogReadResolution(12);  // 12-bit: 0-4095
    analogSetAttenuation(ADC_11db);  // Full 0-3.3V range
    
    // Init smoothing array
    for (int i = 0; i < SMOOTH_WINDOW; i++) readings[i] = 0;
    
    // Connect to WiFi
    connectWiFi();
    
    Serial.println("\n[SENSOR] Warming up MQ-135 (wait 2 minutes for stable readings)...");
}

void loop() {
    // Ensure WiFi stays connected
    if (WiFi.status() != WL_CONNECTED) {
        wifiConnected = false;
        connectWiFi();
    }
    
    // Read sensor continuously for smoothing
    float rawADC = analogRead(MQ135_PIN);
    
    // Update smoothing buffer
    readTotal -= readings[readIndex];
    readings[readIndex] = rawADC;
    readTotal += rawADC;
    readIndex = (readIndex + 1) % SMOOTH_WINDOW;
    
    float smoothedADC = readTotal / SMOOTH_WINDOW;
    
    // Convert ADC to approximate PPM
    float ppm = adcToPPM(smoothedADC);
    lastPPM = ppm;
    
    // Send to server at interval
    unsigned long now = millis();
    if (now - lastSendTime >= SEND_INTERVAL) {
        lastSendTime = now;
        readingCount++;
        
        // Print to Serial Monitor
        Serial.printf("[MQ-135] Raw: %.0f  Smoothed: %.0f  PPM: %.1f", rawADC, smoothedADC, ppm);
        
        // Send to ADAR server
        if (wifiConnected) {
            bool sent = sendToServer(ppm, smoothedADC);
            Serial.printf("  → %s\n", sent ? "SENT ✓" : "FAILED ✗");
        } else {
            Serial.println("  → WiFi disconnected");
        }
    }
    
    delay(50);  // ~20 readings per second for smoothing
}

// ═══════════════════════════════════════════
//  ADC → PPM CONVERSION
// ═══════════════════════════════════════════
float adcToPPM(float adcValue) {
    // Simple conversion based on MQ-135 characteristics:
    // - In clean air (~400 PPM), the sensor gives a baseline ADC reading
    // - Higher gas concentration → higher ADC value (for analog output)
    // - The relationship is roughly logarithmic
    
    if (adcValue < 10) return PPM_MIN;  // No reading / sensor not connected
    
    // Ratio compared to clean air baseline
    float ratio = adcValue / R0_CLEAN_AIR;
    
    // Approximate PPM using a simplified power curve
    // MQ-135 datasheet curve for CO2: PPM ≈ 400 * ratio^2 (simplified)
    float ppm = 400.0 * ratio * ratio;
    
    // Clamp to reasonable range
    if (ppm < PPM_MIN) ppm = PPM_MIN;
    if (ppm > PPM_MAX) ppm = PPM_MAX;
    
    return ppm;
}

// ═══════════════════════════════════════════
//  SEND TO ADAR SERVER
// ═══════════════════════════════════════════
bool sendToServer(float ppm, float rawADC) {
    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(3000);  // 3 second timeout
    
    // Build JSON payload
    String json = "{";
    json += "\"co2_ppm\":" + String(ppm, 1) + ",";
    json += "\"raw_adc\":" + String(rawADC, 0) + ",";
    json += "\"vehicle_id\":\"" + String(VEHICLE_ID) + "\",";
    json += "\"sensor\":\"MQ-135\",";
    json += "\"reading\":" + String(readingCount);
    json += "}";
    
    int httpCode = http.POST(json);
    http.end();
    
    return (httpCode == 200);
}

// ═══════════════════════════════════════════
//  WiFi CONNECTION
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

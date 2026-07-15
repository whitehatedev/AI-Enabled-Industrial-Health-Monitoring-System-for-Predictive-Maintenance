// esp8266_acs712_firebase.ino
#include <ESP8266WiFi.h>
#include <Firebase_ESP_Client.h>
#include <ACS712.h>   // Rob Tillaart's library
#include <time.h>     // for NTP

// WiFi Credentials
#define WIFI_SSID "Airtel_sahi_0849"
#define WIFI_PASSWORD "air99772"

// Firebase Credentials
#define API_KEY "AIzaSyDRK3k7DJ1NmGATWMjcKUmzYiVcxYDsOIQ"
#define DATABASE_URL "https://project-67b08-default-rtdb.firebaseio.com"
#define USER_EMAIL "sb284160@gmail.com"
#define USER_PASSWORD "Password@1"

// ACS712 30A pin
#define CURRENT_PIN A0
#define VREF 3.3
#define MAX_ADC 1023
#define SENSITIVITY_MV_PER_A 66   // 30A module

ACS712 sensor(CURRENT_PIN, VREF, MAX_ADC, SENSITIVITY_MV_PER_A);

// Calibration
float zeroOffset = 0;
float deadband = 0.05;               // 0.05A deadband → show 0.00A for no load

// Firebase objects
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

unsigned long lastUpdateTime = 0;
unsigned long lastHistoryTime = 0;
const unsigned long UPDATE_INTERVAL = 2000;
const unsigned long HISTORY_INTERVAL = 10000;

bool ntpSynced = false;

// ---------- NTP setup ----------
void setupNTP() {
    configTime(0, 0, "pool.ntp.org", "time.nist.gov");
    Serial.print("Waiting for NTP");
    time_t now = time(nullptr);
    int attempts = 0;
    while (now < 8 * 3600 * 2 && attempts < 20) {
        delay(500);
        Serial.print(".");
        now = time(nullptr);
        attempts++;
    }
    if (now > 8 * 3600 * 2) {
        ntpSynced = true;
        Serial.println(" done.");
    } else {
        Serial.println(" failed.");
        ntpSynced = false;
    }
}

// ---------- Timestamp helpers ----------
unsigned long getTimestamp() {
    time_t now = time(nullptr);
    if (ntpSynced && now > 1577836800UL) {   // after 2020
        return (unsigned long)now * 1000UL;
    } else {
        // Fallback: use millis() + base epoch (Jan 1, 2026 00:00:00 UTC)
        const unsigned long BASE_EPOCH_MS = 1740000000000UL;
        return BASE_EPOCH_MS + millis();
    }
}

String formatDateTime(unsigned long epoch_ms) {
    time_t epoch_sec = epoch_ms / 1000;
    struct tm* timeinfo = gmtime(&epoch_sec);
    char buffer[30];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
    return String(buffer);
}

// ---------- Setup ----------
void setup() {
    Serial.begin(115200);
    Serial.println("\n⚡ ACS712 Current Sensor (30A)");

    // Calibrate zero offset
    Serial.println("Calibrating... Ensure no load is connected!");
    delay(2000);
    float sum = 0;
    for (int i = 0; i < 100; i++) {
        sum += analogRead(CURRENT_PIN);
        delay(10);
    }
    zeroOffset = sum / 100.0;
    sensor.setMidPoint(zeroOffset);
    Serial.printf("✅ Zero offset ADC: %.0f\n", zeroOffset);

    // WiFi
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) { Serial.print("."); delay(500); }
    Serial.println("\n✅ WiFi connected. IP: " + WiFi.localIP().toString());

    setupNTP();

    // Firebase
    config.api_key = API_KEY;
    config.database_url = DATABASE_URL;
    auth.user.email = USER_EMAIL;
    auth.user.password = USER_PASSWORD;
    Firebase.begin(&config, &auth);
    Firebase.reconnectWiFi(true);

    delay(1000);
    if (Firebase.ready()) {
        Serial.println("✅ Firebase authenticated!");
        // Test write to /test_current
        if (Firebase.RTDB.setString(&fbdo, "/test_current", "ok")) {
            Serial.println("✅ Test write succeeded.");
        } else {
            Serial.println("❌ Test write FAILED: " + fbdo.errorReason());
        }
    } else {
        Serial.println("❌ Firebase auth FAILED.");
    }
}

// ---------- Read current ----------
float readCurrent() {
    float current_mA = sensor.mA_AC(50, 1);
    float current_A = current_mA / 1000.0;
    if (abs(current_A) < deadband) current_A = 0;
    return current_A;
}

// ---------- Loop ----------
void loop() {
    if (!Firebase.ready()) { delay(1000); return; }
    if (millis() - lastUpdateTime >= UPDATE_INTERVAL) {
        sendLatestData();
        lastUpdateTime = millis();
    }
    if (millis() - lastHistoryTime >= HISTORY_INTERVAL) {
        sendHistoryData();
        lastHistoryTime = millis();
    }
}

// ---------- Send latest ----------
void sendLatestData() {
    float current = readCurrent();
    unsigned long ts = getTimestamp();
    String datetime = formatDateTime(ts);

    FirebaseJson json;
    json.set("value", current);
    json.set("unit", "A");
    json.set("timestamp", ts);
    json.set("datetime", datetime);

    String path = "/machines/machine_01/devices/current/latest";
    if (Firebase.RTDB.setJSON(&fbdo, path, &json)) {
        Serial.printf("⚡ %.2f A | %s → sent\n", current, datetime.c_str());
    } else {
        Serial.println("❌ Failed: " + fbdo.errorReason());
    }
}

// ---------- Send history ----------
void sendHistoryData() {
    float current = readCurrent();
    unsigned long ts = getTimestamp();
    String datetime = formatDateTime(ts);

    FirebaseJson json;
    json.set("value", current);
    json.set("unit", "A");
    json.set("timestamp", ts);
    json.set("datetime", datetime);

    String path = "/machines/machine_01/devices/current/history/" + String(ts);
    if (Firebase.RTDB.setJSON(&fbdo, path, &json)) {
        Serial.println("📝 History saved");
    } else {
        Serial.println("❌ Failed to save history: " + fbdo.errorReason());
    }
}
#include "WiFi.h"
#include "esp_camera.h"
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "soc/soc.h" // Disable brownout problems
#include "soc/rtc_cntl_reg.h" // Disable brownout problems
#include "driver/rtc_io.h"
#include <HTTPClient.h>

// OV2640 camera module pins (CAMERA_MODEL_AI_THINKER)
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
#define FLASH_GPIO_NUM 4

// Button pin
#define BUTTON_PIN 0

// WiFi credentials
const char* ssid = "Abc";
const char* password = "11111111";

// API endpoint URL
const char* apiUrl = "http://18.220.104.120/transcribe";

void initCamera() {
  // Turn off the 'brownout detector'
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  // OV2640 camera module
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // Camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    ESP.restart();
  }
}

void capturePhotoUploadAPI() {
  camera_fb_t *fb = NULL;
  fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  HTTPClient http;

  // Begin the HTTP POST request
  http.begin(apiUrl);

  // Create a unique boundary string
  String boundary = "------------------------" + String(millis());

  // Set the content type header with the boundary
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

  // Start the multipart/form-data
  String body = "--" + boundary + "\r\n";
  body += "Content-Disposition: form-data; name=\"image\"; filename=\"image.jpg\"\r\n";
  body += "Content-Type: image/jpeg\r\n\r\n";

  // Calculate content length
  int contentLength = body.length() + fb->len + 6 + boundary.length() + 4;

  // Add the image data to the request body
  http.addHeader("Content-Length", String(contentLength));
  int httpCode = http.POST((uint8_t*)body.c_str(), body.length());

  if (httpCode > 0) {
    // Write the image data to the request body
    http.write(fb->buf, fb->len);

    // End the multipart/form-data
    String endBoundary = "\r\n--" + boundary + "--\r\n";
    http.write((uint8_t*)endBoundary.c_str(), endBoundary.length());

    // Get the response code
    int httpResponseCode = http.POST((uint8_t*)endBoundary.c_str(), endBoundary.length());

    if (httpResponseCode == 200) {
      Serial.println("Photo uploaded successfully");
    } else {
      Serial.print("Photo upload failed with error code: ");
      Serial.println(httpResponseCode);
    }
  } else {
    Serial.print("Failed to connect, error: ");
    Serial.println(http.errorToString(httpCode));
  }

  // End the request
  http.end();

  // Return the camera frame buffer
  esp_camera_fb_return(fb);
}

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP); // Initialize the button pin
  initCamera();

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
}

void loop() {
  if (digitalRead(BUTTON_PIN) == LOW) { // Check if the button is pressed
    Serial.println("Button pressed, taking photo...");
    capturePhotoUploadAPI();
    delay(1000); // Debounce delay
  }
}

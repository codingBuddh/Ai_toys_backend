# Text-to-Speech Integration for ESP32 Audio Server

This guide explains how to set up and use the Text-to-Speech (TTS) functionality with your ESP32 audio server.

## Server-Side Setup

### 1. Google Cloud TTS Authentication

There are several ways to authenticate with Google Cloud TTS:

#### Option 1: Service Account JSON File

1. Create a Google Cloud account if you don't have one
2. Create a new project in Google Cloud Console
3. Enable the Cloud Text-to-Speech API
4. Create a service account key:
   - Go to IAM & Admin > Service Accounts
   - Create a new service account
   - Grant it the "Cloud Text-to-Speech User" role
   - Create and download a JSON key file

5. Save the JSON key file as `google_credentials.json` in your project root directory
   - The server will automatically look for this file by default
   - You can change the file path using the `TTS_CREDENTIALS_FILE` environment variable

#### Option 2: Environment Variable

Set the environment variable to point to your credentials file:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-credentials-file.json"
```

#### Option 3: Application Default Credentials

If you're running the server on Google Cloud (like Cloud Run, GCE, etc.), you can use Application Default Credentials, which are automatically available.

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. Configure TTS Settings

You can configure TTS settings using environment variables:

```bash
# Enable or disable TTS (set in code)
# ENABLE_TTS=True

# Path to credentials file (if not using default location)
export TTS_CREDENTIALS_FILE="/path/to/google_credentials.json"

# Default voice to use
export TTS_DEFAULT_VOICE="en-US-Wavenet-D"

# Default language code
export TTS_DEFAULT_LANGUAGE="en-US"

# Whether to save audio files (for debugging)
export TTS_SAVE_AUDIO="false"
```

### 4. Start the Server

```bash
python -m app.main
```

## ESP32 Client Setup

### 1. Required Libraries

Install the following libraries in Arduino IDE:

- WebSocketsClient by Links2004
- ArduinoJson
- ESP8266Audio
- ESP8266_Spiram (dependency of ESP8266Audio)

### 2. Hardware Connections

#### I2S Speaker Setup (MAX98357A or similar)
- Connect LRC to GPIO22
- Connect BCLK to GPIO27
- Connect DIN to GPIO26
- Connect GND to GND
- Connect VIN to 3.3V

#### I2S Microphone Setup (INMP441 or similar)
- Connect SCK to GPIO32
- Connect WS to GPIO15
- Connect SD to GPIO33
- Connect GND to GND
- Connect VDD to 3.3V

### 3. Configure the ESP32 Code

1. Update WiFi credentials:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```

2. Update server details:
   ```cpp
   const char* wsHost = "192.168.1.100";  // Your server IP
   const int wsPort = 5000;
   ```

3. Upload the code to your ESP32

## Using the TTS Functionality

### Option 1: Combined Audio + TTS Client

The ESP32 client provides both audio recording and TTS playback capabilities:

1. Open the Serial Monitor (115200 baud)
2. Send commands:
   - `start` - Start recording audio
   - `stop` - Stop recording and process audio
   - The LLM response will automatically be converted to speech and played

### Option 2: TTS-Only Client

For a dedicated TTS client:

1. Open the Serial Monitor (115200 baud)
2. Type any text and press Enter
3. The text will be sent to the server, converted to speech, and played back

### Changing Voices

You can change the TTS voice by sending special commands:

```
/voice:en-US-Wavenet-F
/language:en-US
/voices  (lists available voices)
```

Available voices include:
- en-US-Wavenet-A (male)
- en-US-Wavenet-B (male)
- en-US-Wavenet-C (female)
- en-US-Wavenet-D (male)
- en-US-Wavenet-E (female)
- en-US-Wavenet-F (female)

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Check that your credentials file is valid and has the correct permissions
   - Verify the service account has the "Cloud Text-to-Speech User" role
   - Make sure the Text-to-Speech API is enabled in your Google Cloud project
   - Check server logs for specific authentication errors

2. **No sound from speaker**
   - Check I2S connections
   - Verify the gain setting in the code (default 0.5)
   - Try different GPIO pins if your hardware differs

3. **WebSocket connection failures**
   - Verify server IP address
   - Ensure server is running
   - Check WiFi connectivity

4. **TTS not working**
   - Verify Google Cloud credentials are set correctly
   - Check server logs for TTS-related errors
   - Ensure credentials file exists and is readable

5. **MP3 playback issues**
   - Check if the ESP32 has enough memory
   - Try reducing buffer sizes if memory is limited

### Checking Server Logs

Monitor the server logs to see TTS processing:

```bash
tail -f server.log
tail -f tts_processor.log
```

### Testing Authentication

You can test your Google Cloud credentials with this command:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-credentials-file.json"
python -c "from google.cloud import texttospeech; client = texttospeech.TextToSpeechClient(); voices = client.list_voices(); print(f'Authentication successful! Found {len(voices.voices)} voices.')"
```

## Advanced Configuration

You can modify the following parameters in `app/main.py`:

```python
# Configuration options
VERBOSE_LOGGING = False  # Set to True for detailed logging
ENABLE_TTS = True        # Set to False to disable TTS

# TTS Configuration
TTS_CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None)
TTS_CREDENTIALS_FILE = os.environ.get("TTS_CREDENTIALS_FILE", "google_credentials.json")
TTS_DEFAULT_VOICE = os.environ.get("TTS_DEFAULT_VOICE", "en-US-Wavenet-D")
TTS_DEFAULT_LANGUAGE = os.environ.get("TTS_DEFAULT_LANGUAGE", "en-US")
TTS_SAVE_AUDIO = os.environ.get("TTS_SAVE_AUDIO", "false").lower() == "true"
```

## API Endpoints

- `/ws/audio` - WebSocket endpoint for audio streaming
- `/ws/tts` - WebSocket endpoint for TTS requests
- `/health` - Health check endpoint (returns TTS status)
- `/` - Root endpoint with server information 
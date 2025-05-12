# ESP32 WebSocket Audio Streaming & Whisper STT Server

This project provides a real-time audio streaming server for ESP32 devices that captures audio via WebSocket and performs speech-to-text transcription using OpenAI's Whisper models.

## Features

- WebSocket server for receiving audio streams from ESP32 devices
- Real-time conversion of audio streams to WAV format
- Speech-to-text processing using Whisper models (CPU-only)
- Configurable model selection (tiny, base, small, medium, large)
- Comprehensive benchmarking tools for model comparison
- Performance metrics tracking

## Project Structure

- `app/`: Main application code
  - `main.py`: FastAPI server with WebSocket endpoint
  - `utils/`: Utility functions including audio processing
- `recordings/`: Storage for audio recordings from ESP32
- `transcripts/`: Storage for generated transcripts
- `testing_STT/`: Comprehensive testing and benchmarking tools for Whisper models

## Setup

### Using Docker (Recommended)

The easiest way to run this project is using Docker:

1. Install [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)

2. Build and run the container:

```bash
docker-compose up -d
```

The server will start at `http://localhost:5000` and WebSocket endpoint will be available at `ws://localhost:5000/ws/audio`.

3. View logs:

```bash
docker-compose logs -f
```

4. Stop the server:

```bash
docker-compose down
```

### Manual Setup

1. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required dependencies:

```bash
pip install --upgrade pip setuptools setuptools-rust
pip install -r requirements.txt
```

3. Run the server:

```bash
python -m app.main
```

The server will start at `http://localhost:5000` and WebSocket endpoint will be available at `ws://localhost:5000/ws/audio`.

## Testing Whisper Models

This project includes comprehensive tools for testing Whisper STT models:

```bash
# Run the test script for a single model
cd testing_STT
./test_model.sh -m small

# Run a full benchmark of all models
./run_benchmark.sh
```

See the [testing_STT/README.md](testing_STT/README.md) for more details on the testing capabilities.

## ESP32 Configuration

Configure your ESP32 to connect to the WebSocket server and stream audio data. The ESP32 should:

1. Connect to the WebSocket endpoint at `ws://[server-ip]:5000/ws/audio`
2. Format audio data with a 4-byte chunk ID header followed by PCM audio data
3. Send audio with the following configuration:
   - Sample Rate: 16000 Hz
   - Channels: 1 (Mono)
   - Bits per Sample: 16

## Requirements

- Python 3.8 or higher
- FFmpeg installed on your system
- Rust (for tiktoken, a dependency of Whisper)
- Docker (optional, for containerized deployment)

## Server Deployment Notes

- This server is configured to use CPU-only processing for better compatibility with resource-constrained servers
- The transformers library dependency has been removed to reduce the installation footprint
- If you're deploying on a server with limited memory, this version should be more stable
- Docker deployment isolates dependencies and ensures consistent environment across different systems

## License

[MIT License](LICENSE)

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition models
- [FastAPI](https://fastapi.tiangolo.com/) for the WebSocket server 
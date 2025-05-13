# Troubleshooting Guide

This guide provides solutions to common issues you might encounter when using the ESP32 WebSocket Audio Streaming Server.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Server Startup Issues](#server-startup-issues)
3. [WebSocket Connection Issues](#websocket-connection-issues)
4. [Audio Processing Issues](#audio-processing-issues)
5. [Whisper STT Issues](#whisper-stt-issues)
6. [LLM Integration Issues](#llm-integration-issues)
7. [Docker Issues](#docker-issues)
8. [Log Files](#log-files)
9. [Performance Troubleshooting](#performance-troubleshooting)

## Installation Issues

### Python Dependencies

**Issue**: Error installing dependencies from requirements.txt

**Solution**:
- Make sure you have Python 3.8 or higher installed
- Install build dependencies: `apt-get install -y build-essential python3-dev`
- Try installing dependencies one by one to identify the problematic package
- For PyTorch issues, use the CPU-only version: `pip install torch --extra-index-url https://download.pytorch.org/whl/cpu`

### FFmpeg Missing

**Issue**: "FFmpeg not found" or "Could not read audio file" errors

**Solution**:
- Install FFmpeg:
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- Verify installation: `ffmpeg -version`

### Memory Issues

**Issue**: Out of memory errors during installation or model loading

**Solution**:
- Use a smaller Whisper model (tiny or base)
- Set `WHISPER_MODEL=tiny` in your .env file
- Increase swap space on your server
- Use the CPU-only version of PyTorch

## Server Startup Issues

### Port Already in Use

**Issue**: "Address already in use" error when starting the server

**Solution**:
- Change the port in your .env file: `PORT=5001`
- Find and stop the process using the port: `sudo lsof -i :5000`
- Kill the process: `kill -9 <PID>`

### Module Not Found

**Issue**: "ModuleNotFoundError" when starting the server

**Solution**:
- Make sure you're in the project root directory
- Activate your virtual environment: `source venv/bin/activate`
- Install missing dependencies: `pip install -r requirements.txt`
- If using Docker, rebuild the image: `docker-compose build`

### Environment Variables

**Issue**: Server can't find environment variables

**Solution**:
- Copy .env.example to .env: `cp .env.example .env`
- Edit .env with your API keys and configuration
- If using Docker, make sure the .env file is in the same directory as docker-compose.yml
- Verify environment variables are loaded: `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.environ.get('OPENAI_API_KEY'))"`

## WebSocket Connection Issues

### Connection Refused

**Issue**: Client can't connect to the WebSocket server

**Solution**:
- Verify the server is running: `curl http://localhost:5000/health`
- Check firewall settings: `sudo ufw status`
- Allow the port: `sudo ufw allow 5000`
- If using Docker, make sure ports are mapped correctly in docker-compose.yml
- Check the server logs for errors: `tail -f server.log`

### Connection Drops

**Issue**: WebSocket connection drops unexpectedly

**Solution**:
- Increase client reconnection attempts
- Check network stability
- Look for errors in server.log
- Increase the WebSocket ping interval and timeout in your client code

## Audio Processing Issues

### Invalid Audio Format

**Issue**: "Error processing audio" or "Invalid audio data"

**Solution**:
- Verify audio format matches the expected configuration:
  - Sample Rate: 16000 Hz
  - Channels: 1 (Mono)
  - Bits per Sample: 16
- Check that the chunk ID header is correctly formatted (4 bytes, big-endian)
- Examine the WAV files in the recordings directory to verify they're valid

### Audio Not Saving

**Issue**: Audio files not appearing in recordings directory

**Solution**:
- Check directory permissions: `ls -la recordings/`
- Make sure the directory exists: `mkdir -p recordings`
- If using Docker, verify volume mounting: `docker-compose config`

## Whisper STT Issues

### Model Loading Errors

**Issue**: "Error loading Whisper model" or related errors

**Solution**:
- Check whisper_processor.log for specific errors
- Try a smaller model: `WHISPER_MODEL=tiny`
- Verify you have enough disk space for model files
- If using Docker, increase container memory limits

### Transcription Errors

**Issue**: Empty or incorrect transcriptions

**Solution**:
- Check audio quality and format
- Try a larger Whisper model for better accuracy
- Look for specific errors in whisper_processor.log
- Verify the audio file exists and is not corrupted

## LLM Integration Issues

### API Key Issues

**Issue**: "API key not configured" or authentication errors

**Solution**:
- Verify your OpenAI API key in the .env file
- Check for billing issues on your OpenAI account
- Look for specific errors in llm_processor.log
- Test your API key with a simple curl request

### Slow Responses

**Issue**: LLM responses taking too long

**Solution**:
- Try a faster model: `LLM_MODEL=gpt-3.5-turbo`
- Check your internet connection
- Look at response times in the logs to identify bottlenecks
- Consider implementing caching for common queries

## Docker Issues

### Container Won't Start

**Issue**: Docker container fails to start

**Solution**:
- Check container logs: `docker-compose logs`
- Verify .env file is present and properly formatted
- Rebuild the container: `docker-compose build --no-cache`
- Check for port conflicts: `docker ps -a`

### Volume Mount Issues

**Issue**: Data not persisting or files not accessible

**Solution**:
- Check volume configuration in docker-compose.yml
- Verify directory permissions: `sudo chown -R $(id -u):$(id -g) recordings/ transcripts/ responses/`
- Use absolute paths in docker-compose.yml
- Inspect volumes: `docker volume ls` and `docker volume inspect <volume-name>`

## Log Files

The following log files can help diagnose issues:

- **server.log**: Main server logs
- **whisper_processor.log**: Whisper STT processing logs
- **llm_processor.log**: LLM processing logs
- **test_llm.log**: LLM test logs

View logs with:
```bash
tail -f server.log
```

## Performance Troubleshooting

### High CPU Usage

**Issue**: Server using too much CPU

**Solution**:
- Use a smaller Whisper model
- Reduce concurrent connections
- Monitor with `htop` or `docker stats`
- Consider upgrading your hardware or using a cloud instance with more CPU

### High Memory Usage

**Issue**: Server using too much memory

**Solution**:
- Use a smaller Whisper model: `WHISPER_MODEL=tiny`
- Use the CPU-only version of PyTorch
- Increase swap space
- Monitor with `free -m` or `docker stats`

### Slow Processing

**Issue**: Audio processing taking too long

**Solution**:
- Check real-time factor in logs
- Use a smaller, faster Whisper model
- Optimize audio chunk size
- Consider batch processing for non-real-time applications

If you encounter issues not covered in this guide, please check the detailed logs and report the issue with:
- Log files
- Steps to reproduce
- Environment details (OS, Python version, etc.)
- Error messages 
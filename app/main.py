"""
ESP32 WebSocket Audio Streaming Server with Whisper STT and LLM Integration

This module implements a FastAPI server that:
1. Receives audio streams from ESP32 devices via WebSocket
2. Converts audio to WAV format
3. Processes audio with Whisper for speech-to-text transcription
4. Processes transcripts with LLM for intelligent responses
5. Sends responses back to the client

Debug Info:
- Check server.log for detailed logging
- Check llm_processor.log for LLM-specific logging
- All operations are timed and logged
- Audio files are saved in recordings/ directory
- Transcripts are saved in transcripts/ directory
- LLM responses are saved in responses/ directory
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import os
import struct
import time
from datetime import datetime
from pathlib import Path
import asyncio
import logging
import traceback
import json

from app.utils.audio_processor import process_audio_with_whisper
from app.utils.llm_processor import LLMProcessor
from app.utils.tts_processor import TTSProcessor

# Configuration options
VERBOSE_LOGGING = False  # Set to True for detailed chunk-by-chunk logging
ENABLE_TTS = True        # Set to False to disable Text-to-Speech functionality

# TTS Configuration
TTS_CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None)
TTS_CREDENTIALS_FILE = os.environ.get("TTS_CREDENTIALS_FILE", "google_credentials.json")
TTS_DEFAULT_VOICE = os.environ.get("TTS_DEFAULT_VOICE", "en-US-Wavenet-D")
TTS_DEFAULT_LANGUAGE = os.environ.get("TTS_DEFAULT_LANGUAGE", "en-US")
TTS_SAVE_AUDIO = os.environ.get("TTS_SAVE_AUDIO", "false").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("server.log")
    ]
)
logger = logging.getLogger("audio_server")

app = FastAPI(
    title="ESP32 Audio Streaming Server",
    description="WebSocket server for audio streaming, transcription, and LLM processing",
    version="1.0.0",
)

# Audio config
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
BITS_PER_SAMPLE = 16

# Directory setup
RECORDINGS_DIR = Path("recordings")
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR = Path("transcripts")
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
RESPONSES_DIR = Path("responses")
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

# Initialize LLM processor
llm_processor = None
try:
    llm_processor = LLMProcessor()
    logger.info("LLM Processor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM Processor: {e}")
    logger.error(traceback.format_exc())

# Initialize TTS processor if enabled
tts_processor = None
if ENABLE_TTS:
    try:
        # Try loading credentials from specified file first
        credentials_json = None
        if TTS_CREDENTIALS_FILE and os.path.exists(TTS_CREDENTIALS_FILE):
            try:
                with open(TTS_CREDENTIALS_FILE, "r") as f:
                    credentials_json = f.read()
                logger.info(f"Loaded TTS credentials from {TTS_CREDENTIALS_FILE}")
            except Exception as e:
                logger.error(f"Failed to read TTS credentials file: {e}")
        
        # Initialize TTS processor with credentials
        tts_processor = TTSProcessor(
            credentials_path=TTS_CREDENTIALS_PATH,
            credentials_json=credentials_json
        )
        
        # List available voices if in verbose mode
        if VERBOSE_LOGGING:
            voices = TTSProcessor.get_available_voices(tts_processor.client)
            logger.info(f"Available TTS voices: {', '.join(voices[:10])}{'...' if len(voices) > 10 else ''}")
        
        logger.info("TTS Processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize TTS Processor: {e}")
        logger.error(traceback.format_exc())

# Print startup message
logger.info(f"Server starting...")
logger.info(f"Recordings will be saved to: {RECORDINGS_DIR.absolute()}")
logger.info(f"Transcripts will be saved to: {TRANSCRIPTS_DIR.absolute()}")
logger.info(f"LLM Responses will be saved to: {RESPONSES_DIR.absolute()}")
if ENABLE_TTS:
    logger.info(f"TTS is enabled with voice: {TTS_DEFAULT_VOICE}")
    logger.info(f"TTS language: {TTS_DEFAULT_LANGUAGE}")
else:
    logger.info(f"TTS is disabled")

# Dictionary to track active WebSocket streams
active_streams = {}

def create_wav_header(data_length: int) -> bytes:
    """
    Create a WAV file header for the audio data.
    
    Args:
        data_length: Length of the audio data in bytes
        
    Returns:
        bytes: WAV header
        
    Debug Info:
        - Uses global audio config (SAMPLE_RATE, NUM_CHANNELS, BITS_PER_SAMPLE)
        - Creates standard RIFF/WAVE format header
    """
    byte_rate = SAMPLE_RATE * NUM_CHANNELS * BITS_PER_SAMPLE // 8
    block_align = NUM_CHANNELS * BITS_PER_SAMPLE // 8

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        data_length + 36,
        b'WAVE',
        b'fmt ',
        16,
        1,
        NUM_CHANNELS,
        SAMPLE_RATE,
        byte_rate,
        block_align,
        BITS_PER_SAMPLE,
        b'data',
        data_length
    )
    return header

async def save_to_wav(audio_data: bytes, chunk_id: int, client_id: str):
    """
    Save audio data to a WAV file, process with Whisper, and optionally process with LLM.
    
    Args:
        audio_data: Raw PCM audio data bytes
        chunk_id: ID of the current audio chunk
        client_id: ID of the client sending the audio
        
    Returns:
        tuple: (file_path, transcript, llm_response)
    """
    timestamp = datetime.utcnow().isoformat().replace(":", "-").replace(".", "-")
    file_name = f"recording_{client_id}_chunk{chunk_id}_{timestamp}.wav"
    file_path = RECORDINGS_DIR / file_name

    # Save WAV file with header
    header = create_wav_header(len(audio_data))
    with open(file_path, "wb") as f:
        f.write(header + audio_data)
    
    # Calculate audio duration
    audio_duration = len(audio_data) / (SAMPLE_RATE * NUM_CHANNELS * BITS_PER_SAMPLE / 8)
    
    transcript = ""
    llm_response = ""
    
    try:
        # Process audio with Whisper and save transcript
        transcript = await process_audio_with_whisper(str(file_path))
        
        if transcript:
            transcript_file = TRANSCRIPTS_DIR / f"transcript_{client_id}_chunk{chunk_id}_{timestamp}.txt"
            with open(transcript_file, "w") as f:
                f.write(transcript)
            
            # Process transcript with LLM if available
            if llm_processor:
                logger.info("Processing transcript with LLM...")
                
                # ==========================================
                # CUSTOMIZE YOUR LLM PROMPT HERE
                # ==========================================
                # This system prompt controls how the AI responds to the transcript
                # Modify this prompt to customize the AI's behavior and response style
                system_prompt = """
                You are a helpful assistant responding to spoken input. 
                The user is speaking to you through a speech-to-text system.
                
                Respond concisely in a SINGLE LINE only. Do not use multiple paragraphs or bullet points.
                Keep your response brief but informative.
                
                Be friendly and conversational in your response.
                If the user asks a question, provide a helpful answer.
                If the user gives a command, acknowledge it and respond appropriately.
                """
                # ==========================================
                # END OF CUSTOMIZABLE PROMPT
                # ==========================================
                
                # Process with LLM
                llm_result = await llm_processor.process_transcript(transcript, system_prompt)
                
                if llm_result.get("success"):
                    llm_response = llm_result["response"]
                    response_file = RESPONSES_DIR / f"response_{client_id}_chunk{chunk_id}_{timestamp}.txt"
                    with open(response_file, "w") as f:
                        f.write(llm_response)
                    if VERBOSE_LOGGING:
                        logger.info(f"LLM response generated: \"{llm_response[:50]}{'...' if len(llm_response) > 50 else ''}\"")
                else:
                    logger.error(f"LLM processing failed: {llm_result.get('error')}")
        else:
            logger.warning(f"No transcript generated")
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        logger.error(traceback.format_exc())
        transcript = ""
        llm_response = ""
    
    return file_path, transcript, llm_response

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for receiving audio streams from clients.
    
    This endpoint:
    1. Accepts WebSocket connections
    2. Receives audio data in chunks
    3. Processes audio data when a chunk is complete
    4. Sends transcription and LLM responses back to the client
    
    Debug Info:
        - Client connections are logged with unique IDs
        - Audio chunks are tracked and logged
        - All processing times are measured and logged
        - Errors are caught, logged, and reported to the client
        - Session summaries are generated on disconnect
    """
    await websocket.accept()
    
    # Set WebSocket ping interval and timeout for Google Cloud
    websocket._ping_interval = 20.0  # Send ping every 20 seconds
    websocket._ping_timeout = 60.0   # Wait 60 seconds for pong response
    
    client_id = f"{websocket.client.host}_{websocket.client.port}"
    connection_time = datetime.now().strftime('%H:%M:%S')
    logger.info(f"Client connected: {client_id} at {connection_time}")

    active_streams[client_id] = {
        "chunk_id": None,
        "audio_buffer": b"",
        "start_time": time.time(),
        "chunks_processed": 0,
        "total_audio_duration": 0,
        "accumulated_audio": b"",  # For accumulating all audio data
        "accumulated_chunks": 0,   # Count of accumulated chunks
    }

    try:
        while True:
            data = await websocket.receive_bytes()
            
            # Extract chunk ID from the first 4 bytes (big-endian unsigned int)
            chunk_id = struct.unpack(">I", data[:4])[0]
            
            # Check if this is the termination marker (0xFFFFFFFF)
            if chunk_id == 0xFFFFFFFF:
                current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                logger.info(f"[{current_time}] Processing audio from {client_id}")
                
                client_data = active_streams[client_id]
                if client_data["accumulated_audio"]:
                    process_start_time = time.time()
                    if VERBOSE_LOGGING:
                        logger.info(f"[{current_time}] --- Audio recording complete ({client_data['accumulated_chunks']} chunks). Creating transcript... ---")
                    
                    try:
                        # Create a timestamp for the combined file
                        timestamp = datetime.utcnow().isoformat().replace(":", "-").replace(".", "-")
                        combined_file_name = f"recording_{client_id}_combined_{timestamp}.wav"
                        combined_file_path = RECORDINGS_DIR / combined_file_name
                        
                        # Save all accumulated audio as a single WAV file
                        header = create_wav_header(len(client_data["accumulated_audio"]))
                        with open(combined_file_path, "wb") as f:
                            f.write(header + client_data["accumulated_audio"])
                        
                        # Calculate audio duration for the combined data
                        combined_duration = len(client_data["accumulated_audio"]) / (SAMPLE_RATE * NUM_CHANNELS * BITS_PER_SAMPLE / 8)
                        current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        if VERBOSE_LOGGING:
                            logger.info(f"[{current_time}] Saved combined WAV file: {combined_file_name} ({combined_duration:.2f} seconds)")
                        
                        # Process the combined audio file
                        current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        if VERBOSE_LOGGING:
                            logger.info(f"[{current_time}] Generating transcript from audio recording...")
                        wav_path, transcript, llm_response = await save_to_wav(client_data["accumulated_audio"], 0, client_id + "_combined")
                        
                        # Update client data
                        client_data["total_audio_duration"] += combined_duration
                        client_data["chunks_processed"] += client_data["accumulated_chunks"]
                        
                        process_time = time.time() - process_start_time
                        
                        # Prepare response to client
                        response_data = {
                            "combined_chunks": client_data["accumulated_chunks"],
                            "transcript": transcript,
                            "process_time": f"{process_time:.2f}s",
                            "llm_response": llm_response if llm_response else None,
                            "audio_duration": f"{combined_duration:.2f}s",
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Generate TTS audio if enabled
                        if ENABLE_TTS and tts_processor and llm_response:
                            try:
                                tts_start_time = time.time()
                                audio_content = await tts_processor.text_to_speech(
                                    text=llm_response,
                                    voice_name=TTS_DEFAULT_VOICE,
                                    language_code=TTS_DEFAULT_LANGUAGE,
                                    save_file=TTS_SAVE_AUDIO,
                                    client_id=client_id
                                )
                                tts_time = time.time() - tts_start_time
                                
                                if audio_content:
                                    # Include TTS audio info in response
                                    response_data["tts_audio"] = {
                                        "available": True,
                                        "size": len(audio_content),
                                        "format": "mp3",
                                        "voice": TTS_DEFAULT_VOICE,
                                        "language": TTS_DEFAULT_LANGUAGE,
                                        "process_time": f"{tts_time:.2f}s"
                                    }
                                    logger.info(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] TTS audio generated: {len(audio_content)/1024:.1f} KB")
                            except Exception as e:
                                logger.error(f"Error generating TTS audio: {e}")
                                response_data["tts_audio"] = {"available": False, "error": str(e)}
                        
                        # Send response to client with transcript and LLM response
                        await websocket.send_json(response_data)
                        
                        # Send TTS audio if available
                        if ENABLE_TTS and tts_processor and llm_response and audio_content:
                            await websocket.send_bytes(audio_content)
                            logger.info(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Sent TTS audio to {client_id}")
                        
                        # Reset accumulated data after processing
                        client_data["accumulated_audio"] = b""
                        client_data["accumulated_chunks"] = 0
                        
                        # Log the full transcript and LLM response
                        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Response: \"{llm_response}\"")
                        if VERBOSE_LOGGING:
                            logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Full Transcript: \"{transcript}\"")
                            logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Total processing time: {process_time:.2f}s")
                    except Exception as e:
                        logger.error(f"Error processing accumulated audio: {e}")
                        logger.error(traceback.format_exc())
                        await websocket.send_json({
                            "error": f"Error processing accumulated audio: {str(e)}",
                            "timestamp": datetime.now().isoformat()
                        })
                continue  # Skip the rest of the loop for termination marker
            
            audio_data = data[4:]
            client_data = active_streams[client_id]

            # If this is a new chunk ID, process the previous chunk
            if client_data["chunk_id"] != chunk_id:
                if client_data["chunk_id"] is not None and client_data["audio_buffer"]:
                    # Add current chunk to accumulated audio
                    client_data["accumulated_audio"] += client_data["audio_buffer"]
                    client_data["accumulated_chunks"] += 1
                    
                    # Only log the first chunk received and then when processing starts
                    if client_data["accumulated_chunks"] == 1:
                        current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        logger.info(f"[{current_time}] Started receiving audio from {client_id}")
                    elif VERBOSE_LOGGING and client_data["accumulated_chunks"] % 50 == 0:
                        # Only log milestone chunks if verbose logging is enabled
                        total_accumulated_kb = len(client_data["accumulated_audio"]) / 1024
                        current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        logger.info(f"[{current_time}] Accumulated: {total_accumulated_kb:.1f} KB from {client_data['accumulated_chunks']} chunks")
                
                # Start collecting a new chunk
                client_data["chunk_id"] = chunk_id
                client_data["audio_buffer"] = audio_data
                
                # Only log when recording starts
                if chunk_id == 0:
                    current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    logger.info(f"[{current_time}] Receiving audio stream from {client_id}")
                elif VERBOSE_LOGGING and chunk_id % 50 == 0:
                    # Log milestone chunks if verbose logging is enabled
                    current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    logger.info(f"[{current_time}] Chunk {chunk_id} received - {len(audio_data)/1024:.1f} KB")
            else:
                # Continue collecting data for the current chunk
                client_data["audio_buffer"] += audio_data
                
                # Log buffer size milestones only if verbose logging is enabled
                if VERBOSE_LOGGING:
                    buffer_size_kb = len(client_data["audio_buffer"]) / 1024
                    if int(buffer_size_kb / 100) > int((buffer_size_kb - len(audio_data)/1024) / 100):
                        current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        logger.info(f"[{current_time}] Chunk {chunk_id} buffer: {buffer_size_kb:.1f} KB")
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {client_id}")
        client_data = active_streams.get(client_id)
        if client_data:
            try:
                # Add any remaining data to accumulated audio
                if client_data["audio_buffer"]:
                    client_data["accumulated_audio"] += client_data["audio_buffer"]
                    client_data["accumulated_chunks"] += 1
                
                # Process any accumulated audio (process whatever we have, but log differently based on length)
                if client_data["accumulated_audio"]:
                    audio_duration = len(client_data["accumulated_audio"]) / (SAMPLE_RATE * NUM_CHANNELS * BITS_PER_SAMPLE / 8)
                    
                    if audio_duration >= 1.0:  # At least 1 second
                        logger.info(f"Processing final audio recording ({audio_duration:.2f} seconds)")
                        
                        try:
                            # Save and process accumulated audio
                            timestamp = datetime.utcnow().isoformat().replace(":", "-").replace(".", "-")
                            combined_file_name = f"recording_{client_id}_final_{timestamp}.wav"
                            combined_file_path = RECORDINGS_DIR / combined_file_name
                            
                            # Save all accumulated audio as a single WAV file
                            header = create_wav_header(len(client_data["accumulated_audio"]))
                            with open(combined_file_path, "wb") as f:
                                f.write(header + client_data["accumulated_audio"])
                            
                            if VERBOSE_LOGGING:
                                logger.info(f"Generating transcript from final audio...")
                            
                            # Process the combined audio
                            await save_to_wav(client_data["accumulated_audio"], 999, client_id + "_final")
                            
                            client_data["total_audio_duration"] += audio_duration
                            client_data["chunks_processed"] += client_data["accumulated_chunks"]
                        except Exception as e:
                            logger.error(f"Error processing final audio after disconnect: {e}")
                            logger.error(traceback.format_exc())
                    else:
                        if VERBOSE_LOGGING:
                            logger.info(f"Audio too short ({audio_duration:.2f}s), skipping transcription")
            except Exception as e:
                logger.error(f"Error processing final data after client disconnect: {e}")
                logger.error(traceback.format_exc())
            
            # Print simplified session summary
            session_duration = time.time() - client_data["start_time"]
            if VERBOSE_LOGGING:
                logger.info(f"Session ended: {client_data['chunks_processed']} chunks, {client_data['total_audio_duration']:.2f}s audio")
            else:
                logger.info(f"Session ended for {client_id}")
            
        active_streams.pop(client_id, None)
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        logger.error(traceback.format_exc())
        active_streams.pop(client_id, None)

@app.websocket("/ws/tts")
async def tts_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS audio responses to clients.
    
    This endpoint:
    1. Accepts WebSocket connections
    2. Receives text messages from clients
    3. Converts text to speech using Google Cloud TTS
    4. Streams audio data back to the client
    
    Debug Info:
        - Client connections are logged with unique IDs
        - TTS requests are logged
        - Audio responses are logged
        - Errors are caught and reported to the client
    """
    await websocket.accept()
    
    # Set WebSocket ping interval and timeout
    websocket._ping_interval = 20.0  # Send ping every 20 seconds
    websocket._ping_timeout = 60.0   # Wait 60 seconds for pong response
    
    client_id = f"{websocket.client.host}_{websocket.client.port}"
    connection_time = datetime.now().strftime('%H:%M:%S')
    logger.info(f"TTS client connected: {client_id} at {connection_time}")

    if not tts_processor:
        logger.error(f"TTS is not available. Closing connection with {client_id}")
        await websocket.close(code=1008, reason="TTS service not available")
        return

    # Store client-specific voice preferences
    client_voice = TTS_DEFAULT_VOICE
    client_language = TTS_DEFAULT_LANGUAGE

    try:
        while True:
            # Receive text message from client
            message = await websocket.receive_text()
            
            # Check for special commands
            if message.startswith("/voice:"):
                # Format: /voice:en-US-Wavenet-D
                voice_name = message[7:].strip()
                client_voice = voice_name
                await websocket.send_json({
                    "status": "voice_changed",
                    "voice": voice_name
                })
                logger.info(f"Voice changed to {voice_name} for {client_id}")
                continue
                
            elif message.startswith("/language:"):
                # Format: /language:en-US
                language_code = message[10:].strip()
                client_language = language_code
                await websocket.send_json({
                    "status": "language_changed",
                    "language": language_code
                })
                logger.info(f"Language changed to {language_code} for {client_id}")
                continue
                
            elif message.startswith("/voices"):
                # List available voices
                try:
                    voices = TTSProcessor.get_available_voices(tts_processor.client)
                    await websocket.send_json({
                        "status": "voices_list",
                        "voices": voices
                    })
                    logger.info(f"Sent list of {len(voices)} voices to {client_id}")
                except Exception as e:
                    logger.error(f"Error listing voices: {e}")
                    await websocket.send_json({
                        "status": "error",
                        "message": f"Error listing voices: {str(e)}"
                    })
                continue
                
            current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            logger.info(f"[{current_time}] TTS request from {client_id}: \"{message[:50]}{'...' if len(message) > 50 else ''}\"")
            
            # Process text to speech
            try:
                start_time = time.time()
                audio_content = await tts_processor.text_to_speech(
                    text=message,
                    voice_name=client_voice,
                    language_code=client_language,
                    save_file=TTS_SAVE_AUDIO,
                    client_id=client_id
                )
                process_time = time.time() - start_time
                
                if audio_content:
                    # Send audio size first so client knows what to expect
                    await websocket.send_json({
                        "status": "audio_ready",
                        "size": len(audio_content),
                        "format": "mp3",
                        "voice": client_voice,
                        "language": client_language,
                        "process_time": f"{process_time:.2f}s"
                    })
                    
                    # Then send the actual audio data
                    await websocket.send_bytes(audio_content)
                    logger.info(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Sent {len(audio_content)/1024:.1f} KB audio to {client_id}")
                else:
                    await websocket.send_json({
                        "status": "error",
                        "message": "Failed to generate audio"
                    })
            except Exception as e:
                logger.error(f"Error processing TTS request: {e}")
                logger.error(traceback.format_exc())
                await websocket.send_json({
                    "status": "error",
                    "message": f"Error processing TTS request: {str(e)}"
                })
                
    except WebSocketDisconnect:
        logger.info(f"TTS client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Error in TTS WebSocket connection: {e}")
        logger.error(traceback.format_exc())

@app.get("/")
async def get_root():
    """
    Root endpoint that returns basic server information.
    
    Returns:
        dict: Server status information
    """
    return {
        "message": "Audio Streaming Server is running. Connect to /ws/audio for WebSocket connection.",
        "status": "active",
        "endpoints": {
            "websocket": "/ws/audio",
            "health": "/health"
        },
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns:
        dict: Health status of the server components
    """
    whisper_status = "available"
    llm_status = "available" if llm_processor else "unavailable"
    tts_status = "available" if tts_processor else "unavailable"
    
    return {
        "status": "healthy",
        "components": {
            "server": "running",
            "whisper": whisper_status,
            "llm": llm_status,
            "tts": tts_status
        },
        "active_connections": len(active_streams),
        "uptime": time.time() - app.state.start_time if hasattr(app.state, "start_time") else 0
    }

@app.on_event("startup")
async def startup_event():
    """
    Startup event handler that runs when the server starts.
    """
    app.state.start_time = time.time()
    logger.info("Server started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler that runs when the server stops.
    """
    logger.info("Server shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True) 
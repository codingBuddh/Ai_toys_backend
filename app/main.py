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

# Print startup message
logger.info(f"Server starting...")
logger.info(f"Recordings will be saved to: {RECORDINGS_DIR.absolute()}")
logger.info(f"Transcripts will be saved to: {TRANSCRIPTS_DIR.absolute()}")
logger.info(f"LLM Responses will be saved to: {RESPONSES_DIR.absolute()}")

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
        
    Debug Info:
        - Audio duration is calculated based on data length and audio config
        - All operations are timed and logged
        - Whisper processing errors are caught and logged
        - LLM processing errors are caught and logged
        - File paths use timestamp format for uniqueness
    """
    timestamp = datetime.utcnow().isoformat().replace(":", "-").replace(".", "-")
    file_name = f"recording_{client_id}_chunk{chunk_id}_{timestamp}.wav"
    file_path = RECORDINGS_DIR / file_name

    # Measure timing for saving WAV file
    wav_start_time = time.time()
    header = create_wav_header(len(audio_data))
    with open(file_path, "wb") as f:
        f.write(header + audio_data)
    wav_save_time = time.time() - wav_start_time
    
    # Calculate audio duration
    audio_duration = len(audio_data) / (SAMPLE_RATE * NUM_CHANNELS * BITS_PER_SAMPLE / 8)
    
    logger.info(f"Saved WAV file: {file_name} ({len(audio_data)/1024:.1f} KB, {audio_duration:.2f} seconds)")
    logger.info(f"WAV save time: {wav_save_time:.3f} seconds")
    
    transcript = ""
    llm_response = ""
    
    try:
        # Process audio with Whisper and save transcript
        logger.info(f"Starting transcription at {datetime.now().strftime('%H:%M:%S')}")
        transcription_start_time = time.time()
        transcript = await process_audio_with_whisper(str(file_path))
        transcription_time = time.time() - transcription_start_time
        
        if transcript:
            transcript_file = TRANSCRIPTS_DIR / f"transcript_{client_id}_chunk{chunk_id}_{timestamp}.txt"
            with open(transcript_file, "w") as f:
                f.write(transcript)
            logger.info(f"Saved transcript: {transcript_file}")
            logger.info(f"Total transcription processing time: {transcription_time:.2f} seconds")
            
            # Calculate real-time factor
            rtf = transcription_time / audio_duration if audio_duration > 0 else float('inf')
            logger.info(f"Real-time factor: {rtf:.2f}x (lower is better)")
            
            # Process transcript with LLM if available
            if llm_processor:
                logger.info("Processing transcript with LLM...")
                llm_start_time = time.time()
                
                # System prompt for the LLM
                system_prompt = """
                You are a helpful assistant responding to spoken input. 
                The user is speaking to you through a speech-to-text system.
                Respond concisely and naturally as if in a conversation.
                """
                
                # Process with LLM
                llm_result = await llm_processor.process_transcript(transcript, system_prompt)
                llm_time = time.time() - llm_start_time
                
                if llm_result.get("success"):
                    llm_response = llm_result["response"]
                    response_file = RESPONSES_DIR / f"response_{client_id}_chunk{chunk_id}_{timestamp}.txt"
                    with open(response_file, "w") as f:
                        f.write(llm_response)
                    logger.info(f"LLM processing time: {llm_time:.2f} seconds")
                    logger.info(f"Saved LLM response to: {response_file}")
                    logger.info(f"LLM Response: {llm_response[:100]}...")
                else:
                    logger.error(f"LLM processing failed: {llm_result.get('error')}")
                    # Save error information for debugging
                    error_file = RESPONSES_DIR / f"error_{client_id}_chunk{chunk_id}_{timestamp}.txt"
                    with open(error_file, "w") as f:
                        f.write(f"Error processing transcript with LLM: {llm_result.get('error')}\n\n")
                        f.write(f"Transcript: {transcript}")
        else:
            logger.warning(f"No transcript generated for {file_name}")
            transcript = ""
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
    client_id = f"{websocket.client.host}_{websocket.client.port}"
    connection_time = datetime.now().strftime('%H:%M:%S')
    logger.info(f"Client connected: {client_id} at {connection_time}")

    active_streams[client_id] = {
        "chunk_id": None,
        "audio_buffer": b"",
        "start_time": time.time(),
        "chunks_processed": 0,
        "total_audio_duration": 0
    }

    try:
        while True:
            data = await websocket.receive_bytes()
            
            # Extract chunk ID from the first 4 bytes (big-endian unsigned int)
            chunk_id = struct.unpack(">I", data[:4])[0]
            audio_data = data[4:]

            client_data = active_streams[client_id]

            # If this is a new chunk ID, process the previous chunk
            if client_data["chunk_id"] != chunk_id:
                if client_data["chunk_id"] is not None and client_data["audio_buffer"]:
                    process_start_time = time.time()
                    logger.info(f"\n--- Processing chunk {client_data['chunk_id']} from {client_id} ---")
                    
                    try:
                        wav_path, transcript, llm_response = await save_to_wav(client_data["audio_buffer"], client_data["chunk_id"], client_id)
                        
                        # Calculate audio duration for this chunk
                        chunk_duration = len(client_data["audio_buffer"]) / (SAMPLE_RATE * NUM_CHANNELS * BITS_PER_SAMPLE / 8)
                        client_data["total_audio_duration"] += chunk_duration
                        client_data["chunks_processed"] += 1
                        
                        process_time = time.time() - process_start_time
                        
                        # Prepare response to client
                        response_data = {
                            "chunk_id": client_data["chunk_id"],
                            "transcript": transcript,
                            "process_time": f"{process_time:.2f}s",
                            "llm_response": llm_response if llm_response else None,
                            "audio_duration": f"{chunk_duration:.2f}s",
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Send response to client with transcript and LLM response
                        await websocket.send_json(response_data)
                        
                        logger.info(f"Total chunks processed: {client_data['chunks_processed']}")
                        logger.info(f"Total audio duration: {client_data['total_audio_duration']:.2f} seconds")
                        logger.info(f"Session duration: {time.time() - client_data['start_time']:.2f} seconds")
                        logger.info(f"--- Finished processing chunk {client_data['chunk_id']} ---\n")
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")
                        logger.error(traceback.format_exc())
                        await websocket.send_json({
                            "error": f"Error processing chunk {client_data['chunk_id']}: {str(e)}",
                            "chunk_id": client_data["chunk_id"],
                            "timestamp": datetime.now().isoformat()
                        })

                # Start collecting a new chunk
                client_data["chunk_id"] = chunk_id
                client_data["audio_buffer"] = audio_data
                logger.info(f"Started new chunk {chunk_id} from {client_id} - {len(audio_data)/1024:.1f} KB")
            else:
                # Continue collecting data for the current chunk
                client_data["audio_buffer"] += audio_data
                # Periodically report buffer size
                if len(client_data["audio_buffer"]) % 10240 < 1024:  # Report roughly every 10KB
                    logger.info(f"Chunk {chunk_id} buffer size: {len(client_data['audio_buffer'])/1024:.1f} KB")
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {client_id}")
        client_data = active_streams.get(client_id)
        if client_data and client_data["chunk_id"] is not None:
            try:
                # Process the final chunk if there's data
                await save_to_wav(client_data["audio_buffer"], client_data["chunk_id"], client_id)
            except Exception as e:
                logger.error(f"Error processing final chunk: {e}")
                logger.error(traceback.format_exc())
            
            # Print session summary
            session_duration = time.time() - client_data["start_time"]
            logger.info(f"\n--- Session Summary for {client_id} ---")
            logger.info(f"Session duration: {session_duration:.2f} seconds")
            logger.info(f"Chunks processed: {client_data['chunks_processed']}")
            logger.info(f"Total audio duration: {client_data['total_audio_duration']:.2f} seconds")
            
            # Calculate average processing factor
            if client_data['total_audio_duration'] > 0:
                avg_factor = session_duration/client_data['total_audio_duration']
                logger.info(f"Average processing factor: {avg_factor:.2f}x real-time")
            else:
                logger.info("Average processing factor: N/A (no audio processed)")
                
            logger.info(f"--- End of Session ---\n")
            
        active_streams.pop(client_id, None)
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        logger.error(traceback.format_exc())
        active_streams.pop(client_id, None)

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
    
    return {
        "status": "healthy",
        "components": {
            "server": "running",
            "whisper": whisper_status,
            "llm": llm_status
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import os
import struct
import time
from datetime import datetime
from pathlib import Path
import asyncio

from app.utils.audio_processor import process_audio_with_whisper

app = FastAPI()

# Audio config
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
BITS_PER_SAMPLE = 16

# Directory setup
RECORDINGS_DIR = Path("recordings")
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR = Path("transcripts")
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# Print startup message
print(f"Server starting...")
print(f"Recordings will be saved to: {RECORDINGS_DIR.absolute()}")
print(f"Transcripts will be saved to: {TRANSCRIPTS_DIR.absolute()}")

active_streams = {}

def create_wav_header(data_length: int) -> bytes:
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
    
    print(f"Saved WAV file: {file_name} ({len(audio_data)/1024:.1f} KB, {audio_duration:.2f} seconds)")
    print(f"WAV save time: {wav_save_time:.3f} seconds")
    
    try:
        # Process audio with Whisper and save transcript
        print(f"Starting transcription at {datetime.now().strftime('%H:%M:%S')}")
        transcription_start_time = time.time()
        transcript = await process_audio_with_whisper(str(file_path))
        transcription_time = time.time() - transcription_start_time
        
        if transcript:
            transcript_file = TRANSCRIPTS_DIR / f"transcript_{client_id}_chunk{chunk_id}_{timestamp}.txt"
            with open(transcript_file, "w") as f:
                f.write(transcript)
            print(f"Saved transcript: {transcript_file}")
            print(f"Total transcription processing time: {transcription_time:.2f} seconds")
            
            # Calculate real-time factor
            rtf = transcription_time / audio_duration if audio_duration > 0 else float('inf')
            print(f"Real-time factor: {rtf:.2f}x (lower is better)")
        else:
            print(f"No transcript generated for {file_name}")
            transcript = ""
    except Exception as e:
        print(f"Error in transcription: {e}")
        transcript = ""
    
    return file_path, transcript

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = f"{websocket.client.host}_{websocket.client.port}"
    connection_time = datetime.now().strftime('%H:%M:%S')
    print(f"Client connected: {client_id} at {connection_time}")

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
            chunk_id = struct.unpack(">I", data[:4])[0]
            audio_data = data[4:]

            client_data = active_streams[client_id]

            if client_data["chunk_id"] != chunk_id:
                if client_data["chunk_id"] is not None and client_data["audio_buffer"]:
                    process_start_time = time.time()
                    print(f"\n--- Processing chunk {client_data['chunk_id']} from {client_id} ---")
                    
                    try:
                        wav_path, transcript = await save_to_wav(client_data["audio_buffer"], client_data["chunk_id"], client_id)
                        
                        # Calculate audio duration for this chunk
                        chunk_duration = len(client_data["audio_buffer"]) / (SAMPLE_RATE * NUM_CHANNELS * BITS_PER_SAMPLE / 8)
                        client_data["total_audio_duration"] += chunk_duration
                        client_data["chunks_processed"] += 1
                        
                        process_time = time.time() - process_start_time
                        
                        # Send response to client with timing information
                        response_message = f"Processed chunk {client_data['chunk_id']} - {len(transcript or '')} chars transcribed in {process_time:.2f}s"
                        await websocket.send_text(response_message)
                        
                        print(f"Total chunks processed: {client_data['chunks_processed']}")
                        print(f"Total audio duration: {client_data['total_audio_duration']:.2f} seconds")
                        print(f"Session duration: {time.time() - client_data['start_time']:.2f} seconds")
                        print(f"--- Finished processing chunk {client_data['chunk_id']} ---\n")
                    except Exception as e:
                        print(f"Error processing chunk: {e}")
                        await websocket.send_text(f"Error processing chunk {client_data['chunk_id']}: {str(e)}")

                client_data["chunk_id"] = chunk_id
                client_data["audio_buffer"] = audio_data
                print(f"Started new chunk {chunk_id} from {client_id} - {len(audio_data)/1024:.1f} KB")
            else:
                client_data["audio_buffer"] += audio_data
                # Periodically report buffer size
                if len(client_data["audio_buffer"]) % 10240 < 1024:  # Report roughly every 10KB
                    print(f"Chunk {chunk_id} buffer size: {len(client_data['audio_buffer'])/1024:.1f} KB")
    except WebSocketDisconnect:
        print(f"Client disconnected: {client_id}")
        client_data = active_streams.get(client_id)
        if client_data and client_data["chunk_id"] is not None:
            try:
                await save_to_wav(client_data["audio_buffer"], client_data["chunk_id"], client_id)
            except Exception as e:
                print(f"Error processing final chunk: {e}")
            
            # Print session summary
            session_duration = time.time() - client_data["start_time"]
            print(f"\n--- Session Summary for {client_id} ---")
            print(f"Session duration: {session_duration:.2f} seconds")
            print(f"Chunks processed: {client_data['chunks_processed']}")
            print(f"Total audio duration: {client_data['total_audio_duration']:.2f} seconds")
            print(f"Average processing factor: {session_duration/client_data['total_audio_duration']:.2f}x real-time")
            print(f"--- End of Session ---\n")
            
        active_streams.pop(client_id, None)
    except Exception as e:
        print(f"Error: {e}")
        active_streams.pop(client_id, None)

@app.get("/")
async def get_root():
    return {"message": "Audio Streaming Server is running. Connect to /ws/audio for WebSocket connection."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True) 
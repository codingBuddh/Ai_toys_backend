#!/usr/bin/env python3
"""
Test WebSocket client to simulate an ESP32 sending audio data.
This script sends a test WAV file in chunks over WebSocket.
"""

import asyncio
import websockets
import struct
import time
import argparse
import os
from pathlib import Path

async def send_audio_file(websocket_uri, audio_file, chunk_size=1024):
    """
    Send an audio file in chunks over WebSocket.
    
    Args:
        websocket_uri: WebSocket server URI
        audio_file: Path to the audio file (WAV format)
        chunk_size: Size of each chunk in bytes
    """
    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} not found")
        return
        
    # Read the WAV file, skip the header (44 bytes for standard WAV)
    with open(audio_file, 'rb') as f:
        # Skip WAV header
        f.seek(44)  
        audio_data = f.read()
    
    print(f"Connecting to {websocket_uri}...")
    
    async with websockets.connect(websocket_uri) as websocket:
        print(f"Connected to {websocket_uri}")
        
        # Simulate sending audio in chunks
        chunk_id = 1
        total_sent = 0
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            if not chunk:
                break
                
            # Prepend 4-byte chunk ID
            data_with_header = struct.pack(">I", chunk_id) + chunk
            
            # Send the chunk
            await websocket.send(data_with_header)
            total_sent += len(chunk)
            
            print(f"Sent chunk {chunk_id}: {len(chunk)} bytes")
            
            # Wait a bit to simulate real-time streaming
            await asyncio.sleep(0.1)
            
            # Every 10 chunks, start a new chunk_id to simulate a new recording
            if i > 0 and i % (chunk_size * 10) == 0:
                response = await websocket.recv()
                print(f"Server response: {response}")
                chunk_id += 1
                print(f"Starting new chunk with ID: {chunk_id}")
        
        print(f"Finished sending {total_sent} bytes in {chunk_id} chunks")
        
        # Wait for the final response
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"Final server response: {response}")
        except asyncio.TimeoutError:
            print("No final response received from server")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket audio streaming test client")
    parser.add_argument("--server", default="ws://localhost:8000/ws/audio", 
                        help="WebSocket server URI")
    parser.add_argument("--file", default=None, 
                        help="Path to WAV audio file to send")
    parser.add_argument("--chunk-size", type=int, default=1024, 
                        help="Size of each audio chunk in bytes")
    
    args = parser.parse_args()
    
    if args.file is None:
        print("Error: Please specify an audio file with --file")
        exit(1)
    
    asyncio.run(send_audio_file(args.server, args.file, args.chunk_size)) 
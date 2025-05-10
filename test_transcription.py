#!/usr/bin/env python3
"""
Test script to verify the Whisper speech-to-text functionality.
This script takes an audio file and transcribes it.
"""

import asyncio
import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Import the Whisper transcription function from our app
from app.utils.audio_processor import process_audio_with_whisper

async def test_transcription(audio_file_path):
    """Test transcription on a given audio file."""
    print(f"\n{'='*50}")
    print(f"WHISPER TRANSCRIPTION TEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    print(f"Testing transcription on file: {audio_file_path}")
    
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return
    
    overall_start_time = time.time()
    print(f"Starting transcription process at {datetime.now().strftime('%H:%M:%S')}")
    
    transcript = await process_audio_with_whisper(audio_file_path)
    
    overall_time = time.time() - overall_start_time
    
    if transcript:
        print(f"\n{'-'*50}")
        print("TRANSCRIPTION RESULT:")
        print(f"{'-'*50}")
        print(transcript)
        print(f"{'-'*50}")
        
        # Save the transcript to a file
        output_file = Path("transcripts") / f"test_transcript_{Path(audio_file_path).stem}.txt"
        with open(output_file, "w") as f:
            f.write(transcript)
        print(f"Transcript saved to: {output_file}")
    else:
        print("No transcript was generated. There might be an issue with the audio file or the Whisper model.")
    
    print(f"\n{'-'*50}")
    print(f"PERFORMANCE SUMMARY:")
    print(f"{'-'*50}")
    print(f"Total processing time: {overall_time:.2f} seconds")
    print(f"End time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # Use the provided audio file or default to the temp.wav in recordings directory
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "recordings/temp.wav"
    
    asyncio.run(test_transcription(audio_path)) 
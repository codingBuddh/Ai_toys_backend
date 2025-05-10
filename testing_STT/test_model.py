#!/usr/bin/env python3
"""
Test script to quickly check a single Whisper model with timing information.
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path to access the main application modules if needed
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))

import torch
import whisper

# Directory configurations
SCRIPT_DIR = Path(__file__).parent.absolute()
TRANSCRIPTS_DIR = SCRIPT_DIR / "transcripts"
RECORDINGS_DIR = parent_dir / "recordings"  # Using the main recordings directory

async def get_audio_duration(audio_file_path):
    """Get the duration of an audio file in seconds."""
    try:
        import wave
        with wave.open(audio_file_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Could not determine audio duration: {e}")
        return None

async def test_model(model_name, audio_file_path, verbose=False):
    """Test a single Whisper model on an audio file with timing information."""
    print(f"\n{'='*80}")
    print(f"WHISPER MODEL TEST: {model_name}")
    print(f"{'='*80}")
    print(f"Audio file: {audio_file_path}")
    print(f"Device: {'CUDA/GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Ensure file exists
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return
    
    # Ensure transcripts directory exists
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get audio file size
    file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
    print(f"Audio file size: {file_size_mb:.2f} MB")
    
    # Get audio duration
    audio_duration = await get_audio_duration(audio_file_path)
    if audio_duration:
        print(f"Audio duration: {audio_duration:.2f} seconds")
    
    # Test the model
    try:
        # 1. Load the model
        print(f"Loading model '{model_name}'...")
        load_start = time.time()
        model = whisper.load_model(model_name)
        load_time = time.time() - load_start
        print(f"✓ Model loaded in {load_time:.2f} seconds")
        
        # Get model size
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        print(f"Model size: {model_size_mb:.2f} MB")
        
        # 2. Transcribe
        print(f"Transcribing audio...")
        trans_start = time.time()
        
        # Set transcript options based on verbosity
        if verbose:
            transcription = model.transcribe(audio_file_path, verbose=True)
        else:
            transcription = model.transcribe(audio_file_path)
            
        trans_time = time.time() - trans_start
        print(f"✓ Audio transcribed in {trans_time:.2f} seconds")
        
        # Calculate real-time factor
        if audio_duration:
            rtf = trans_time / audio_duration
            print(f"Real-time factor: {rtf:.2f}x (lower is better)")
        
        # 3. Save transcript
        audio_file_name = Path(audio_file_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_file = TRANSCRIPTS_DIR / f"test_{model_name}_{audio_file_name}_{timestamp}.txt"
        
        with open(transcript_file, "w") as f:
            f.write(transcription["text"])
        print(f"Transcript saved to: {transcript_file}")
        
        # 4. Print transcript
        print(f"\n{'-'*80}")
        print("TRANSCRIPT:")
        print(f"{'-'*80}")
        print(transcription["text"])
        print(f"{'-'*80}")
        
        # 5. Print summary
        print(f"\n{'-'*80}")
        print("PERFORMANCE SUMMARY:")
        print(f"{'-'*80}")
        print(f"Model: {model_name}")
        print(f"Model size: {model_size_mb:.2f} MB")
        print(f"Load time: {load_time:.2f} seconds")
        print(f"Transcription time: {trans_time:.2f} seconds")
        if audio_duration:
            print(f"Audio duration: {audio_duration:.2f} seconds")
            print(f"Real-time factor: {rtf:.2f}x")
        print(f"{'-'*80}")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a Whisper STT model on an audio file")
    parser.add_argument("--model", "-m", default="small", 
                        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", 
                                "medium", "medium.en", "large"],
                        help="Whisper model to use (default: small)")
    parser.add_argument("--file", "-f", default=None,
                        help="Audio file to transcribe (default: recordings/temp.wav)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed transcription progress")
    args = parser.parse_args()
    
    # Get audio file
    if args.file:
        audio_path = args.file
    else:
        audio_path = str(RECORDINGS_DIR / "temp.wav")
    
    asyncio.run(test_model(args.model, audio_path, args.verbose)) 
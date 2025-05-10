#!/usr/bin/env python3
"""
Setup script for ESP32 WebSocket Audio Streaming & Transcription Server.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_prerequisites():
    """Check if the required prerequisites are installed."""
    prerequisites_ok = True
    
    # Check for FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE, 
                       check=True)
        print("✅ FFmpeg is installed")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ FFmpeg is not installed or not in PATH")
        print("   Please install FFmpeg (required for Whisper)")
        print("   - macOS: brew install ffmpeg")
        print("   - Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg")
        print("   - Windows: Download from ffmpeg.org or use choco install ffmpeg")
        prerequisites_ok = False
    
    # Check for Rust (needed for tiktoken)
    try:
        subprocess.run(["rustc", "--version"], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE, 
                       check=True)
        print("✅ Rust is installed")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("⚠️  Rust is not installed or not in PATH")
        print("   This may be required for tiktoken (a dependency of Whisper)")
        print("   Visit https://rustup.rs/ to install Rust if needed")
    
    return prerequisites_ok

def setup():
    """Create necessary directories."""
    dirs = [
        "recordings",
        "transcripts"
    ]
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
        else:
            print(f"✅ Directory already exists: {dir_path}")

if __name__ == "__main__":
    print("Setting up ESP32 WebSocket Audio Streaming & Transcription Server...")
    
    # Check prerequisites
    prereq_ok = check_prerequisites()
    
    # Create directories
    setup()
    
    print("\nSetup complete!")
    
    if not prereq_ok:
        print("\n⚠️  Some prerequisites are missing. Please install them before running the server.")
    
    print("\nTo run the server, execute: python -m app.main") 
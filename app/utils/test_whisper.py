"""
Utility to test Whisper transcription performance from the main application.
This provides a simple function to test the current Whisper model configuration.
"""

import time
import os
from pathlib import Path
import sys

# Add the testing_STT directory to the path
ROOT_DIR = Path(__file__).parent.parent.parent
TESTING_DIR = ROOT_DIR / "testing_STT"
sys.path.append(str(TESTING_DIR))

from app.utils.audio_processor import process_audio_with_whisper

async def test_whisper_performance(audio_file_path=None):
    """
    Test the performance of the current Whisper model configuration.
    
    Args:
        audio_file_path: Path to the audio file to use for testing. If None,
                         will use the default temp.wav in the recordings directory.
                         
    Returns:
        dict: Performance metrics
    """
    # Use default audio if none provided
    if not audio_file_path:
        audio_file_path = str(ROOT_DIR / "recordings" / "temp.wav")
    
    if not os.path.exists(audio_file_path):
        return {
            "success": False,
            "error": f"Audio file not found: {audio_file_path}"
        }
    
    # Get audio file size
    file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
    
    try:
        # Get audio duration if possible
        try:
            import wave
            with wave.open(audio_file_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
        except Exception as e:
            duration = None
            
        # Process with Whisper and measure time
        start_time = time.time()
        transcript = await process_audio_with_whisper(audio_file_path)
        process_time = time.time() - start_time
        
        # Calculate performance metrics
        metrics = {
            "success": True,
            "audio_file": audio_file_path,
            "file_size_mb": file_size_mb,
            "audio_duration": duration,
            "process_time": process_time,
            "transcript_length": len(transcript) if transcript else 0,
            "has_transcript": bool(transcript),
            "device": "cpu"  # We're only using CPU now
        }
        
        # Add real-time factor if duration is available
        if duration:
            metrics["real_time_factor"] = process_time / duration
            
        return metrics
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "audio_file": audio_file_path,
            "file_size_mb": file_size_mb
        }


if __name__ == "__main__":
    """Example usage when run directly."""
    import asyncio
    
    async def main():
        # Use from command line if an argument is provided
        if len(sys.argv) > 1:
            audio_path = sys.argv[1]
        else:
            audio_path = None
            
        print(f"Testing Whisper performance with audio: {audio_path or 'default audio'}")
        
        metrics = await test_whisper_performance(audio_path)
        
        if metrics["success"]:
            print("\n--- Whisper Performance Test Results ---")
            for key, value in metrics.items():
                if key != "success":
                    if isinstance(value, float):
                        print(f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
            
            if "real_time_factor" in metrics:
                rtf = metrics["real_time_factor"]
                print(f"\nProcessing speed: {rtf:.2f}x real-time")
                
                if rtf < 1.0:
                    print("⚠️  Warning: Processing is slower than real-time")
                elif rtf < 2.0:
                    print("✓ Processing is faster than real-time but may struggle with continuous streaming")
                else:
                    print("✓ Processing is significantly faster than real-time and suitable for streaming")
        else:
            print(f"Error: {metrics.get('error', 'Unknown error')}")
    
    asyncio.run(main()) 
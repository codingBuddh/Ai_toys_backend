import os
import asyncio
import time
from datetime import datetime
from pathlib import Path
import torch

# Initialize the USE_TRANSFORMERS flag with a default value
USE_TRANSFORMERS = False

# We'll attempt to use the official OpenAI Whisper package
# If it fails, we'll fall back to the transformers implementation
try:
    import whisper
    print("Using OpenAI's Whisper package for speech recognition")
except ImportError:
    # Fallback to the transformers library for Whisper model
    from transformers import pipeline
    USE_TRANSFORMERS = True
    print("Using transformers pipeline for speech recognition")

model = None

async def load_model():
    global model, USE_TRANSFORMERS
    if model is None:
        print("Loading Whisper model...")
        start_time = time.time()
        # Load the model in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        
        if not USE_TRANSFORMERS:
            # Using the official OpenAI Whisper package
            try:
                model = await loop.run_in_executor(None, lambda: whisper.load_model("small"))
                load_time = time.time() - start_time
                print(f"OpenAI Whisper model loaded successfully! (Took {load_time:.2f} seconds)")
            except Exception as e:
                print(f"Error loading OpenAI Whisper model: {e}")
                print("Falling back to transformers implementation...")
                from transformers import pipeline
                USE_TRANSFORMERS = True
                fallback_start_time = time.time()
                model = await loop.run_in_executor(None, lambda: pipeline(
                    "automatic-speech-recognition", 
                    model="openai/whisper-small",
                    chunk_length_s=30,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                ))
                load_time = time.time() - fallback_start_time
                print(f"Transformers Whisper model loaded successfully! (Took {load_time:.2f} seconds)")
        else:
            # Using the transformers library
            model = await loop.run_in_executor(None, lambda: pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-small",
                chunk_length_s=30,
                device="cuda" if torch.cuda.is_available() else "cpu"
            ))
            load_time = time.time() - start_time
            print(f"Transformers Whisper model loaded successfully! (Took {load_time:.2f} seconds)")

async def process_audio_with_whisper(audio_file_path: str) -> str:
    """
    Process audio file with OpenAI Whisper and return the transcript.
    
    Args:
        audio_file_path: Path to the WAV audio file
        
    Returns:
        Transcript text
    """
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file {audio_file_path} does not exist")
        return ""
    
    file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
    print(f"Audio file size: {file_size_mb:.2f} MB")
    
    # Get audio duration if possible
    try:
        import wave
        with wave.open(audio_file_path, 'rb') as wf:
            # Calculate duration in seconds
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            print(f"Audio duration: {duration:.2f} seconds")
    except Exception as e:
        print(f"Could not determine audio duration: {e}")
    
    # Make sure the model is loaded
    model_start_time = time.time()
    await load_model()
    model_load_time = time.time() - model_start_time
    print(f"Model preparation took {model_load_time:.2f} seconds")
    
    try:
        # Transcribe the audio file
        print(f"Starting transcription at {datetime.now().strftime('%H:%M:%S')}")
        transcribe_start_time = time.time()
        loop = asyncio.get_event_loop()
        
        if not USE_TRANSFORMERS:
            # Using the official OpenAI Whisper package
            result = await loop.run_in_executor(None, lambda: model.transcribe(audio_file_path))
            transcript = result["text"].strip()
        else:
            # Using the transformers library
            result = await loop.run_in_executor(None, lambda: model(audio_file_path))
            transcript = result["text"].strip()
        
        transcribe_time = time.time() - transcribe_start_time
        
        if not transcript:
            print(f"Warning: No transcript generated for {audio_file_path}")
            return ""
            
        print(f"Generated transcript: {transcript[:50]}...")
        print(f"Transcription completed in {transcribe_time:.2f} seconds")
        
        # Calculate processing speed ratio (audio duration / processing time)
        try:
            speed_ratio = duration / transcribe_time
            print(f"Processing speed: {speed_ratio:.2f}x real-time")
        except:
            pass
            
        return transcript
        
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return "" 
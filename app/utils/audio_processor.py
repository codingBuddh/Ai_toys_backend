"""
Audio Processing Utility for Whisper Speech-to-Text

This module provides functionality for processing audio files with OpenAI's Whisper
speech-to-text model. It handles model loading, audio transcription, and performance tracking.

Debug Info:
- Model loading is done asynchronously to avoid blocking
- All operations are timed and logged
- Audio file information (size, duration) is logged
- Processing speed is calculated and reported
"""

import os
import asyncio
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("whisper_processor.log")
    ]
)
logger = logging.getLogger("whisper_processor")

# Single implementation - only using OpenAI Whisper
import whisper
logger.info("Using OpenAI's Whisper package for speech recognition")

# Global model instance
model = None
model_name = os.environ.get("WHISPER_MODEL", "tiny.en")  # Default to tiny.en for speed

async def load_model(model_size: str = None):
    """
    Load the Whisper model asynchronously.
    
    Args:
        model_size: Size of the model to load (tiny, base, small, medium, large)
                   If None, uses the WHISPER_MODEL env var or defaults to "tiny.en"
    
    Raises:
        Exception: If the model fails to load
        
    Debug Info:
        - Model loading time is measured and logged
        - Model is loaded in a separate thread to avoid blocking
        - Model size is configurable via environment variable
    """
    global model, model_name
    
    # Use provided model size or default
    if model_size:
        model_name = model_size
    
    # Force tiny.en model if env var is not respected
    if model_name not in ["tiny.en", "tiny", "base.en", "base"]:
        logger.warning(f"Model '{model_name}' may be too large for this application. Falling back to tiny.en")
        model_name = "tiny.en"
    
    if model is None:
        logger.info(f"Loading Whisper model '{model_name}'...")
        start_time = time.time()
        
        # Load the model in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        
        try:
            model = await loop.run_in_executor(None, lambda: whisper.load_model(model_name))
            load_time = time.time() - start_time
            logger.info(f"Whisper model '{model_name}' loaded successfully! (Took {load_time:.2f} seconds)")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            logger.error(traceback.format_exc())
            raise  # Re-raise the exception since we no longer have a fallback

async def process_audio_with_whisper(audio_file_path: str) -> str:
    """
    Process audio file with OpenAI Whisper and return the transcript.
    
    Args:
        audio_file_path: Path to the WAV audio file
        
    Returns:
        str: Transcript text or empty string if processing failed
        
    Debug Info:
        - Audio file existence is verified
        - Audio file size and duration are logged
        - Model loading time is measured and logged
        - Transcription time is measured and logged
        - Processing speed ratio is calculated (audio duration / processing time)
        - Errors are caught and logged
    """
    if not os.path.exists(audio_file_path):
        logger.error(f"Error: Audio file {audio_file_path} does not exist")
        return ""
    
    file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
    logger.info(f"Audio file size: {file_size_mb:.2f} MB")
    
    # Get audio duration if possible
    duration = None
    try:
        import wave
        with wave.open(audio_file_path, 'rb') as wf:
            # Calculate duration in seconds
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            # Skip very short audio files to avoid processing errors
            if duration < 0.5:
                logger.warning(f"Audio file too short ({duration:.2f}s), skipping transcription")
                return ""
    except Exception as e:
        logger.warning(f"Could not determine audio duration: {e}")
    
    # Make sure the model is loaded
    model_start_time = time.time()
    await load_model()
    model_load_time = time.time() - model_start_time
    logger.info(f"Model preparation took {model_load_time:.2f} seconds")
    
    try:
        # Transcribe the audio file
        logger.info(f"Starting transcription at {datetime.now().strftime('%H:%M:%S')}")
        transcribe_start_time = time.time()
        loop = asyncio.get_event_loop()
        
        # Using the OpenAI Whisper package with optimized parameters
        transcription_options = {
            'language': 'en',             # Specify English for faster processing
            'fp16': False,                # Avoid FP16 warning on CPU
            'beam_size': 3,               # Reduced beam size for faster processing
            'best_of': 1,                 # Only return the best result
            'temperature': (0.0, 0.2, 0.4), # Use lower temperature for simpler transcription
            'compression_ratio_threshold': 2.4, # Adjusted threshold
            'condition_on_previous_text': True,
            'initial_prompt': "This is audio from an ESP32 microphone."
        }
        
        result = await loop.run_in_executor(
            None, 
            lambda: model.transcribe(audio_file_path, **transcription_options)
        )
        transcript = result["text"].strip()
        
        transcribe_time = time.time() - transcribe_start_time
        
        if not transcript:
            logger.warning(f"Warning: No transcript generated for {audio_file_path}")
            return ""
            
        logger.info(f"Generated transcript: {transcript[:50]}...")
        logger.info(f"Transcription completed in {transcribe_time:.2f} seconds")
        
        # Calculate processing speed ratio (audio duration / processing time)
        if duration:
            speed_ratio = duration / transcribe_time
            logger.info(f"Processing speed: {speed_ratio:.2f}x real-time")
            
        # Log full transcript at debug level
        logger.debug(f"Full transcript: {transcript}")
            
        return transcript
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        logger.error(traceback.format_exc())
        return ""

async def get_available_models():
    """
    Get information about available Whisper models.
    
    Returns:
        dict: Dictionary with model information
        
    Debug Info:
        - Lists all available models with their properties
        - Indicates which model is currently loaded
    """
    models_info = {
        "tiny": {"parameters": "39M", "english_only": False, "multilingual": True},
        "base": {"parameters": "74M", "english_only": False, "multilingual": True},
        "small": {"parameters": "244M", "english_only": False, "multilingual": True},
        "medium": {"parameters": "769M", "english_only": False, "multilingual": True},
        "large": {"parameters": "1550M", "english_only": False, "multilingual": True},
        "tiny.en": {"parameters": "39M", "english_only": True, "multilingual": False},
        "base.en": {"parameters": "74M", "english_only": True, "multilingual": False},
        "small.en": {"parameters": "244M", "english_only": True, "multilingual": False},
        "medium.en": {"parameters": "769M", "english_only": True, "multilingual": False}
    }
    
    # Mark currently loaded model
    global model_name
    for name in models_info:
        models_info[name]["loaded"] = (name == model_name)
    
    return models_info 
"""
Google Cloud Text-to-Speech processor for converting LLM responses to audio

This module provides a class for converting text to speech using Google Cloud TTS.
The audio is returned as MP3 bytes that can be streamed to the client.
"""

import os
import json
import logging
from pathlib import Path
from google.cloud import texttospeech
from google.oauth2 import service_account
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tts_processor.log")
    ]
)
logger = logging.getLogger("tts_processor")

class TTSProcessor:
    """
    Text-to-Speech processor using Google Cloud TTS.
    
    This class handles:
    1. Converting text to speech using Google Cloud TTS
    2. Configuring voice parameters
    3. Returning audio as MP3 bytes
    4. Saving audio files for debugging
    """
    
    def __init__(self, credentials_path=None, credentials_json=None):
        """
        Initialize the TTS processor.
        
        Args:
            credentials_path: Path to Google Cloud credentials JSON file
            credentials_json: JSON string containing Google Cloud credentials
        """
        self.client = None
        self.credentials = None
        
        # Try loading credentials in this order:
        # 1. From provided JSON string
        # 2. From provided file path
        # 3. From environment variable
        # 4. From default location
        
        if credentials_json:
            try:
                # Parse JSON string and create credentials
                cred_info = json.loads(credentials_json)
                self.credentials = service_account.Credentials.from_service_account_info(cred_info)
                logger.info(f"Using Google Cloud credentials from provided JSON string")
            except Exception as e:
                logger.error(f"Failed to parse credentials JSON: {e}")
        
        elif credentials_path and os.path.exists(credentials_path):
            try:
                # Load credentials from specified file
                self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
                logger.info(f"Using Google Cloud credentials from: {credentials_path}")
            except Exception as e:
                logger.error(f"Failed to load credentials from {credentials_path}: {e}")
        
        elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            # Use environment variable
            cred_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
            if os.path.exists(cred_path):
                try:
                    self.credentials = service_account.Credentials.from_service_account_file(cred_path)
                    logger.info(f"Using Google Cloud credentials from environment variable: {cred_path}")
                except Exception as e:
                    logger.error(f"Failed to load credentials from {cred_path}: {e}")
            else:
                logger.warning(f"Credentials file from environment variable does not exist: {cred_path}")
        
        # Try to create the client
        try:
            if self.credentials:
                # Create client with explicit credentials
                self.client = texttospeech.TextToSpeechClient(credentials=self.credentials)
            else:
                # Try default authentication
                self.client = texttospeech.TextToSpeechClient()
            
            # Test the client with a simple request to verify authentication
            voices = self.client.list_voices()
            logger.info(f"TTS Processor initialized successfully. Found {len(voices.voices)} voices.")
        except Exception as e:
            logger.error(f"Failed to initialize TTS Processor: {e}")
            raise
        
        # Create directory for saving audio files
        self.audio_dir = Path("tts_audio")
        self.audio_dir.mkdir(parents=True, exist_ok=True)
    
    async def text_to_speech(self, text, voice_name="en-US-Wavenet-D", language_code="en-US", 
                      save_file=False, client_id=None):
        """
        Convert text to speech using Google Cloud TTS.
        
        Args:
            text: Text to convert to speech
            voice_name: Name of the voice to use
            language_code: Language code for the voice
            save_file: Whether to save the audio file for debugging
            client_id: Client ID for file naming
            
        Returns:
            bytes: Audio content as MP3 bytes
        """
        try:
            # Build the synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Build the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )
            
            # Select audio format (MP3 for better compatibility with ESP32)
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,  # 1.0 is normal speed
                pitch=0.0,  # 0.0 is normal pitch
                volume_gain_db=0.0  # 0.0 is normal volume
            )
            
            # Perform the request
            logger.info(f"Generating TTS for: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Save the audio file if requested
            if save_file:
                timestamp = datetime.utcnow().isoformat().replace(":", "-").replace(".", "-")
                file_name = f"tts_{client_id or 'unknown'}_{timestamp}.mp3"
                file_path = self.audio_dir / file_name
                
                with open(file_path, "wb") as out:
                    out.write(response.audio_content)
                logger.info(f"Audio saved to: {file_path}")
            
            return response.audio_content
        
        except Exception as e:
            logger.error(f"Error in text_to_speech: {e}")
            return None
            
    @staticmethod
    def get_available_voices(client=None):
        """
        Get a list of available voices from Google Cloud TTS.
        
        Args:
            client: Optional TextToSpeechClient instance
            
        Returns:
            list: List of voice names
        """
        try:
            if client is None:
                client = texttospeech.TextToSpeechClient()
            
            voices = client.list_voices()
            return [voice.name for voice in voices.voices]
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
            return [] 
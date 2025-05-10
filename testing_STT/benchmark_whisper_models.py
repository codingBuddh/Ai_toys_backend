#!/usr/bin/env python3
"""
Benchmark script to compare the performance of different Whisper models.
This script loads each model and measures the transcription time on the same audio file.
"""

import asyncio
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path to access the main application modules if needed
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))

import torch
import whisper

# Configuration
MODELS = [
    {"name": "tiny", "english_only": False, "description": "Tiny: Fastest, least accurate"},
    {"name": "tiny.en", "english_only": True, "description": "Tiny (English): Faster for English content"},
    {"name": "base", "english_only": False, "description": "Base: Fast, decent accuracy"},
    {"name": "base.en", "english_only": True, "description": "Base (English): Faster for English content"},
    {"name": "small", "english_only": False, "description": "Small: Good balance of speed and accuracy"},
    {"name": "small.en", "english_only": True, "description": "Small (English): Better for English content"},
    {"name": "medium", "english_only": False, "description": "Medium: More accurate but slower"},
    {"name": "medium.en", "english_only": True, "description": "Medium (English): More accurate for English"},
    {"name": "large", "english_only": False, "description": "Large: Most accurate but slowest"},
    # Uncomment if you have large-v2, large-v3 or distil-large-v2 models
    # {"name": "large-v2", "english_only": False, "description": "Large V2: Improved large model"},
    # {"name": "large-v3", "english_only": False, "description": "Large V3: Latest large model"},
    # {"name": "distil-large-v2", "english_only": False, "description": "Distil Large V2: Faster large model"}
]

# Directory configurations
SCRIPT_DIR = Path(__file__).parent.absolute()
TRANSCRIPTS_DIR = SCRIPT_DIR / "transcripts"
RESULTS_DIR = SCRIPT_DIR
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

async def benchmark_model(model_info, audio_file_path):
    """Benchmark a single Whisper model."""
    model_name = model_info["name"]
    is_english = model_info["english_only"]
    description = model_info["description"]
    
    print(f"\n{'-'*80}")
    print(f"Testing model: {model_name} - {description}")
    print(f"{'-'*80}")
    
    result = {
        "model_name": model_name,
        "english_only": is_english,
        "description": description,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    
    # 1. Measure model loading time
    try:
        print(f"Loading model {model_name}...")
        load_start = time.time()
        model = whisper.load_model(model_name)
        load_time = time.time() - load_start
        result["load_time"] = load_time
        result["load_success"] = True
        print(f"✓ Model loaded in {load_time:.2f} seconds")
        
        # Get model size
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        result["model_size_mb"] = model_size_mb
        print(f"Model size: {model_size_mb:.2f} MB")
        
        # 2. Measure transcription time
        try:
            print(f"Transcribing audio...")
            transcribe_start = time.time()
            transcription = model.transcribe(audio_file_path)
            transcribe_time = time.time() - transcribe_start
            result["transcribe_time"] = transcribe_time
            result["transcribe_success"] = True
            
            # Audio duration for computing real-time factor
            audio_duration = await get_audio_duration(audio_file_path)
            result["audio_duration"] = audio_duration
            
            if audio_duration:
                rtf = transcribe_time / audio_duration
                result["real_time_factor"] = rtf
                print(f"✓ Audio transcribed in {transcribe_time:.2f} seconds")
                print(f"Real-time factor: {rtf:.2f}x (lower is better)")
            else:
                print(f"✓ Audio transcribed in {transcribe_time:.2f} seconds")
                
            # Save transcription
            result["transcript"] = transcription["text"]
            audio_file_name = Path(audio_file_path).stem
            transcript_path = TRANSCRIPTS_DIR / f"benchmark_{model_name}_{audio_file_name}.txt"
            with open(transcript_path, "w") as f:
                f.write(transcription["text"])
            print(f"Transcript saved to: {transcript_path}")
            
        except Exception as e:
            print(f"✗ Transcription failed: {e}")
            result["transcribe_success"] = False
            result["transcribe_error"] = str(e)
    
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        result["load_success"] = False
        result["load_error"] = str(e)
    
    return result

async def run_benchmark(audio_file_path):
    """Run benchmarks for all models on the specified audio file."""
    print(f"\n{'='*80}")
    print(f"WHISPER MODEL BENCHMARK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"Audio file: {audio_file_path}")
    print(f"Device: {'CUDA/GPU' if torch.cuda.is_available() else 'CPU'}")
    
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
    
    # Run benchmarks for each model
    results = []
    for model_info in MODELS:
        result = await benchmark_model(model_info, audio_file_path)
        results.append(result)
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"whisper_benchmark_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Size (MB)':<12} {'Load Time (s)':<15} {'Transcribe (s)':<15} {'RTF':<8}")
    print(f"{'-'*80}")
    
    for result in results:
        if result.get("load_success", False) and result.get("transcribe_success", False):
            model_name = result["model_name"]
            size = f"{result.get('model_size_mb', 0):.1f}"
            load = f"{result.get('load_time', 0):.2f}"
            trans = f"{result.get('transcribe_time', 0):.2f}"
            rtf = f"{result.get('real_time_factor', 0):.2f}x"
            print(f"{model_name:<15} {size:<12} {load:<15} {trans:<15} {rtf:<8}")
        else:
            model_name = result["model_name"]
            print(f"{model_name:<15} {'Failed':<12} {'Failed':<15} {'Failed':<15} {'N/A':<8}")
    
    print(f"{'='*80}")
    print(f"Benchmark completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

if __name__ == "__main__":
    # Use the provided audio file or default to the temp.wav in recordings directory
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        default_audio = RECORDINGS_DIR / "temp.wav"
        audio_path = str(default_audio)
    
    # Run the benchmark
    asyncio.run(run_benchmark(audio_path)) 
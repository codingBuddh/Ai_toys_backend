#!/bin/bash
# Run the Whisper model benchmark and generate visualizations

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it first."
    echo "macOS: brew install ffmpeg"
    echo "Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg"
    exit 1
fi

# Check if audio file is provided
if [ $# -eq 0 ]; then
    # No audio file provided, use default
    AUDIO_FILE="$PARENT_DIR/recordings/temp.wav"
    echo "Using default audio file: $AUDIO_FILE"
else
    # Use provided audio file
    AUDIO_FILE="$1"
    echo "Using audio file: $AUDIO_FILE"
fi

# Check if audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file $AUDIO_FILE not found."
    exit 1
fi

# Create directories if they don't exist
mkdir -p "$SCRIPT_DIR/transcripts" "$SCRIPT_DIR/benchmark_plots"

echo "==================================================================="
echo "WHISPER MODEL BENCHMARK"
echo "==================================================================="
echo ""
echo "Step 1: Running benchmark on all Whisper models..."
cd "$SCRIPT_DIR" && python3 benchmark_whisper_models.py "$AUDIO_FILE"

echo ""
echo "Step 2: Generating visualizations..."
cd "$SCRIPT_DIR" && python3 visualize_benchmark.py

echo ""
echo "==================================================================="
echo "Benchmark and visualization complete!"
echo "- Benchmark results are saved as JSON files in: $SCRIPT_DIR"
echo "- Visualizations are saved in: $SCRIPT_DIR/benchmark_plots"
echo "- Individual transcripts are saved in: $SCRIPT_DIR/transcripts"
echo "===================================================================" 
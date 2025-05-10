#!/bin/bash
# Test a single Whisper model

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Default model to test
MODEL="small"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      MODEL="$2"
      shift 2
      ;;
    -f|--file)
      AUDIO_FILE="$2"
      shift 2
      ;;
    -v|--verbose)
      VERBOSE="-v"
      shift
      ;;
    *)
      # Default to treating unknown args as the audio file
      AUDIO_FILE="$1"
      shift
      ;;
  esac
done

# Check if audio file is provided
if [ -z "$AUDIO_FILE" ]; then
    # No audio file provided, use default
    AUDIO_FILE="$PARENT_DIR/recordings/temp.wav"
    echo "Using default audio file: $AUDIO_FILE"
fi

# Check if audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file $AUDIO_FILE not found."
    exit 1
fi

# Create transcripts directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/transcripts"

echo "==================================================================="
echo "TESTING WHISPER MODEL: $MODEL"
echo "==================================================================="
echo "Audio file: $AUDIO_FILE"
echo "Starting test at $(date '+%H:%M:%S')"

# Run the test
cd "$SCRIPT_DIR" && python3 test_model.py --model "$MODEL" --file "$AUDIO_FILE" $VERBOSE

echo "Test completed at $(date '+%H:%M:%S')"
echo "===================================================================" 
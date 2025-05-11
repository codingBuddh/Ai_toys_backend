# Whisper STT Model Testing Tools

This directory contains scripts for testing and benchmarking Whisper speech-to-text models.

## Available Tools

1. **Single Model Test** - Test a specific Whisper model with detailed timing information
2. **Full Benchmark** - Run benchmarks on all Whisper models and generate comparison visualizations

## Single Model Test

To test a single Whisper model on an audio file:

```bash
./test_model.sh --model small path/to/audio.wav
```

Options:
- `-m, --model`: Model to test (tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large)
- `-f, --file`: Audio file to transcribe
- `-v, --verbose`: Show detailed progress during transcription

Examples:
```bash
# Test the small model with default audio
./test_model.sh -m small

# Test the tiny.en model with verbose output
./test_model.sh -m tiny.en -v

# Test the medium model with a specific audio file
./test_model.sh -m medium -f /path/to/audio.wav
```

## Full Benchmark

To benchmark all Whisper models on an audio file:

```bash
./run_benchmark.sh path/to/audio.wav
```

This will:
1. Test each model one by one
2. Generate comparative visualizations
3. Save all results and transcripts

## Results

- Transcripts are saved in the `transcripts/` directory
- Benchmark results are saved as JSON files
- Visualizations are saved in the `benchmark_plots/` directory

## Technical Notes

- All tests now run on CPU only for better deployment compatibility
- The transformers dependency has been removed to reduce memory requirements
- The benchmarks will measure actual model size, load time, and transcription speed

## Model Information

| Model      | Size (MB) | English-only | Speed (RTF) | Accuracy |
|------------|-----------|--------------|-------------|----------|
| tiny       | 39 MB     | No           | ~10x        | Low      |
| tiny.en    | 39 MB     | Yes          | >10x        | Low      |
| base       | 74 MB     | No           | ~7x         | Medium   |
| base.en    | 74 MB     | Yes          | >7x         | Medium   |
| small      | 244 MB    | No           | ~4x         | Good     |
| small.en   | 244 MB    | Yes          | >4x         | Good     |
| medium     | 769 MB    | No           | ~2x         | Better   |
| medium.en  | 769 MB    | Yes          | >2x         | Better   |
| large      | 1550 MB   | No           | ~1x         | Best     |
| large-v2   | 1550 MB   | No           | ~1x         | Best     |
| large-v3   | 1550 MB   | No           | ~1x         | Best     |

RTF = Real-Time Factor (lower is better) 
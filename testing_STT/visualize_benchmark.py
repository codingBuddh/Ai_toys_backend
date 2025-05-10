#!/usr/bin/env python3
"""
Visualize the benchmark results from whisper_benchmark_results_*.json files.
Creates charts comparing model performance.
"""

import os
import sys
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Directory configurations
SCRIPT_DIR = Path(__file__).parent.absolute()
PLOTS_DIR = SCRIPT_DIR / "benchmark_plots"

def load_most_recent_benchmark():
    """Load the most recent benchmark results file."""
    result_files = glob.glob(str(SCRIPT_DIR / "whisper_benchmark_results_*.json"))
    
    if not result_files:
        print("No benchmark result files found.")
        return None
    
    most_recent = max(result_files, key=os.path.getctime)
    print(f"Loading most recent benchmark results: {most_recent}")
    
    with open(most_recent, 'r') as f:
        return json.load(f)

def create_visualizations(results):
    """Create visualizations from benchmark results."""
    if not results:
        return
    
    # Filter out failed models
    successful_results = [r for r in results if r.get("load_success", False) and r.get("transcribe_success", False)]
    
    if not successful_results:
        print("No successful benchmark results to visualize.")
        return
    
    # Extract data for plotting
    models = [r["model_name"] for r in successful_results]
    sizes = [r.get("model_size_mb", 0) for r in successful_results]
    load_times = [r.get("load_time", 0) for r in successful_results]
    transcribe_times = [r.get("transcribe_time", 0) for r in successful_results]
    rtfs = [r.get("real_time_factor", 0) for r in successful_results]
    
    # Create directory for plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set up the style
    plt.style.use('ggplot')
    
    # 1. Model Sizes
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, sizes, color='skyblue')
    plt.title('Whisper Model Sizes')
    plt.xlabel('Model')
    plt.ylabel('Size (MB)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.1f}',
                 ha='center', va='bottom', rotation=0)
    
    plt.savefig(PLOTS_DIR / 'model_sizes.png', dpi=300)
    
    # 2. Load Times
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, load_times, color='lightgreen')
    plt.title('Model Loading Times')
    plt.xlabel('Model')
    plt.ylabel('Load Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}s',
                 ha='center', va='bottom', rotation=0)
    
    plt.savefig(PLOTS_DIR / 'load_times.png', dpi=300)
    
    # 3. Transcription Times
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, transcribe_times, color='salmon')
    plt.title('Audio Transcription Times')
    plt.xlabel('Model')
    plt.ylabel('Transcription Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}s',
                 ha='center', va='bottom', rotation=0)
    
    plt.savefig(PLOTS_DIR / 'transcription_times.png', dpi=300)
    
    # 4. Real-Time Factors
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, rtfs, color='mediumpurple')
    plt.title('Real-Time Factors (lower is better)')
    plt.xlabel('Model')
    plt.ylabel('RTF (x times real-time)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}x',
                 ha='center', va='bottom', rotation=0)
    
    plt.savefig(PLOTS_DIR / 'rtf.png', dpi=300)
    
    # 5. Combined Performance Comparison
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.3
    
    plt.bar(x - width/2, load_times, width, label='Load Time (s)', color='lightgreen')
    plt.bar(x + width/2, transcribe_times, width, label='Transcribe Time (s)', color='salmon')
    
    plt.title('Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Time (seconds)')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(PLOTS_DIR / 'performance_comparison.png', dpi=300)
    
    # 6. Size vs. Speed Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(sizes, rtfs, s=100, alpha=0.7, c='teal')
    
    # Add model names as annotations
    for i, model in enumerate(models):
        plt.annotate(model, (sizes[i], rtfs[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Model Size vs. Speed (RTF)')
    plt.xlabel('Size (MB)')
    plt.ylabel('Real-Time Factor (lower is better)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(PLOTS_DIR / 'size_vs_speed.png', dpi=300)
    
    print(f"Visualizations saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    # Option to specify a specific results file
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error loading file {results_file}: {e}")
            sys.exit(1)
    else:
        # Load the most recent benchmark results
        results = load_most_recent_benchmark()
    
    if results:
        create_visualizations(results)
    else:
        print("No benchmark results to visualize. Run benchmark_whisper_models.py first.") 
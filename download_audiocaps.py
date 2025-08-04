#!/usr/bin/env python3
"""
Script to download AudioCaps audio files for discriminative evaluation
"""

import os
import sys
import pandas as pd
import subprocess
from tqdm import tqdm
from datasets import load_dataset

def install_dependencies():
    """Install required dependencies"""
    try:
        import yt_dlp
    except ImportError:
        print("Installing yt-dlp...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
    
    try:
        import audiocaps_download
    except ImportError:
        print("Installing audiocaps-download...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "audiocaps-download"])

def get_required_audio_indices(discriminative_datasets):
    """Get all unique audio indices from discriminative datasets"""
    all_audio_indices = set()
    
    for dataset_name in discriminative_datasets:
        print(f"Loading {dataset_name}...")
        try:
            dataset = load_dataset(dataset_name)
            audio_indices = set(dataset['test']['audio_index'])
            all_audio_indices.update(audio_indices)
            print(f"Found {len(audio_indices)} audio indices in {dataset_name}")
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
    
    print(f"Total unique audio indices needed: {len(all_audio_indices)}")
    return all_audio_indices

def download_with_audiocaps_download(output_dir="./audiocaps", n_jobs=8):
    """Download using audiocaps-download package"""
    try:
        from audiocaps_download import Downloader
        
        print(f"Downloading AudioCaps to {output_dir} with {n_jobs} parallel jobs...")
        downloader = Downloader(root_path=output_dir, n_jobs=n_jobs)
        downloader.download(format='wav')
        print("Download completed!")
        
        # List downloaded files
        if os.path.exists(output_dir):
            wav_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
            print(f"Downloaded {len(wav_files)} audio files")
            return True
    except Exception as e:
        print(f"Error with audiocaps-download: {e}")
        return False

def download_specific_files(audio_indices, output_dir="./audiocaps"):
    """Download specific audio files using yt-dlp"""
    import yt_dlp
    
    os.makedirs(output_dir, exist_ok=True)
    
    # yt-dlp options for audio extraction
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'ignoreerrors': True,
    }
    
    downloaded = 0
    failed = 0
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for audio_id in tqdm(audio_indices, desc="Downloading audio files"):
            try:
                youtube_url = f"https://www.youtube.com/watch?v={audio_id}"
                ydl.download([youtube_url])
                downloaded += 1
            except Exception as e:
                print(f"Failed to download {audio_id}: {e}")
                failed += 1
    
    print(f"Download completed: {downloaded} successful, {failed} failed")
    return downloaded

def verify_downloads(audio_indices, output_dir="./audiocaps"):
    """Verify which audio files are available"""
    available = []
    missing = []
    
    for audio_id in audio_indices:
        audio_path = os.path.join(output_dir, f"{audio_id}.wav")
        if os.path.exists(audio_path):
            available.append(audio_id)
        else:
            missing.append(audio_id)
    
    print(f"Audio verification:")
    print(f"  Available: {len(available)}/{len(audio_indices)} ({len(available)/len(audio_indices)*100:.1f}%)")
    print(f"  Missing: {len(missing)}")
    
    if missing and len(missing) < 20:  # Show up to 20 missing files
        print(f"  Missing files: {missing[:20]}")
    
    return available, missing

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download AudioCaps audio files")
    parser.add_argument("--output_dir", type=str, default="./audiocaps",
                        help="Output directory for audio files")
    parser.add_argument("--method", choices=["audiocaps-download", "specific", "verify"], 
                        default="audiocaps-download",
                        help="Download method")
    parser.add_argument("--n_jobs", type=int, default=8,
                        help="Number of parallel download jobs")
    parser.add_argument("--install_deps", action="store_true",
                        help="Install required dependencies")
    
    args = parser.parse_args()
    
    if args.install_deps:
        install_dependencies()
    
    # Discriminative datasets
    discriminative_datasets = [
        "kuanhuggingface/AudioHallucination_AudioCaps-Random",
        "kuanhuggingface/AudioHallucination_AudioCaps-Popular", 
        "kuanhuggingface/AudioHallucination_AudioCaps-Adversarial"
    ]
    
    if args.method == "audiocaps-download":
        print("Using audiocaps-download package...")
        success = download_with_audiocaps_download(args.output_dir, args.n_jobs)
        
        if success:
            # Verify what we need vs what we have
            print("Checking which files are needed for discriminative evaluation...")
            required_indices = get_required_audio_indices(discriminative_datasets)
            verify_downloads(required_indices, args.output_dir)
    
    elif args.method == "specific":
        print("Downloading only required files...")
        required_indices = get_required_audio_indices(discriminative_datasets)
        download_specific_files(required_indices, args.output_dir)
    
    elif args.method == "verify":
        print("Verifying existing downloads...")
        required_indices = get_required_audio_indices(discriminative_datasets)
        verify_downloads(required_indices, args.output_dir)
    
    print(f"\nAudio files should be in: {os.path.abspath(args.output_dir)}")
    print("You can now run the discriminative evaluation with:")
    print(f"python discriminative_evaluation.py --audio_root_dir {args.output_dir}")

if __name__ == "__main__":
    main()
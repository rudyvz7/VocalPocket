import argparse
import json
import sys
import os
from core.ducker import duck_beat

def load_preset(name: str) -> dict:
    preset_path = os.path.join("presets", f"{name}.json")
    if not os.path.exists(preset_path):
        available = [f.replace('.json', '') for f in os.listdir('presets') if f.endswith('.json')]
        print(f"Error: Unknown preset '{name}'.")
        print(f"Available presets: {', '.join(available)}")
        sys.exit(1)
    
    with open(preset_path, "r") as f:
        data = json.load(f)
        return data.get("settings", {})

def main():
    parser = argparse.ArgumentParser(
        description="Automated vocal ducking tool that dynamically lowers beat volume when vocals are active.",
        epilog="Example usage: python main.py vocals.wav beat.wav output.wav --preset melodic"
    )
    
    # Positional arguments
    parser.add_argument("vocal", help="Path to vocal WAV file")
    parser.add_argument("beat", help="Path to beat WAV file")
    parser.add_argument("output", help="Path for output WAV file")
    
    # Optional flags
    parser.add_argument("--depth", type=float, help="Ducking depth in dB, e.g. -3 for subtle, -12 for aggressive")
    parser.add_argument("--attack", type=float, help="Attack time in milliseconds")
    parser.add_argument("--release", type=float, help="Release time in milliseconds")
    parser.add_argument("--threshold", type=float, help="RMS threshold for vocal detection, 0.0-1.0")
    parser.add_argument("--lookahead", type=float, help="Lookahead time in milliseconds")
    parser.add_argument("--preset", type=str, help="Name of a preset to load")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Default parameters if no preset and no flags are passed
    params = {
        "ducking_depth_db": -12.0,
        "attack_ms": 25.0,
        "release_ms": 150.0,
        "rms_threshold": 0.10,
        "lookahead_ms": 15.0
    }
    
    # Override with preset if provided
    if args.preset:
        preset_settings = load_preset(args.preset)
        params.update(preset_settings)
        
    # Override with explicit CLI flags
    if args.depth is not None:
        params["ducking_depth_db"] = args.depth
    if args.attack is not None:
        params["attack_ms"] = args.attack
    if args.release is not None:
        params["release_ms"] = args.release
    if args.threshold is not None:
        params["rms_threshold"] = args.threshold
    if args.lookahead is not None:
        params["lookahead_ms"] = args.lookahead

    # Execute ducking
    duck_beat(
        vocal_path=args.vocal,
        beat_path=args.beat,
        output_path=args.output,
        ducking_depth_db=params["ducking_depth_db"],
        attack_ms=params["attack_ms"],
        release_ms=params["release_ms"],
        rms_threshold=params["rms_threshold"],
        lookahead_ms=params["lookahead_ms"],
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
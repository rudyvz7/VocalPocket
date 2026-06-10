# VocalPocket

A Python tool that automatically ducks your beat whenever your vocals are active — no manual volume automation required.

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-purple)

VocalPocket solves the tedious problem of manually drawing volume automation curves to carve out space for vocals in a dense instrumental track. It uses RMS energy analysis and lookahead buffering to detect vocal transients, and applies a sigmoid-interpolated gain envelope to the beat, ducking the volume only exactly when needed.

<img width="2856" height="1688" alt="image" src="https://github.com/user-attachments/assets/ac23f03d-a215-4067-af83-212d8da238aa" />


## Installation

```bash
git clone https://github.com/rudyvz7/vocalpocket.git
cd vocalpocket
pip install -r requirements.txt
```

## Usage — GUI

```bash
python gui.py
```
Select your vocal and beat tracks, dial in a preset or tweak the sliders, and click Process Audio. You can preview the processed track live against the original beat before saving the output locally.

> **Note for browser DAW users (BandLab, Soundtrap, etc.):** Browser security prevents dragging files directly from a web app into a desktop application. Export your track to a local folder first, then drag from Windows Explorer or use the Browse button.

## Usage — CLI

```bash
python main.py vocals.wav beat.wav output.wav --preset melodic
python main.py vocals.wav beat.wav output.wav --depth -6 --attack 20 --verbose
```

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | None | Name of a preset to load (e.g. melodic, rap) |
| `--depth` | -12.0 | Ducking depth in dB, e.g. -3 for subtle, -12 for aggressive |
| `--attack` | 25.0 | Attack time in milliseconds |
| `--release` | 150.0 | Release time in milliseconds |
| `--threshold` | 0.10 | RMS threshold for vocal detection, 0.0-1.0 |
| `--lookahead` | 15.0 | Lookahead time in milliseconds |
| `--verbose` | False | Enable verbose output |

## Presets

| Preset | Description | Depth | Attack | Release |
|--------|-------------|-------|--------|---------|
| **Melodic** | Subtle ducking for floaty, whispery vocals | -3.0 dB | 25 ms | 150 ms |
| **Pop** | Balanced ducking for melodic vocals with punchy moments | -6.0 dB | 25 ms | 120 ms |
| **Podcast** | Clean ducking for spoken word and voice-over | -8.0 dB | 30 ms | 200 ms |
| **Rap** | Aggressive ducking for fast, dense syllables | -12.0 dB | 25 ms | 80 ms |

## How it works

1. **RMS Energy Analysis**: Extracts the root mean square energy from the vocal track per frame.
2. **Global 99th-Percentile Normalization**: Normalizes the RMS energy locally, discarding the top 1% of peaks to prevent transient squashing.
3. **Binary Gating**: Applies the user threshold to convert energy into a boolean active/inactive gate.
4. **Lookahead Buffering**: Shifts the gate back in time by the lookahead window to pre-duck before vocal hits.
5. **Linear Smoothing**: Steps through the binary gate and calculates a raw linear envelope based on attack/release MS.
6. **Sigmoid Curve Shaping**: Transforms the linear envelope with an S-curve for transparent, organic ducking.
7. **Sample-Level Interpolation**: Upsamples the frame-based gain envelope to match the beat's sample rate exactly.
8. **dB-to-Linear Conversion**: Re-maps the target ducking dB limit back into linear amplitude space.
9. **Export**: Prints the array logic into an optimized 24-bit PCM WAV.

## Project Structure

- `gui.py` — The CustomTkinter graphical interface and preview visualizer
- `main.py` — The CLI entry point and argument parser
- `core/analyzer.py` — Vocal RMS detection and lookahead gating
- `core/ducker.py` — Pipeline orchestrator combining analyzer and envelope algorithms
- `core/envelope.py` — Linear to Sigmoid smoothing mathematics
- `presets/` — JSON definition files for quick style setups
- `docs/` — Images and documentation assets

## License

MIT License — see LICENSE

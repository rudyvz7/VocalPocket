import librosa
import soundfile as sf
import numpy as np
from core.analyzer import detect_vocal_activity
from core.envelope import smooth_envelope

def duck_beat(
    vocal_path: str,
    beat_path: str,
    output_path: str,
    ducking_depth_db: float = -12.0,
    attack_ms: float = 25.0,
    release_ms: float = 150.0,
    rms_threshold: float = 0.10,
    lookahead_ms: float = 15.0,
    sample_rate: int = 44100
) -> None:
    """
    Full pipeline: analyzes vocal, builds gain envelope,
    applies gain reduction to beat, exports result.
    """

    # ── Step 1: Detect vocal activity ───────────────────────
    print("Step 1/4 — Analyzing vocal track...")
    activity, sr, hop_length = detect_vocal_activity(
        vocal_path,
        sample_rate=sample_rate,
        rms_threshold=rms_threshold,
        lookahead_ms=lookahead_ms
    )

    # ── Step 2: Build smooth envelope ───────────────────────
    print("Step 2/4 — Building gain envelope...")
    attack_frames  = max(int((attack_ms  / 1000.0) * sr / hop_length), 1)
    release_frames = max(int((release_ms / 1000.0) * sr / hop_length), 1)
    envelope = smooth_envelope(activity, attack_frames, release_frames)

    # ── Step 3: Load beat ────────────────────────────────────
    print("Step 3/4 — Loading beat track...")
    beat, sr = librosa.load(beat_path, sr=sample_rate, mono=False)

    # Handle mono beats — convert to stereo so output is always stereo
    if beat.ndim == 1:
        beat = np.stack([beat, beat])

    # ── Step 4: Apply gain reduction ────────────────────────
    print("Step 4/4 — Applying ducking and exporting...")

    # Upsample envelope from frame-level to sample-level
    # envelope has one value per frame, beat has one value per sample
    envelope_upsampled = np.interp(
        np.arange(beat.shape[1]),
        np.arange(len(envelope)) * hop_length,
        envelope
    )

    # Convert ducking depth from dB to a linear gain multiplier
    # e.g. -12dB → 0.25 (beat plays at 25% volume when vocal is active)
    duck_gain_linear = 10 ** (ducking_depth_db / 20.0)

    # Build the final gain curve:
    # when envelope = 0 (silence) → gain = 1.0 (full volume)
    # when envelope = 1 (vocal)   → gain = duck_gain_linear (reduced)
    gain_curve = 1.0 - (envelope_upsampled * (1.0 - duck_gain_linear))

    # Apply gain to both stereo channels
    ducked_beat = beat * gain_curve[np.newaxis, :]

    # Export as 24-bit WAV
    sf.write(output_path, ducked_beat.T, sr, subtype="PCM_24")
    print(f"\n✅ Done! Exported to: {output_path}")
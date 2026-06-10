import librosa
import numpy as np

def detect_vocal_activity(
    vocal_path: str,
    sample_rate: int = 44100,
    frame_length: int = 2048,
    hop_length: int = 512,
    rms_threshold: float = 0.02,
    lookahead_ms: float = 15.0
) -> tuple[np.ndarray, int, int]:
    """
    Analyzes a vocal audio file and returns a frame-by-frame
    activity array. 1.0 = vocal present, 0.0 = silence.

    Returns:
        activity   : float array of vocal presence per frame
        sample_rate: the sample rate used (for downstream math)
        hop_length : the hop length used (for downstream math)
    """

    print(f"Loading vocal file: {vocal_path}")
    y, sr = librosa.load(vocal_path, sr=sample_rate, mono=True)
    print(f"Loaded {len(y)} samples at {sr}Hz ({len(y)/sr:.1f} seconds)")

    # Calculate RMS energy for each frame
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # Normalize so the loudest moment = 1.0
    rms_normalized = rms / (np.max(rms) + 1e-9)

    # Gate: is the vocal loud enough to count as "active"?
    activity = (rms_normalized > rms_threshold).astype(float)

    # Lookahead: shift the activity window earlier in time
    # so ducking starts BEFORE the vocal hits
    lookahead_frames = int((lookahead_ms / 1000.0) * sr / hop_length)
    activity = np.roll(activity, -lookahead_frames)
    activity[-lookahead_frames:] = 0.0  # clean up the wrapped tail

    print(f"Vocal activity detected in {np.sum(activity)}/{len(activity)} frames")
    print(f"Approx. {np.sum(activity)/len(activity)*100:.1f}% of track has vocal activity")

    return activity, sr, hop_length
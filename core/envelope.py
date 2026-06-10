import numpy as np

def smooth_envelope(
    activity: np.ndarray,
    attack_frames: int = 5,
    release_frames: int = 20,
    curve: str = "sigmoid"
) -> np.ndarray:
    """
    Converts a binary gate [0.0 / 1.0] into a smooth gain envelope.

    Without this, gain snaps instantly between 0 and 1, creating
    audible clicks and pops on every single word boundary.

    attack_frames  : how fast the envelope rises (vocal starts)
                     fewer frames = faster duck = tighter but harsher
    release_frames : how fast the envelope falls (vocal ends)
                     more frames = slower release = more natural fade
    curve          : "sigmoid" for S-curve (natural), "linear" for straight ramp
    """

    envelope = np.zeros_like(activity, dtype=float)
    current = 0.0

    for i, target in enumerate(activity):
        if target > current:
            # Vocal just started — move toward 1.0 at attack speed
            step = 1.0 / max(attack_frames, 1)
        else:
            # Vocal just ended — move toward 0.0 at release speed
            step = 1.0 / max(release_frames, 1)

        current += np.sign(target - current) * step
        current = np.clip(current, 0.0, 1.0)
        envelope[i] = current

    if curve == "sigmoid":
        # S-curve: eases in AND out — sounds much more natural than linear
        # Maps 0→0 and 1→1 but curves the middle smoothly
        envelope = 1.0 / (1.0 + np.exp(-10.0 * (envelope - 0.5)))

    return envelope
from core.ducker import duck_beat

# ── Verified settings from testing (RudySample, Melodic/Floaty style) ──
# -3dB = natural and consistent for whispery/floaty vocals
# -6dB = too aggressive for this style (good default for Rap preset later)
# rms_threshold = 0.10 = correctly separates vocal from silence on processed vocals

duck_beat(
    vocal_path       = "RudySample.wav",
    beat_path        = "SampleBeat.wav",
    output_path      = "output_final.wav",
    ducking_depth_db = -3.0,
    attack_ms        = 25.0,
    release_ms       = 150.0,
    rms_threshold    = 0.10,
    lookahead_ms     = 15.0
)
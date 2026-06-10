import os
import json
import threading
import tempfile
import time
import shutil
import soundfile as sf
import sounddevice as sd
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
from tkinterdnd2 import TkinterDnD, DND_FILES
from core.ducker import duck_beat
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Set appearance and theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Constants
START_WIDTH = 900
START_HEIGHT = 720
MIN_WIDTH = 700
MIN_HEIGHT = 600
LEFT_PANEL_WIDTH = 340
PADDING_X = 20
PADDING_Y = 10

def build_mix(vocal: np.ndarray, beat: np.ndarray) -> np.ndarray:
    """Sum vocal and beat arrays, handle length mismatch, normalize to prevent clipping."""
    min_len = min(len(vocal), len(beat))
    mix = vocal[:min_len] + beat[:min_len]
    # Normalize to prevent clipping
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix = mix / peak
    return mix

class App(ctk.CTk, TkinterDnD.DnDWrapper):
    """
    Main application class for VocalPocket GUI.
    Handles layout, file picking, preset loading, background audio processing, 
    audio playback previews, and matplotlib waveform visualization.
    """
    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)

        self.title("VocalPocket")
        self.minsize(MIN_WIDTH, MIN_HEIGHT)
        self.geometry(f"{START_WIDTH}x{START_HEIGHT}")
        
        # Center the window on the screen
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (START_WIDTH // 2)
        y = (screen_height // 2) - (START_HEIGHT // 2)
        self.geometry(f"+{x}+{y}")

        # Grid configuration for two-column responsive layout
        self.grid_columnconfigure(0, weight=0, minsize=LEFT_PANEL_WIDTH)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # State variables
        self.vocal_path = ""
        self.beat_path = ""
        self.temp_output_path = ""
        self.verbose_var = ctk.BooleanVar(value=False)
        
        # Feature states
        self._has_processed_once = False
        self._is_processing = False
        self._suppress_reprocess = False
        self._reprocess_timer = None
        self._last_vocal_path = ""
        self._last_beat_path = ""
        
        # Playback tracking
        self.currently_playing = None  # None, 'orig_mix', 'duck_mix', 'beat_orig', 'beat_duck'
        self.audio_original_mix = None
        self.audio_ducked_mix = None
        self.audio_beat_original = None
        self.audio_beat_ducked = None
        self.playback_sample_rate = 44100
        
        self._playback_start_time = 0
        self._playback_array = None
        self._pending_resume_position: float | None = None
        self.is_playing = False
        
        self.cursor_lines = []
        self._axes = []
        self.fig = None
        self.canvas = None
        self._waveform_bg = None
        
        self.preset_data = {}
        self.load_presets()
        
        # Info strings for parameters
        self._info_texts = {
            "ducking_depth_db": "How much quieter the beat gets when your vocals are active.\n\n-3 dB is very subtle — good for soft, melodic vocals where you want the beat present.\n-12 dB is aggressive — the beat nearly disappears under the vocal.\n\nUse a lower number (closer to 0) if the ducking sounds too obvious. Use a higher number if the beat is still competing with your voice.",
            "attack_ms": "How quickly the beat ducks down when your vocals start.\n\nA fast attack (5–15 ms) is tight and punchy — good for rap where syllables hit hard.\nA slow attack (40–80 ms) is gentler — the beat fades out instead of snapping down.\n\nUse a faster attack if the beat sounds too loud at the very start of words. Use a slower one if the ducking sounds unnatural or abrupt.",
            "release_ms": "How quickly the beat comes back up after your vocals stop.\n\nA fast release (20–60 ms) snaps back immediately — can sound choppy between words.\nA slow release (150–300 ms) fades back in smoothly — more natural for singing.\n\nUse a slower release if the beat keeps cutting in and out between words. Use a faster one if there's too much silence when you pause.",
            "rms_threshold": "How loud your vocal needs to be before the ducking kicks in.\n\nA lower threshold (0.05) means even quiet breaths or room tone will trigger ducking.\nA higher threshold (0.20) means only loud, clear vocal phrases trigger it.\n\nUse a higher threshold if the beat is ducking when it shouldn't be — during pauses or breaths. Use a lower one if the beat isn't ducking early enough when vocals start.",
            "lookahead_ms": "How far ahead the tool \"looks\" to start ducking before your vocal hits.\n\nWithout lookahead, the beat ducks a split second after the vocal starts — you hear a brief blip of full-volume beat before it ducks.\nWith lookahead (10–20 ms), the duck starts just before the vocal, so the transition is seamless.\n\nLeave this at the default (15 ms) unless you notice the beat ducking too early before words start."
        }

        # Build UI
        self.build_ui()
        
        # Cleanup temp file on close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_presets(self):
        self.preset_data = {}
        presets_dir = "presets"
        if os.path.exists(presets_dir):
            for file in os.listdir(presets_dir):
                if file.endswith(".json"):
                    try:
                        with open(os.path.join(presets_dir, file), "r") as f:
                            data = json.load(f)
                            name = data.get("name", file.replace(".json", "").title())
                            self.preset_data[name] = data.get("settings", {})
                    except Exception:
                        pass
        
    def build_ui(self):
        # Left Panel (Controls) - now a ScrollableFrame to prevent clipping on small heights
        self.left_panel = ctk.CTkScrollableFrame(self, width=LEFT_PANEL_WIDTH, fg_color="transparent")
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        
        # Right Panel (Visualization)
        self.right_panel = ctk.CTkFrame(self, fg_color="#1a1a2e")
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=PADDING_X, pady=PADDING_Y)
        
        # Placeholder for Right Panel
        self.placeholder_label = ctk.CTkLabel(self.right_panel, text="Process audio to see waveform visualization", text_color="gray", font=ctk.CTkFont(size=14))
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor="center")

        # --- Populate Left Panel ---
        # Section 1 — File Inputs
        file_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        file_frame.pack(fill="x", pady=(PADDING_Y*2, PADDING_Y))
        
        self.vocal_row, self.vocal_label = self.build_file_picker(file_frame, "Vocal Track", self.browse_vocal, "vocal")
        self.beat_row, self.beat_label = self.build_file_picker(file_frame, "Beat Track", self.browse_beat, "beat")
        
        # Section 2 — Preset Selector
        preset_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        preset_frame.pack(fill="x", pady=PADDING_Y)
        
        ctk.CTkLabel(preset_frame, text="Preset", width=100, anchor="w", font=ctk.CTkFont(weight="bold")).pack(side="left")
        preset_options = ["None"] + list(self.preset_data.keys())
        self.preset_menu = ctk.CTkOptionMenu(preset_frame, values=preset_options, command=self.on_preset_change)
        self.preset_menu.pack(side="left", fill="x", expand=True, padx=(10, 0))

        # Section 3 — Parameter Sliders
        slider_frame = ctk.CTkFrame(self.left_panel)
        slider_frame.pack(fill="x", pady=PADDING_Y)
        
        self.sliders = {}
        self.slider_vars = {}
        self.slider_labels = {}
        
        self.defaults = {
            "ducking_depth_db": -12.0,
            "attack_ms": 25.0,
            "release_ms": 150.0,
            "rms_threshold": 0.10,
            "lookahead_ms": 15.0
        }
        
        self.build_slider(slider_frame, "ducking_depth_db", "Ducking Depth", -20, -1, self.defaults["ducking_depth_db"], "dB", 1)
        self.build_slider(slider_frame, "attack_ms", "Attack", 5, 100, self.defaults["attack_ms"], "ms", 0)
        self.build_slider(slider_frame, "release_ms", "Release", 20, 500, self.defaults["release_ms"], "ms", 0)
        self.build_slider(slider_frame, "rms_threshold", "Threshold", 0.01, 0.30, self.defaults["rms_threshold"], "", 2)
        self.build_slider(slider_frame, "lookahead_ms", "Lookahead", 0, 50, self.defaults["lookahead_ms"], "ms", 0)

        # Section 4 — Verbose Toggle
        toggle_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        toggle_frame.pack(fill="x", pady=PADDING_Y)
        self.verbose_switch = ctk.CTkSwitch(toggle_frame, text="Show progress in terminal", variable=self.verbose_var)
        self.verbose_switch.pack(side="left")

        # Section 5 — Run Button + Status
        self.run_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.run_frame.pack(fill="x", pady=(10, PADDING_Y))
        
        self.run_btn = ctk.CTkButton(
            self.run_frame, 
            text="Process Audio", 
            font=ctk.CTkFont(weight="bold", size=16),
            height=45,
            state="disabled",
            command=self.start_processing
        )
        self.run_btn.pack(fill="x", pady=(0, PADDING_Y))
        
        self.progress_bar = ctk.CTkProgressBar(self.run_frame, mode="indeterminate")
        
        self.status_label = ctk.CTkLabel(self.run_frame, text="", font=ctk.CTkFont(size=14))
        self.status_label.pack(fill="x", pady=(5, 0))

        # Section 6 — Post-Processing Controls (Hidden initially)
        self.post_process_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        
        # Dual Export Buttons Grid
        save_grid = ctk.CTkFrame(self.post_process_frame, fg_color="transparent")
        save_grid.pack(fill="x", pady=(0, 15))
        save_grid.grid_columnconfigure(0, weight=1)
        save_grid.grid_columnconfigure(1, weight=1)
        
        self.btn_save_ducked = ctk.CTkButton(save_grid, text="💾 Save Ducked Beat", command=self.save_ducked_beat, fg_color="#28a745", hover_color="#218838")
        self.btn_save_ducked.grid(row=0, column=0, padx=2, sticky="ew")

        self.btn_save_mix = ctk.CTkButton(save_grid, text="💾 Save Full Mix", command=self.save_full_mix, fg_color="#28a745", hover_color="#218838")
        self.btn_save_mix.grid(row=0, column=1, padx=2, sticky="ew")

        # 2x2 grid for preview buttons
        btn_grid = ctk.CTkFrame(self.post_process_frame, fg_color="transparent")
        btn_grid.pack(fill="x")
        
        btn_grid.grid_columnconfigure(0, weight=1)
        btn_grid.grid_columnconfigure(1, weight=1)
        
        self.btn_play_orig_mix = ctk.CTkButton(btn_grid, text="▶ Original Mix", command=lambda: self.toggle_play("orig_mix"))
        self.btn_play_orig_mix.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        
        self.btn_play_duck_mix = ctk.CTkButton(btn_grid, text="▶ Ducked Mix", command=lambda: self.toggle_play("duck_mix"))
        self.btn_play_duck_mix.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        
        self.btn_play_beat_orig = ctk.CTkButton(btn_grid, text="▶ Beat Only (Original)", command=lambda: self.toggle_play("beat_orig"))
        self.btn_play_beat_orig.grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        
        self.btn_play_beat_duck = ctk.CTkButton(btn_grid, text="▶ Beat Only (Ducked)", command=lambda: self.toggle_play("beat_duck"))
        self.btn_play_beat_duck.grid(row=1, column=1, padx=2, pady=2, sticky="ew")

    def _show_info_popup(self, title: str, message: str) -> None:
        if hasattr(self, '_info_popup') and self._info_popup.winfo_exists():
            self._info_popup.destroy()
        
        popup = ctk.CTkToplevel(self)
        popup.title(title)
        popup.resizable(False, False)
        popup.transient(self)
        popup.grab_set()  # Modal — user must close it before interacting with main window
        
        # Center relative to main window
        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - 160
        y = self.winfo_y() + (self.winfo_height() // 2) - 110
        popup.geometry(f"320x220+{x}+{y}")
        
        scroll_frame = ctk.CTkScrollableFrame(popup, fg_color="transparent")
        scroll_frame.pack(padx=10, pady=(10, 5), fill="both", expand=True)
        
        ctk.CTkLabel(
            scroll_frame, text=message, wraplength=260, justify="left", anchor="w"
        ).pack(padx=5, pady=5, fill="both", expand=True)
        
        ctk.CTkButton(popup, text="Got it", width=100, command=popup.destroy).pack(pady=(5, 15))
        
        self._info_popup = popup

    def build_file_picker(self, parent, label_text, command, target_type):
        row_frame = ctk.CTkFrame(parent, fg_color="transparent")
        row_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(row_frame, text=label_text, font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        
        input_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
        input_frame.pack(fill="x")
        
        val_label = ctk.CTkLabel(input_frame, text="Not selected", text_color="gray", width=250, anchor="w")
        val_label.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        btn = ctk.CTkButton(input_frame, text="Browse", width=80, command=command)
        btn.pack(side="right")
        
        # Setup Drag and Drop
        row_frame.drop_target_register(DND_FILES)
        row_frame.dnd_bind('<<Drop>>', lambda e, t=target_type, row=row_frame: self.on_drop(e, t, row))
        row_frame.dnd_bind('<<DropEnter>>', lambda e, row=row_frame: self.on_drop_enter(e, row))
        row_frame.dnd_bind('<<DropLeave>>', lambda e, row=row_frame: self.on_drop_leave(e, row))
        
        return row_frame, val_label

    def on_drop_enter(self, event, row_frame):
        row_frame.configure(border_width=2, border_color="#1f538d") # Accent highlight

    def on_drop_leave(self, event, row_frame):
        row_frame.configure(border_width=0)

    # NOTE: Drag-and-drop works from OS file managers (Windows Explorer, macOS Finder)
    # and local applications. It does NOT work from browser-based DAWs (BandLab, Soundtrap,
    # etc.) because browsers sandbox file drag events inside their own security model.
    # Users of browser DAWs should export/bounce their track to a local folder first,
    # then drag from Explorer/Finder or use the Browse button.
    def on_drop(self, event, target_type, row_frame):
        row_frame.configure(border_width=0)
        path = event.data.strip().strip('{}')
        ext = os.path.splitext(path)[1].lower()
        if ext not in ('.wav', '.mp3'):
            self.set_status("Only .wav and .mp3 files are supported", error=True)
            return
            
        if target_type == "vocal":
            self.vocal_path = path
            self.vocal_label.configure(text=self._truncate_filename(path), text_color="#E0E0E0")
        elif target_type == "beat":
            self.beat_path = path
            self.beat_label.configure(text=self._truncate_filename(path), text_color="#E0E0E0")
            
        self.update_run_button_state()
        self.reset_post_processing()

    def set_status(self, text, error=False, color=None):
        if color:
            c = color
        else:
            c = "#dc3545" if error else "#28a745"
        self.status_label.configure(text=text, text_color=c)

    def _schedule_reprocess(self) -> None:
        if self._suppress_reprocess or not self._has_processed_once:
            return
        if self._reprocess_timer:
            self.after_cancel(self._reprocess_timer)
        self._reprocess_timer = self.after(650, self._auto_reprocess)

    def build_slider(self, parent, param_key, label_text, min_val, max_val, default_val, unit, decimals):
        row_frame = ctk.CTkFrame(parent, fg_color="transparent")
        row_frame.pack(fill="x", padx=10, pady=10)
        
        row_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(row_frame, text=label_text, width=110, anchor="w").grid(row=0, column=0, sticky="w")
        
        var = ctk.DoubleVar(value=default_val)
        self.slider_vars[param_key] = var
        
        def format_val(v):
            return f"{int(v) if decimals == 0 else f'{v:.{decimals}f}'} {unit}".strip()

        readout_label = ctk.CTkLabel(row_frame, text=format_val(default_val), width=60, anchor="e")
        readout_label.grid(row=0, column=2, sticky="e", padx=(10, 5))
        self.slider_labels[param_key] = (readout_label, format_val)
        
        def on_slide(value):
            readout_label.configure(text=format_val(value))
            self._schedule_reprocess()
            
        slider = ctk.CTkSlider(row_frame, from_=min_val, to=max_val, variable=var, command=on_slide)
        slider.grid(row=0, column=1, sticky="ew", padx=10)
        
        self.sliders[param_key] = slider

        info_btn = ctk.CTkButton(
            row_frame, text="ⓘ", width=28, height=28, corner_radius=14,
            fg_color="transparent", hover_color="#2b2b2b", text_color="#888888",
            font=ctk.CTkFont(size=14),
            command=lambda p=param_key, t=label_text: self._show_info_popup(t, self._info_texts[p])
        )
        info_btn.grid(row=0, column=3, sticky="e")

    def update_run_button_state(self):
        if self.vocal_path and self.beat_path:
            self.run_btn.configure(state="normal")
        else:
            self.run_btn.configure(state="disabled")

    def _truncate_filename(self, path, max_length=45):
        filename = os.path.basename(path)
        if len(filename) > max_length:
            return filename[:max_length-3] + "..."
        return filename

    def browse_vocal(self):
        path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3"), ("WAV files", "*.wav"), ("MP3 files", "*.mp3")])
        if path:
            self.vocal_path = path
            self.vocal_label.configure(text=self._truncate_filename(path), text_color="#E0E0E0")
            self.update_run_button_state()
            self.reset_post_processing()

    def browse_beat(self):
        path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3"), ("WAV files", "*.wav"), ("MP3 files", "*.mp3")])
        if path:
            self.beat_path = path
            self.beat_label.configure(text=self._truncate_filename(path), text_color="#E0E0E0")
            self.update_run_button_state()
            self.reset_post_processing()

    def on_preset_change(self, preset_name):
        values = self.defaults if preset_name == "None" else self.preset_data.get(preset_name, self.defaults)
        self._suppress_reprocess = True
        for key, val in values.items():
            if key in self.slider_vars:
                self.slider_vars[key].set(val)
                readout_label, format_val = self.slider_labels[key]
                readout_label.configure(text=format_val(val))
        self._suppress_reprocess = False

    def reset_post_processing(self):
        self.stop_playback()
        self.post_process_frame.pack_forget()
        self.status_label.configure(text="")
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor="center")
        self._has_processed_once = False

    def save_ducked_beat(self):
        if not self.temp_output_path or not os.path.exists(self.temp_output_path):
            return
            
        default_name = "output_ducked_beat.wav"
        if self.vocal_path:
            base = os.path.splitext(os.path.basename(self.vocal_path))[0]
            default_name = f"{base}_ducked_beat.wav"
            
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            initialfile=default_name,
            filetypes=[("WAV Audio (24-bit)", "*.wav")]
        )
        if path:
            try:
                shutil.copy2(self.temp_output_path, path)
                self.set_status("✅ Ducked beat saved", color="green")
            except Exception as e:
                self.set_status(f"Error saving: {str(e)}", error=True)

    def save_full_mix(self):
        if self.audio_ducked_mix is None:
            return
            
        default_name = "output_full_mix.wav"
        if self.vocal_path:
            base = os.path.splitext(os.path.basename(self.vocal_path))[0]
            default_name = f"{base}_full_mix.wav"
            
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            initialfile=default_name,
            filetypes=[("WAV Audio (24-bit)", "*.wav")]
        )
        if path:
            try:
                sf.write(path, self.audio_ducked_mix, self.playback_sample_rate, subtype="PCM_24")
                self.set_status("✅ Full mix saved", color="green")
            except Exception as e:
                self.set_status(f"Error saving: {str(e)}", error=True)

    def toggle_play(self, audio_type):
        if self.currently_playing == audio_type:
            self.stop_playback()
            return
            
        self.stop_playback()
        self.currently_playing = audio_type
        
        self.btn_play_orig_mix.configure(text="⏹ Stop" if audio_type == "orig_mix" else "▶ Original Mix")
        self.btn_play_duck_mix.configure(text="⏹ Stop" if audio_type == "duck_mix" else "▶ Ducked Mix")
        self.btn_play_beat_orig.configure(text="⏹ Stop" if audio_type == "beat_orig" else "▶ Beat Only (Original)")
        self.btn_play_beat_duck.configure(text="⏹ Stop" if audio_type == "beat_duck" else "▶ Beat Only (Ducked)")
        
        audio_map = {
            "orig_mix": self.audio_original_mix,
            "duck_mix": self.audio_ducked_mix,
            "beat_orig": self.audio_beat_original,
            "beat_duck": self.audio_beat_ducked
        }
        
        data = audio_map.get(audio_type)
        if data is not None:
            self._playback_start_time = time.time()
            self._playback_array = data
            self.is_playing = True
            sd.play(data, samplerate=self.playback_sample_rate)
            self.update_cursor()

    def _clear_cursor_visual(self) -> None:
        """Erase the cursor line from the canvas by restoring the clean background."""
        if getattr(self, '_waveform_bg', None) is not None and self.canvas:
            self.canvas.restore_region(self._waveform_bg)
            self.canvas.blit(self.fig.bbox)

    def stop_playback(self):
        sd.stop()
        self.currently_playing = None
        self.is_playing = False
        self.btn_play_orig_mix.configure(text="▶ Original Mix")
        self.btn_play_duck_mix.configure(text="▶ Ducked Mix")
        self.btn_play_beat_orig.configure(text="▶ Beat Only (Original)")
        self.btn_play_beat_duck.configure(text="▶ Beat Only (Ducked)")
        self._clear_cursor_visual()

    def update_cursor(self):
        if not self.is_playing:
            return
            
        if getattr(self, '_waveform_bg', None) is None:
            return
            
        elapsed = time.time() - self._playback_start_time
        duration = len(self._playback_array) / self.playback_sample_rate
        
        if elapsed >= duration:
            self.stop_playback()
            return
            
        # Move lines
        for line in self.cursor_lines:
            line.set_xdata([elapsed, elapsed])
            
        # Blit background and redraw lines over it
        self.canvas.restore_region(self._waveform_bg)
        for ax, line in zip(self._axes, self.cursor_lines):
            ax.draw_artist(line)
        self.canvas.blit(self.fig.bbox)
            
        self.after(100, self.update_cursor)

    def on_waveform_click(self, event):
        if not self.is_playing or event.xdata is None or self._playback_array is None:
            return
            
        seek_sec = max(0, event.xdata)
        sd.stop()
        seek_sample = int(seek_sec * self.playback_sample_rate)
        if seek_sample < len(self._playback_array):
            sd.play(self._playback_array[seek_sample:], samplerate=self.playback_sample_rate)
            self._playback_start_time = time.time() - seek_sec

    def _get_current_params(self) -> dict:
        return {
            "ducking_depth_db": self.slider_vars["ducking_depth_db"].get(),
            "attack_ms": self.slider_vars["attack_ms"].get(),
            "release_ms": self.slider_vars["release_ms"].get(),
            "rms_threshold": self.slider_vars["rms_threshold"].get(),
            "lookahead_ms": self.slider_vars["lookahead_ms"].get(),
        }

    def _auto_reprocess(self) -> None:
        if self._is_processing:
            return
            
        was_playing = self.is_playing
        playback_position = 0.0
        if was_playing:
            playback_position = time.time() - self._playback_start_time
            self.stop_playback()
            
        self.set_status("Updating...", color="gray")
        self._is_processing = True
        self.run_btn.configure(state="disabled", text="Updating...")
        
        params = self._get_current_params()
        params["verbose"] = self.verbose_var.get()
        
        thread = threading.Thread(
            target=self._run_reprocess,
            args=(self._last_vocal_path, self._last_beat_path, params, was_playing, playback_position),
            daemon=True
        )
        thread.start()

    def _run_reprocess(self, vocal_path, beat_path, params, resume_playing, resume_position) -> None:
        try:
            result = duck_beat(
                vocal_path=vocal_path,
                beat_path=beat_path,
                output_path=self.temp_output_path,
                **params
            )
            self.after(0, lambda: self._on_reprocess_complete(result, resume_playing, resume_position))
        except Exception as e:
            self.after(0, lambda: self._on_reprocess_error(str(e)))

    def _on_reprocess_complete(self, result: dict, resume_playing: bool, resume_position: float) -> None:
        self._is_processing = False
        self.run_btn.configure(state="normal", text="Process Audio")
        
        if resume_playing and getattr(self, 'audio_ducked_mix', None) is not None:
            clipped = min(resume_position, len(self.audio_ducked_mix) / self.playback_sample_rate - 0.5)
            self._pending_resume_position = max(0.0, clipped)
        else:
            self._pending_resume_position = None

        if result:
            self.draw_waveforms(result)
            
        self.set_status("✅ Updated", color="green")

    def _on_reprocess_error(self, error_msg: str) -> None:
        self._is_processing = False
        self.run_btn.configure(state="normal", text="Process Audio")
        self.set_status(f"Error: {error_msg}", error=True)

    def start_processing(self):
        self.reset_post_processing()
        self.run_btn.configure(state="disabled", text="Processing...")
        self.progress_bar.pack(fill="x", pady=(0, 10))
        self.progress_bar.start()
        
        fd, self.temp_output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        params = self._get_current_params()
        params["verbose"] = self.verbose_var.get()
        
        self._is_processing = True
        
        threading.Thread(target=self.process_audio_thread, kwargs=params, daemon=True).start()

    def process_audio_thread(self, **kwargs):
        try:
            plot_data = duck_beat(
                vocal_path=self.vocal_path,
                beat_path=self.beat_path,
                output_path=self.temp_output_path,
                **kwargs
            )
            self.after(0, self.on_processing_success, plot_data)
        except Exception as e:
            self.after(0, self.on_processing_error, str(e))
            
    def on_processing_success(self, plot_data):
        self._is_processing = False
        self._has_processed_once = True
        self._last_vocal_path = self.vocal_path
        self._last_beat_path = self.beat_path
        
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.run_btn.configure(state="normal", text="Process Audio")
        self.set_status("✅ Processing complete!", color="green")
        
        self.post_process_frame.pack(fill="x", pady=(0, PADDING_Y))
        
        if plot_data:
            self.draw_waveforms(plot_data)

    def on_processing_error(self, error_msg):
        self._is_processing = False
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.run_btn.configure(state="normal", text="Process Audio")
        self.set_status(f"Error: {error_msg}", error=True)

    def draw_waveforms(self, data):
        self.placeholder_label.place_forget()
        
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        self.fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})
        self._axes = [ax1, ax2, ax3]
        self.fig.patch.set_facecolor('#1a1a2e')
        
        for ax in self._axes:
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            for spine in ['top', 'right', 'left']:
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color('white')
            ax.yaxis.set_visible(False)

        sr = data['sample_rate']
        self.playback_sample_rate = sr
        
        y_vocal = data['vocal_audio'].astype(np.float32)
        y_beat_orig = data['beat_original'].astype(np.float32)
        y_out = data['output_audio'].astype(np.float32)
        
        # Pre-compute Mix Arrays for playback instantly on processing completion
        self.audio_original_mix = build_mix(y_vocal, y_beat_orig).astype(np.float32)
        self.audio_ducked_mix = build_mix(y_vocal, y_out).astype(np.float32)
        self.audio_beat_original = y_beat_orig.copy()
        self.audio_beat_ducked = y_out.copy()

        time_vocal = np.arange(len(y_vocal)) / sr
        ax1.plot(time_vocal, y_vocal, color='#6c7a89', linewidth=0.5)
        ax1.set_title('Vocal', color='white', pad=5, fontsize=10)

        time_beat = np.arange(len(y_beat_orig)) / sr
        ax2.plot(time_beat, y_beat_orig, color='#4e7a5e', linewidth=0.5)
        
        gain = data['gain_curve']
        if len(gain) != len(y_beat_orig):
            gain = np.interp(np.arange(len(y_beat_orig)), np.arange(len(gain)) * len(y_beat_orig)/len(gain), gain)
            
        # Changed envelope fill color to a muted steel blue to prevent red overlap
        ax2.fill_between(time_beat, -gain, gain, color='#4a8db5', alpha=0.35)
        ax2.set_title('Beat + Ducking Envelope', color='white', pad=5, fontsize=10)

        time_out = np.arange(len(y_out)) / sr
        ax3.plot(time_out, y_out, color='#8B2232', linewidth=0.5)
        ax3.set_title('Output', color='white', pad=5, fontsize=10)
        ax3.set_xlabel('Time (seconds)', color='white')

        # Add cursor lines with animated=True for blitting
        self.cursor_lines = [
            ax1.axvline(x=0, color='red', linewidth=1.5, alpha=0.8, animated=True),
            ax2.axvline(x=0, color='red', linewidth=1.5, alpha=0.8, animated=True),
            ax3.axvline(x=0, color='red', linewidth=1.5, alpha=0.8, animated=True)
        ]

        self.fig.tight_layout(pad=1.5)

        # Update idletasks to guarantee winfo_width is populated properly before sizing canvas
        self.update_idletasks()
        fig_width_inches = max(self.right_panel.winfo_width(), 400) / self.fig.dpi
        fig_height_inches = 4.5
        self.fig.set_size_inches(fig_width_inches, fig_height_inches)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.mpl_connect('button_press_event', self.on_waveform_click)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Force synchronous draw once, then capture background later to avoid geometry races
        self.canvas.draw()
        self.after(80, self._capture_bg)

        # Explicitly lock sounddevice to this file's native SR to prevent driver mismatch glitches 
        sd.default.samplerate = self.playback_sample_rate

    def _capture_bg(self) -> None:
        if self.canvas and self.fig:
            self._waveform_bg = self.canvas.copy_from_bbox(self.fig.bbox)
            if getattr(self, '_pending_resume_position', None) is not None:
                self._do_resume_playback(self._pending_resume_position)
                self._pending_resume_position = None
                
    def _do_resume_playback(self, position_seconds: float) -> None:
        """Start playback from a specific position. Only call from _capture_bg."""
        if self.audio_ducked_mix is None:
            return
        seek_sample = max(0, int(position_seconds * self.playback_sample_rate))
        seek_sample = min(seek_sample, len(self.audio_ducked_mix) - 1)
        
        self.currently_playing = "duck_mix"
        self.btn_play_orig_mix.configure(text="▶ Original Mix")
        self.btn_play_duck_mix.configure(text="⏹ Stop")
        self.btn_play_beat_orig.configure(text="▶ Beat Only (Original)")
        self.btn_play_beat_duck.configure(text="▶ Beat Only (Ducked)")
        
        sd.play(self.audio_ducked_mix[seek_sample:], samplerate=self.playback_sample_rate)
        self._playback_start_time = time.time() - position_seconds
        self._playback_array = self.audio_ducked_mix
        self.is_playing = True
        self.update_cursor()

    def on_closing(self):
        self.stop_playback()
        if self.temp_output_path and os.path.exists(self.temp_output_path):
            try:
                os.remove(self.temp_output_path)
            except:
                pass
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()

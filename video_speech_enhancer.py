#!/usr/bin/env python3
"""
Video Speech Enhancer - AI-powered speech enhancement using DeepFilterNet2.
Real-time noise suppression while keeping human voice completely intact.

New in this version:
  - Stationary static-noise removal pass (spectral subtraction via noisereduce)
  - Adjustable noise-reduction strength slider
  - Processing profile presets (Subtle / Balanced / Aggressive / Podcast)
  - Custom output filename
  - Batch processing (process multiple videos in one click)
  - Estimated time remaining during processing
  - Cross-platform output folder open (Windows / macOS / Linux)
  - Clear log button
  - Dual waveform comparison (before / after)
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Optional, Tuple, Callable, List

import numpy as np
import soundfile as sf
import imageio_ffmpeg
import torch
import torchaudio
# DeepFilterNet is loaded lazily inside AudioProcessor._load_model()


# ---------------------------------------------------------------------------
# Processing profile presets
# ---------------------------------------------------------------------------
PRESETS = {
    "Subtle": {
        "description": "Light touch — minimal artefacts, preserves naturalness",
        "hp_cutoff_hz": 60,
        "nr_prop_decrease": 0.30,
        "static_prop_decrease": 0.40,
        "eq_2k_db": 2.0,
        "eq_3k5_db": 1.0,
    },
    "Balanced": {
        "description": "General-purpose — good noise removal with clean speech",
        "hp_cutoff_hz": 80,
        "nr_prop_decrease": 0.50,
        "static_prop_decrease": 0.55,
        "eq_2k_db": 4.0,
        "eq_3k5_db": 2.5,
    },
    "Aggressive": {
        "description": "Maximum noise removal — best for very noisy environments",
        "hp_cutoff_hz": 100,
        "nr_prop_decrease": 0.70,
        "static_prop_decrease": 0.75,
        "eq_2k_db": 6.0,
        "eq_3k5_db": 4.0,
    },
    "Podcast": {
        "description": "Optimised for voice-only content — warmth + presence",
        "hp_cutoff_hz": 80,
        "nr_prop_decrease": 0.45,
        "static_prop_decrease": 0.50,
        "eq_2k_db": 3.5,
        "eq_3k5_db": 4.0,
    },
}


class AudioProcessor:
    """Backend for audio extraction, enhancement, and muxing operations."""

    # DeepFilterNet2 – real-time speech enhancement (NOT separation).
    # Works at 48 kHz natively; far superior to SepFormer for single-speaker
    # noise-suppression (no musical-noise / chopping artefacts).
    MODEL_NAME = "DeepFilterNet2"
    TARGET_SAMPLE_RATE = 48000   # DeepFilterNet native sample rate

    def __init__(
        self,
        device: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.progress_callback = progress_callback
        self.model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        def report(stage: str, percent: float):
            if self.progress_callback:
                self.progress_callback(stage, percent)

        report("Checking for DeepFilterNet...", 10)

        # Fix for PyInstaller windowed mode (console=False sets sys.stderr/stdout
        # to None, which causes loguru inside DeepFilterNet to crash on import).
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w")

        # Auto-install deepfilternet if missing
        try:
            from df import init_df  # noqa: F401
        except ImportError:
            import subprocess
            report("Installing DeepFilterNet (first run, ~30 MB)...", 15)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "deepfilternet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        report("Loading DeepFilterNet2 model weights...", 40)
        from df import init_df
        # init_df() downloads weights on first run (~30 MB), then caches them.
        # device is handled internally by DeepFilterNet.
        self.model, self.df_state, _ = init_df()

        report("Model ready", 100)

    # ------------------------------------------------------------------
    # Audio extraction
    # ------------------------------------------------------------------

    def extract_audio(self, video_path: str, output_audio_path: str) -> Tuple[bool, str]:
        """Extract mono 16 kHz WAV from video using ffmpeg."""
        try:
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            if not os.path.exists(ffmpeg_bin):
                return False, "FFmpeg binary not found"

            cmd = [
                ffmpeg_bin,
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", str(self.TARGET_SAMPLE_RATE),
                "-y",
                output_audio_path,
            ]

            import subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

            if result.returncode != 0:
                return False, f"FFmpeg extraction failed: {result.stderr}"

            if not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) == 0:
                return False, "Extracted audio file is empty or missing"

            return True, "Audio extracted successfully"

        except Exception as e:
            return False, f"Audio extraction error: {str(e)}"

    # ------------------------------------------------------------------
    # AI enhancement (DeepFilterNet2)
    # ------------------------------------------------------------------

    def enhance_audio(
        self,
        input_audio_path: str,
        output_audio_path: str,
        progress_callback=None,
        nr_strength: float = 1.0,
    ) -> Tuple[bool, str]:
        """
        Enhance audio using DeepFilterNet2.

        DeepFilterNet2 is a causal, real-time speech enhancement model trained
        on the DNS and VoiceBank+DEMAND datasets.  Unlike SepFormer (which
        performs *speech separation* and chops up single-speaker audio),
        DeepFilterNet suppresses noise while keeping the voice completely
        intact — no musical noise, no chopping artefacts.

        Parameters
        ----------
        nr_strength : float, 0.0 – 1.0
            Maps to DeepFilterNet's ``atten_lim_db`` parameter:
              0.0  →  no suppression (pass-through)
              0.5  →  moderate (~50 dB attenuation limit)
              1.0  →  unlimited / maximum suppression (default DeepFilterNet behaviour)
        """
        try:
            from df.enhance import enhance
            from df.io import load_audio, save_audio

            if progress_callback:
                progress_callback("Loading audio into DeepFilterNet...")

            # load_audio resamples to df_state.sr() (48 kHz) automatically
            audio, _ = load_audio(input_audio_path, sr=self.df_state.sr())

            if progress_callback:
                progress_callback(
                    f"Running DeepFilterNet2 speech enhancement "
                    f"(NR strength {nr_strength:.0%})..."
                )

            # atten_lim_db: None = unlimited (max suppression);
            # a finite value caps how many dB of noise gets removed,
            # useful for preserving room ambience at lower NR strengths.
            if nr_strength >= 0.99:
                atten_lim = None          # full power
            elif nr_strength <= 0.01:
                atten_lim = 0.0           # effectively a pass-through
            else:
                atten_lim = nr_strength * 100.0   # 0 – 100 dB range

            with torch.no_grad():
                enhanced_audio = enhance(
                    self.model,
                    self.df_state,
                    audio,
                    atten_lim_db=atten_lim,
                )

            if progress_callback:
                progress_callback("Saving DeepFilterNet-enhanced audio...")

            save_audio(output_audio_path, enhanced_audio, self.df_state.sr())

            if not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) < 100:
                return False, (
                    f"Enhanced audio file not created properly "
                    f"(size: {os.path.getsize(output_audio_path) if os.path.exists(output_audio_path) else 0} bytes)"
                )

            return True, "DeepFilterNet2 enhancement completed"

        except torch.cuda.OutOfMemoryError:
            return False, "GPU out of memory. Try a shorter video or switch to CPU."
        except Exception as e:
            return False, f"Enhancement error: {str(e)}"

    # ------------------------------------------------------------------
    # Static noise removal — voice-aware
    # ------------------------------------------------------------------

    def remove_static_noise(
        self,
        input_audio_path: str,
        output_audio_path: str,
        prop_decrease: float = 0.90,
        progress_callback=None,
    ) -> Tuple[bool, str]:
        """
        Voice-aware static noise removal using quietest segment as noise profile.

        Analyzes the quietest part of the audio to build a noise fingerprint,
        then uses spectral subtraction only on non-speech segments.

        Strategy
        --------
        1. Scan entire audio to find quietest 500ms → use as noise fingerprint.
        2. Detect voiced frames using energy + spectral centroid VAD.
        3. Apply gentle spectral subtraction ONLY to non-speech frames.
        4. Use very conservative reduction (max 40%) to avoid artefacts.
        5. Crossfade at boundaries (30ms ramp) for smooth transitions.
        6. RMS-match output to input to preserve perceived loudness.
        """
        try:
            import noisereduce as nr
            from scipy.signal import butter, sosfilt

            if progress_callback:
                progress_callback("Loading audio for voice-aware static noise removal...")

            audio_data, sample_rate = sf.read(input_audio_path, dtype="float32")

            if audio_data.ndim == 2:
                audio_data = audio_data.mean(axis=1).astype(np.float32)

            total_samples = len(audio_data)

            # ----------------------------------------------------------
            # Step 1: Find the quietest 500ms window → noise fingerprint
            # ----------------------------------------------------------
            if progress_callback:
                progress_callback("Analyzing audio to find quietest segment (noise profile)...")

            # Use 500ms window for better noise profile accuracy
            win_samples = int(0.50 * sample_rate)
            step = max(1, win_samples // 4)  # finer granularity
            best_rms = float("inf")
            best_start = 0
            noise_clip = audio_data[:win_samples] if total_samples >= win_samples else audio_data.copy()

            for start in range(0, max(1, total_samples - win_samples), step):
                seg = audio_data[start : start + win_samples]
                rms = float(np.sqrt(np.mean(seg ** 2)))
                if rms < best_rms:
                    best_rms = rms
                    noise_clip = seg.copy()
                    best_start = start

            noise_floor_rms = best_rms
            noise_floor_db = 20 * np.log10(noise_floor_rms + 1e-10)

            if progress_callback:
                progress_callback(f"Noise floor: {noise_floor_db:.1f} dB (at {best_start/sample_rate:.2f}s)")

            # ----------------------------------------------------------
            # Step 2: Energy-based Voice Activity Detection (VAD)
            #
            # A frame is "speech" when its RMS exceeds the noise floor by
            # a configurable SNR threshold AND its spectral centroid sits in
            # the voiced-speech range (200 – 4000 Hz).
            # We dilate the mask by ±60 ms to protect speech onset/offset.
            # ----------------------------------------------------------
            if progress_callback:
                progress_callback("Running voice activity detection (VAD)...")

            frame_ms   = 20                               # analysis frame length
            frame_size = int(frame_ms * sample_rate / 1000)
            snr_threshold_db = 8.0                        # speech must be >8 dB above noise
            snr_linear = 10 ** (snr_threshold_db / 20.0)
            speech_rms_threshold = noise_floor_rms * snr_linear

            n_frames = max(1, total_samples // frame_size)
            is_speech_frame = np.zeros(n_frames, dtype=bool)

            fft_freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
            voiced_lo, voiced_hi = 200.0, 4000.0

            for i in range(n_frames):
                s = i * frame_size
                e = min(s + frame_size, total_samples)
                frame = audio_data[s:e]

                # RMS check
                frame_rms = float(np.sqrt(np.mean(frame ** 2)))
                if frame_rms < speech_rms_threshold:
                    continue  # too quiet → noise only

                # Spectral centroid check — voiced speech energy clusters 200–4k Hz
                mag = np.abs(np.fft.rfft(frame))
                total_mag = mag.sum()
                if total_mag < 1e-10:
                    continue
                centroid = float(np.dot(fft_freqs, mag) / total_mag)
                if voiced_lo <= centroid <= voiced_hi:
                    is_speech_frame[i] = True

            # Dilate: mark frames within ±60 ms of any speech frame as speech
            # so we don't clip the attack/decay of words.
            dilation_frames = max(1, int(60 / frame_ms))
            dilated = is_speech_frame.copy()
            for offset in range(1, dilation_frames + 1):
                dilated[offset:] |= is_speech_frame[:-offset]
                dilated[:-offset] |= is_speech_frame[offset:]
            is_speech_frame = dilated

            speech_pct = 100.0 * is_speech_frame.sum() / n_frames
            if progress_callback:
                progress_callback(
                    f"VAD: {speech_pct:.0f}% of frames detected as speech "
                    f"(those frames will NOT be filtered)"
                )

            # ----------------------------------------------------------
            # Step 3: Build a fully noise-reduced version of the whole signal
            # We will only USE these samples where VAD says silence.
            # ----------------------------------------------------------
            if progress_callback:
                progress_callback(
                    f"Applying spectral subtraction to non-speech frames "
                    f"(strength {prop_decrease:.0%})..."
                )

            denoised_full = nr.reduce_noise(
                y=audio_data,
                y_noise=noise_clip,
                sr=sample_rate,
                stationary=True,
                prop_decrease=prop_decrease,
                freq_mask_smooth_hz=300,
                time_mask_smooth_ms=40,
            ).astype(np.float32)

            # ----------------------------------------------------------
            # Step 4: Blend — speech frames keep the ORIGINAL signal,
            # silence frames get the DENOISED signal, with a short
            # raised-cosine crossfade at every transition boundary.
            # ----------------------------------------------------------
            if progress_callback:
                progress_callback("Blending: restoring original voice on speech frames...")

            output = denoised_full.copy()

            # Build a per-sample "keep original" weight (0 = use denoised, 1 = use original)
            weight = np.zeros(total_samples, dtype=np.float32)
            for i in range(n_frames):
                if is_speech_frame[i]:
                    s = i * frame_size
                    e = min(s + frame_size, total_samples)
                    weight[s:e] = 1.0

            # Raised-cosine crossfade over ±20 ms at boundaries
            fade_samples = int(0.020 * sample_rate)
            if fade_samples > 1:
                # Smooth the weight array with a simple moving average (≈ raised cosine)
                kernel = np.hanning(fade_samples * 2 + 1).astype(np.float32)
                kernel /= kernel.sum()
                weight = np.convolve(weight, kernel, mode="same")
                weight = np.clip(weight, 0.0, 1.0)

            # Blend: output = weight * original + (1 - weight) * denoised
            output = weight * audio_data + (1.0 - weight) * denoised_full

            # ----------------------------------------------------------
            # Step 5: RMS-match output to input so loudness is preserved
            # ----------------------------------------------------------
            in_rms  = float(np.sqrt(np.mean(audio_data ** 2))) + 1e-9
            out_rms = float(np.sqrt(np.mean(output ** 2)))      + 1e-9
            output  = (output * (in_rms / out_rms)).astype(np.float32)

            # ----------------------------------------------------------
            # Step 6: Peak-normalise to 0.95 FS
            # ----------------------------------------------------------
            max_val = np.max(np.abs(output))
            if max_val > 0 and max_val < 1e-6:
                return False, f"Static-removed audio is silent (max: {max_val})"
            if max_val > 0:
                output = output / max_val * 0.95

            if progress_callback:
                progress_callback("Saving voice-protected static-removed audio...")

            sf.write(output_audio_path, output, sample_rate, subtype="PCM_16")

            if not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) < 100:
                return False, "Static-removed audio not created properly"

            return True, "Voice-aware static noise removal completed"

        except ImportError as e:
            return False, f"Missing dependency: {e}. Run: pip install noisereduce scipy"
        except Exception as e:
            return False, f"Static noise removal error: {str(e)}"

    # ------------------------------------------------------------------
    # Post-processing  (high-pass + spectral gate + EQ)
    # ------------------------------------------------------------------

    def post_process_audio(
        self,
        input_audio_path: str,
        output_audio_path: str,
        preset: str = "Balanced",
        nr_strength_override: Optional[float] = None,
        progress_callback=None,
    ) -> Tuple[bool, str]:
        """
        Apply spectral noise gating and peaking EQ.

        Pipeline:
            1. High-pass filter (cutoff from preset) — kills sub-bass rumble.
            2. noisereduce non-stationary spectral gate (strength from preset or slider).
            3. Peaking EQ +N dB @ 2kHz   — consonant intelligibility.
            4. Peaking EQ +N dB @ 3.5kHz — speech air and clarity.
            5. Peak-normalise to 0.95.
        """
        try:
            import noisereduce as nr
            from scipy.signal import butter, sosfilt, lfilter

            cfg = PRESETS.get(preset, PRESETS["Balanced"])
            hp_cutoff = cfg["hp_cutoff_hz"]
            nr_prop = nr_strength_override if nr_strength_override is not None else cfg["nr_prop_decrease"]
            eq_2k_db = cfg["eq_2k_db"]
            eq_3k5_db = cfg["eq_3k5_db"]

            if progress_callback:
                progress_callback(
                    f"Post-processing with preset '{preset}' "
                    f"(NR strength {nr_prop:.0%})..."
                )

            audio_data, sample_rate = sf.read(input_audio_path, dtype="float32")

            if audio_data.ndim == 2:
                audio_data = audio_data.mean(axis=1).astype(np.float32)

            # ── Step 1: High-pass filter ──────────────────────────────
            if progress_callback:
                progress_callback(f"Applying {hp_cutoff}Hz high-pass filter...")

            sos_hp = butter(4, float(hp_cutoff), btype="highpass", fs=sample_rate, output="sos")
            audio_data = sosfilt(sos_hp, audio_data).astype(np.float32)

            # ── Step 2: Voice-aware spectral gate ────────────────────
            # Use a VAD to find the quietest window (noise-only) and only
            # apply the NR mask to frames where there is no speech, so the
            # human voice is never dampened.
            if progress_callback:
                progress_callback("Running voice-aware spectral noise gate...")

            # Find quietest 300 ms window as noise profile
            win_s = int(0.30 * sample_rate)
            step_s = max(1, win_s // 2)
            best_rms_s = float("inf")
            noise_clip = audio_data[:win_s] if len(audio_data) >= win_s else audio_data.copy()
            for st in range(0, max(1, len(audio_data) - win_s), step_s):
                seg = audio_data[st : st + win_s]
                r = float(np.sqrt(np.mean(seg ** 2)))
                if r < best_rms_s:
                    best_rms_s = r
                    noise_clip = seg.copy()

            # Simple energy VAD — mark speech frames
            frame_sz = int(0.020 * sample_rate)
            snr_thr  = best_rms_s * (10 ** (8.0 / 20.0))
            fft_f    = np.fft.rfftfreq(frame_sz, d=1.0 / sample_rate)
            n_fr     = max(1, len(audio_data) // frame_sz)
            speech_mask = np.zeros(n_fr, dtype=bool)
            for i in range(n_fr):
                s, e = i * frame_sz, min((i + 1) * frame_sz, len(audio_data))
                fr = audio_data[s:e]
                if float(np.sqrt(np.mean(fr ** 2))) >= snr_thr:
                    mag = np.abs(np.fft.rfft(fr))
                    tot = mag.sum()
                    if tot > 1e-10:
                        c = float(np.dot(fft_f, mag) / tot)
                        if 200.0 <= c <= 4000.0:
                            speech_mask[i] = True

            # Dilate ±60 ms
            dil = max(1, int(60 / 20))
            dilated_s = speech_mask.copy()
            for off in range(1, dil + 1):
                dilated_s[off:] |= speech_mask[:-off]
                dilated_s[:-off] |= speech_mask[off:]
            speech_mask = dilated_s

            # Denoise whole signal, then blend back original on speech frames
            denoised_pp = nr.reduce_noise(
                y=audio_data,
                y_noise=noise_clip,
                sr=sample_rate,
                stationary=False,
                prop_decrease=nr_prop,
                freq_mask_smooth_hz=500,
                time_mask_smooth_ms=50,
            ).astype(np.float32)

            weight_s = np.zeros(len(audio_data), dtype=np.float32)
            for i in range(n_fr):
                if speech_mask[i]:
                    s, e = i * frame_sz, min((i + 1) * frame_sz, len(audio_data))
                    weight_s[s:e] = 1.0
            fade_s = int(0.020 * sample_rate)
            if fade_s > 1:
                kern = np.hanning(fade_s * 2 + 1).astype(np.float32)
                kern /= kern.sum()
                weight_s = np.clip(np.convolve(weight_s, kern, mode="same"), 0.0, 1.0)

            audio_data = (weight_s * audio_data + (1.0 - weight_s) * denoised_pp).astype(np.float32)

            # RMS-match to pre-gate level
            pre_rms  = float(np.sqrt(np.mean(audio_data ** 2))) + 1e-9
            post_rms = float(np.sqrt(np.mean(denoised_pp ** 2))) + 1e-9
            # (audio_data already blended; just prevent level shift)
            blend_rms = float(np.sqrt(np.mean(audio_data ** 2))) + 1e-9
            if blend_rms > 1e-9:
                audio_data = (audio_data * (pre_rms / blend_rms)).astype(np.float32)

            # ── Steps 3 & 4: Peaking EQ ───────────────────────────────
            def apply_peaking_eq(
                signal: np.ndarray, center_hz: float, gain_db: float, Q: float, fs: int
            ) -> np.ndarray:
                w0 = 2.0 * np.pi * center_hz / fs
                alpha = np.sin(w0) / (2.0 * Q)
                A = 10.0 ** (gain_db / 40.0)
                b0 = 1.0 + alpha * A
                b1 = -2.0 * np.cos(w0)
                b2 = 1.0 - alpha * A
                a0 = 1.0 + alpha / A
                a1 = -2.0 * np.cos(w0)
                a2 = 1.0 - alpha / A
                b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
                a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
                return lfilter(b, a, signal.astype(np.float64)).astype(np.float32)

            if progress_callback:
                progress_callback(
                    f"Applying EQ +{eq_2k_db}dB @ 2kHz, +{eq_3k5_db}dB @ 3.5kHz..."
                )

            audio_data = apply_peaking_eq(audio_data, 2000.0, eq_2k_db, 1.5, sample_rate)
            audio_data = apply_peaking_eq(audio_data, 3500.0, eq_3k5_db, 2.0, sample_rate)

            # ── Step 5: Normalise ─────────────────────────────────────
            if progress_callback:
                progress_callback("Normalising post-processed audio...")

            max_val = np.max(np.abs(audio_data))
            if max_val > 0 and max_val < 1e-6:
                return False, f"Post-processed audio is silent (max: {max_val})"
            if max_val > 0:
                audio_data = audio_data / max_val * 0.95

            if progress_callback:
                progress_callback("Saving post-processed audio...")

            sf.write(output_audio_path, audio_data, sample_rate, subtype="PCM_16")

            if not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) < 100:
                return False, "Post-processed audio not created properly"

            return True, "Post-processing completed"

        except ImportError as e:
            return False, f"Missing dependency: {e}. Run: pip install noisereduce scipy"
        except Exception as e:
            return False, f"Post-processing error: {str(e)}"

    # ------------------------------------------------------------------
    # Muxing
    # ------------------------------------------------------------------

    def mux_video(
        self,
        original_video_path: str,
        enhanced_audio_path: str,
        output_video_path: str,
    ) -> Tuple[bool, str]:
        """Mux enhanced audio back into original video without re-encoding video."""
        try:
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

            if not os.path.exists(enhanced_audio_path):
                return False, f"Enhanced audio file not found: {enhanced_audio_path}"

            audio_file_size = os.path.getsize(enhanced_audio_path)
            if audio_file_size < 100:
                return False, f"Enhanced audio file too small: {audio_file_size} bytes"

            cmd = [
                ffmpeg_bin,
                "-i", original_video_path,
                "-i", enhanced_audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                "-y",
                output_video_path,
            ]

            import subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

            if result.returncode != 0:
                return False, f"FFmpeg muxing failed: {result.stderr}"

            if not os.path.exists(output_video_path):
                return False, "Output video file was not created"

            output_size = os.path.getsize(output_video_path)
            if output_size < 1000:
                return False, f"Output video too small: {output_size} bytes (possible corruption)"

            return True, "Video muxing completed successfully"

        except Exception as e:
            return False, f"Video muxing error: {str(e)}"


# ===========================================================================
# Waveform canvas
# ===========================================================================

class WaveformCanvas(tk.Canvas):
    """Canvas widget for displaying audio waveform preview."""

    def __init__(self, parent, color="#00ff88", **kwargs):
        super().__init__(parent, **kwargs)
        self.waveform_data: Optional[np.ndarray] = None
        self.waveform_color = color
        self.bg_color = "#0a0a0f"

    def set_waveform(self, audio_path: str) -> None:
        try:
            audio_data, _ = sf.read(audio_path, dtype="float32")
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            self.waveform_data = audio_data
            self.draw_waveform()
        except Exception:
            self.waveform_data = None

    def draw_waveform(self) -> None:
        if self.waveform_data is None:
            return

        self.delete("all")
        width = self.winfo_width()
        height = self.winfo_height()

        if width <= 1 or height <= 1:
            return

        num_samples = len(self.waveform_data)
        if num_samples == 0:
            return

        num_points = min(width, num_samples)
        samples_per_point = max(1, num_samples // num_points)

        center_y = height // 2
        max_amp = np.max(np.abs(self.waveform_data))
        if max_amp == 0:
            max_amp = 1

        scale_factor = (height / 2 - 5) / max_amp

        for i in range(num_points):
            start_idx = i * samples_per_point
            end_idx = min(start_idx + samples_per_point, num_samples)
            chunk = self.waveform_data[start_idx:end_idx]
            amplitude = np.max(np.abs(chunk)) * scale_factor
            y_top = center_y - amplitude
            y_bottom = center_y + amplitude
            self.create_line(i, y_top, i, y_bottom, fill=self.waveform_color, width=1)

        self.create_line(0, center_y, width, center_y, fill="#1a1a2e", width=1)

    def clear(self) -> None:
        self.delete("all")
        self.waveform_data = None


# ===========================================================================
# Main GUI
# ===========================================================================

class VideoSpeechEnhancerGUI:
    """Main application GUI for video speech enhancement."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Video Speech Enhancer")
        self.root.geometry("760x900")
        self.root.resizable(True, True)

        self.video_paths: List[str] = []          # batch list
        self.current_video_path: Optional[str] = None
        self.output_folder: Optional[str] = None
        self.processor: Optional[AudioProcessor] = None
        self.use_gpu = tk.BooleanVar(value=True)
        self.selected_preset = tk.StringVar(value="Balanced")
        self.nr_strength = tk.DoubleVar(value=0.85)   # 0–1
        self.custom_output_name = tk.StringVar(value="")
        self._processing_start_time: Optional[float] = None

        self._setup_styles()
        self._build_ui()
        self._initialize_processor()

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    def _setup_styles(self) -> None:
        self.colors = {
            "bg": "#0a0a0f",
            "panel": "#12121a",
            "panel_highlight": "#1a1a2e",
            "text": "#e0e0e0",
            "text_muted": "#888899",
            "accent": "#00ff88",
            "accent_hover": "#00cc6a",
            "accent2": "#00aaff",
            "error": "#ff4466",
            "warning": "#ffaa00",
            "border": "#2a2a3e",
        }

        self.root.configure(bg=self.colors["bg"])

        self.font_title = ("Segoe UI", 18, "bold")
        self.font_label = ("Segoe UI", 10)
        self.font_small = ("Segoe UI", 9)
        self.font_button = ("Segoe UI", 11, "bold")
        self.font_status = ("Consolas", 9)

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.main_frame = tk.Frame(self.root, bg=self.colors["bg"])
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self._build_header(self.main_frame)
        self._build_device_selector(self.main_frame)
        self._build_model_status_panel(self.main_frame)
        self._build_preset_and_strength(self.main_frame)
        self._build_video_select(self.main_frame)
        self._build_output_name(self.main_frame)
        self._build_waveform_preview(self.main_frame)
        self._build_progress_section(self.main_frame)
        self._build_action_buttons(self.main_frame)
        self._build_status_section(self.main_frame)

    # ── Header ─────────────────────────────────────────────────────────

    def _build_header(self, parent) -> None:
        header_frame = tk.Frame(parent, bg=self.colors["bg"])
        header_frame.pack(fill=tk.X, pady=(0, 20))

        tk.Label(
            header_frame,
            text="VIDEO SPEECH ENHANCER",
            font=self.font_title,
            fg=self.colors["accent"],
            bg=self.colors["bg"],
        ).pack(anchor=tk.W)

        tk.Label(
            header_frame,
            text="Real-time speech enhancement via DeepFilterNet2",
            font=self.font_label,
            fg=self.colors["text_muted"],
            bg=self.colors["bg"],
        ).pack(anchor=tk.W, pady=(5, 0))

    # ── Device selector ────────────────────────────────────────────────

    def _build_device_selector(self, parent) -> None:
        device_frame = tk.LabelFrame(
            parent,
            text=" Device Selection ",
            font=self.font_label,
            fg=self.colors["accent"],
            bg=self.colors["panel"],
            padx=15,
            pady=10,
        )
        device_frame.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            device_frame,
            text="Processing Device:",
            font=self.font_label,
            fg=self.colors["text"],
            bg=self.colors["panel"],
        ).pack(side=tk.LEFT, padx=(0, 15))

        tk.Radiobutton(
            device_frame,
            text="GPU (CUDA)",
            variable=self.use_gpu,
            value=True,
            font=self.font_label,
            fg=self.colors["accent"],
            bg=self.colors["panel"],
            selectcolor=self.colors["panel_highlight"],
            activebackground=self.colors["panel"],
            activeforeground=self.colors["accent"],
            command=self._on_device_change,
        ).pack(side=tk.LEFT, padx=(0, 20))

        tk.Radiobutton(
            device_frame,
            text="CPU",
            variable=self.use_gpu,
            value=False,
            font=self.font_label,
            fg=self.colors["text_muted"],
            bg=self.colors["panel"],
            selectcolor=self.colors["panel_highlight"],
            activebackground=self.colors["panel"],
            activeforeground=self.colors["text_muted"],
            command=self._on_device_change,
        ).pack(side=tk.LEFT)

        self.device_info_label = tk.Label(
            device_frame,
            text="",
            font=self.font_label,
            fg=self.colors["text_muted"],
            bg=self.colors["panel"],
        )
        self.device_info_label.pack(side=tk.RIGHT)
        self._update_device_info()

    # ── Model status panel ─────────────────────────────────────────────

    def _build_model_status_panel(self, parent) -> None:
        self.model_status_frame = tk.LabelFrame(
            parent,
            text=" Model Initialization ",
            font=self.font_label,
            fg=self.colors["accent"],
            bg=self.colors["panel"],
            padx=15,
            pady=12,
        )
        self.model_status_frame.pack(fill=tk.X, pady=(0, 12))

        self.model_status_label = tk.Label(
            self.model_status_frame,
            text="Waiting to start...",
            font=self.font_label,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            anchor=tk.W,
        )
        self.model_status_label.pack(fill=tk.X, pady=(0, 8))

        self.model_progress_var = tk.DoubleVar(value=0)
        self.model_progress_bar = tk.Canvas(
            self.model_status_frame,
            height=16,
            bg=self.colors["bg"],
            highlightthickness=0,
        )
        self.model_progress_bar.pack(fill=tk.X)

        self.model_percent_label = tk.Label(
            self.model_status_frame,
            text="0%",
            font=self.font_label,
            fg=self.colors["accent"],
            bg=self.colors["panel"],
        )
        self.model_percent_label.pack(pady=(5, 0))

    # ── Preset & strength ──────────────────────────────────────────────

    def _build_preset_and_strength(self, parent) -> None:
        frame = tk.LabelFrame(
            parent,
            text=" Processing Settings ",
            font=self.font_label,
            fg=self.colors["accent"],
            bg=self.colors["panel"],
            padx=15,
            pady=10,
        )
        frame.pack(fill=tk.X, pady=(0, 12))

        # Row 1: Presets
        preset_row = tk.Frame(frame, bg=self.colors["panel"])
        preset_row.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            preset_row,
            text="Profile:",
            font=self.font_label,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            width=10,
            anchor=tk.W,
        ).pack(side=tk.LEFT)

        for name in PRESETS:
            btn = tk.Radiobutton(
                preset_row,
                text=name,
                variable=self.selected_preset,
                value=name,
                font=self.font_small,
                fg=self.colors["accent"],
                bg=self.colors["panel"],
                selectcolor=self.colors["panel_highlight"],
                activebackground=self.colors["panel"],
                activeforeground=self.colors["accent"],
                command=self._on_preset_change,
            )
            btn.pack(side=tk.LEFT, padx=(0, 16))

        self.preset_desc_label = tk.Label(
            frame,
            text=PRESETS["Balanced"]["description"],
            font=self.font_small,
            fg=self.colors["text_muted"],
            bg=self.colors["panel"],
            anchor=tk.W,
        )
        self.preset_desc_label.pack(fill=tk.X, pady=(0, 8))

        # Row 2: Noise-reduction strength slider
        strength_row = tk.Frame(frame, bg=self.colors["panel"])
        strength_row.pack(fill=tk.X)

        tk.Label(
            strength_row,
            text="NR Strength:",
            font=self.font_label,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            width=10,
            anchor=tk.W,
        ).pack(side=tk.LEFT)

        self.nr_slider = tk.Scale(
            strength_row,
            variable=self.nr_strength,
            from_=0.10,
            to=1.00,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            length=300,
            font=self.font_small,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            troughcolor=self.colors["bg"],
            highlightthickness=0,
            activebackground=self.colors["accent"],
        )
        self.nr_slider.pack(side=tk.LEFT, padx=(0, 10))

        self.nr_strength_label = tk.Label(
            strength_row,
            text="85%",
            font=self.font_label,
            fg=self.colors["accent"],
            bg=self.colors["panel"],
            width=5,
        )
        self.nr_strength_label.pack(side=tk.LEFT)

        self.nr_strength.trace_add("write", self._on_nr_strength_change)

    # ── Video file selection (single + batch) ──────────────────────────

    def _build_video_select(self, parent) -> None:
        self._video_select_frame = tk.LabelFrame(
            parent,
            text=" Video File(s) ",
            font=self.font_label,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            padx=15,
            pady=12,
        )
        self._video_select_frame.pack(fill=tk.X, pady=(0, 12))

        self.video_path_var = tk.StringVar(value="No file selected")
        self._file_label = tk.Label(
            self._video_select_frame,
            textvariable=self.video_path_var,
            font=self.font_label,
            fg=self.colors["text_muted"],
            bg=self.colors["panel"],
            wraplength=480,
            anchor=tk.W,
        )
        self._file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_col = tk.Frame(self._video_select_frame, bg=self.colors["panel"])
        btn_col.pack(side=tk.RIGHT)

        self.select_btn = tk.Button(
            btn_col,
            text="SELECT VIDEO",
            font=self.font_button,
            fg=self.colors["bg"],
            bg=self.colors["accent"],
            activebackground=self.colors["accent_hover"],
            activeforeground=self.colors["bg"],
            cursor="hand2",
            relief=tk.FLAT,
            padx=14,
            pady=6,
            command=self._select_video,
        )
        self.select_btn.pack(pady=(0, 4))

        self.batch_btn = tk.Button(
            btn_col,
            text="ADD BATCH",
            font=self.font_small,
            fg=self.colors["text"],
            bg=self.colors["panel_highlight"],
            activebackground=self.colors["border"],
            activeforeground=self.colors["text"],
            cursor="hand2",
            relief=tk.FLAT,
            padx=14,
            pady=4,
            command=self._add_batch,
        )
        self.batch_btn.pack()

    # ── Custom output filename ─────────────────────────────────────────

    def _build_output_name(self, parent) -> None:
        frame = tk.LabelFrame(
            parent,
            text=" Output Filename (optional) ",
            font=self.font_label,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            padx=15,
            pady=10,
        )
        frame.pack(fill=tk.X, pady=(0, 12))

        row = tk.Frame(frame, bg=self.colors["panel"])
        row.pack(fill=tk.X)

        tk.Label(
            row,
            text="Save as:",
            font=self.font_label,
            fg=self.colors["text"],
            bg=self.colors["panel"],
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.output_name_entry = tk.Entry(
            row,
            textvariable=self.custom_output_name,
            font=self.font_label,
            fg=self.colors["text"],
            bg=self.colors["panel_highlight"],
            insertbackground=self.colors["accent"],
            relief=tk.FLAT,
            width=30,
        )
        self.output_name_entry.pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(
            row,
            text=".mp4   (leave blank → cleaned_<original>.mp4)",
            font=self.font_small,
            fg=self.colors["text_muted"],
            bg=self.colors["panel"],
        ).pack(side=tk.LEFT)

    # ── Waveform preview (before / after) ─────────────────────────────

    def _build_waveform_preview(self, parent) -> None:
        waveform_frame = tk.LabelFrame(
            parent,
            text=" Audio Preview ",
            font=self.font_label,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            padx=10,
            pady=8,
        )
        waveform_frame.pack(fill=tk.X, pady=(0, 12))

        row = tk.Frame(waveform_frame, bg=self.colors["panel"])
        row.pack(fill=tk.X)

        left = tk.Frame(row, bg=self.colors["panel"])
        left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        tk.Label(left, text="Before", font=self.font_small, fg=self.colors["text_muted"],
                 bg=self.colors["panel"]).pack(anchor=tk.W)
        self.waveform_before = WaveformCanvas(
            left, color="#00ff88",
            bg=self.colors["bg"], height=70,
            highlightthickness=1, highlightbackground=self.colors["border"],
        )
        self.waveform_before.pack(fill=tk.X)

        right = tk.Frame(row, bg=self.colors["panel"])
        right.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        tk.Label(right, text="After", font=self.font_small, fg=self.colors["text_muted"],
                 bg=self.colors["panel"]).pack(anchor=tk.W)
        self.waveform_after = WaveformCanvas(
            right, color="#00aaff",
            bg=self.colors["bg"], height=70,
            highlightthickness=1, highlightbackground=self.colors["border"],
        )
        self.waveform_after.pack(fill=tk.X)

        # keep old reference for backward compat
        self.waveform_canvas = self.waveform_before

    # ── Progress ───────────────────────────────────────────────────────

    def _build_progress_section(self, parent) -> None:
        progress_frame = tk.LabelFrame(
            parent,
            text=" Progress ",
            font=self.font_label,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            padx=15,
            pady=12,
        )
        progress_frame.pack(fill=tk.X, pady=(0, 12))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = tk.Canvas(
            progress_frame,
            height=20,
            bg=self.colors["bg"],
            highlightthickness=0,
        )
        self.progress_bar.pack(fill=tk.X)

        info_row = tk.Frame(progress_frame, bg=self.colors["panel"])
        info_row.pack(fill=tk.X, pady=(6, 0))

        self.progress_label = tk.Label(
            info_row,
            text="0%",
            font=self.font_label,
            fg=self.colors["accent"],
            bg=self.colors["panel"],
        )
        self.progress_label.pack(side=tk.LEFT)

        self.eta_label = tk.Label(
            info_row,
            text="",
            font=self.font_small,
            fg=self.colors["text_muted"],
            bg=self.colors["panel"],
        )
        self.eta_label.pack(side=tk.RIGHT)

        self.batch_label = tk.Label(
            info_row,
            text="",
            font=self.font_small,
            fg=self.colors["text_muted"],
            bg=self.colors["panel"],
        )
        self.batch_label.pack(side=tk.RIGHT, padx=(0, 20))

    # ── Action buttons ─────────────────────────────────────────────────

    def _build_action_buttons(self, parent) -> None:
        button_frame = tk.Frame(parent, bg=self.colors["bg"])
        button_frame.pack(fill=tk.X, pady=(0, 12))

        self.process_btn = tk.Button(
            button_frame,
            text="PROCESS VIDEO",
            font=self.font_button,
            fg=self.colors["bg"],
            bg=self.colors["accent"],
            activebackground=self.colors["accent_hover"],
            activeforeground=self.colors["bg"],
            cursor="hand2",
            relief=tk.FLAT,
            padx=30,
            pady=12,
            command=self._start_processing,
            state=tk.DISABLED,
        )
        self.process_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.open_folder_btn = tk.Button(
            button_frame,
            text="OPEN OUTPUT FOLDER",
            font=self.font_button,
            fg=self.colors["text"],
            bg=self.colors["panel_highlight"],
            activebackground=self.colors["border"],
            activeforeground=self.colors["text"],
            cursor="hand2",
            relief=tk.FLAT,
            padx=20,
            pady=12,
            command=self._open_output_folder,
            state=tk.DISABLED,
        )
        self.open_folder_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.clear_log_btn = tk.Button(
            button_frame,
            text="CLEAR LOG",
            font=self.font_small,
            fg=self.colors["text_muted"],
            bg=self.colors["panel"],
            activebackground=self.colors["border"],
            activeforeground=self.colors["text"],
            cursor="hand2",
            relief=tk.FLAT,
            padx=14,
            pady=12,
            command=self._clear_log,
        )
        self.clear_log_btn.pack(side=tk.RIGHT)

    # ── Status log ─────────────────────────────────────────────────────

    def _build_status_section(self, parent) -> None:
        status_frame = tk.LabelFrame(
            parent,
            text=" Status Log ",
            font=self.font_label,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            padx=10,
            pady=10,
        )
        status_frame.pack(fill=tk.BOTH, expand=True)

        self.status_text = tk.Text(
            status_frame,
            height=10,
            font=self.font_status,
            fg=self.colors["text"],
            bg=self.colors["bg"],
            insertbackground=self.colors["accent"],
            selectbackground=self.colors["panel_highlight"],
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(
            status_frame,
            command=self.status_text.yview,
            bg=self.colors["border"],
            troughcolor=self.colors["bg"],
            activebackground=self.colors["accent"],
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)

        self.status_label = tk.Label(
            parent,
            text="Ready",
            font=self.font_label,
            fg=self.colors["text_muted"],
            bg=self.colors["bg"],
            anchor=tk.W,
        )
        self.status_label.pack(fill=tk.X, pady=(10, 0))

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    def _on_device_change(self) -> None:
        self._update_device_info()
        device_str = "GPU" if self.use_gpu.get() else "CPU"
        self._log_status(f"Device changed to: {device_str}")
        if self.processor is not None:
            self._log_status("Reloading model on new device...")
            self._reinitialize_processor()

    def _on_preset_change(self) -> None:
        preset = self.selected_preset.get()
        desc = PRESETS[preset]["description"]
        self.preset_desc_label.config(text=desc)
        # Sync slider to preset default
        default_nr = PRESETS[preset]["nr_prop_decrease"]
        self.nr_strength.set(default_nr)

    def _on_nr_strength_change(self, *_) -> None:
        val = self.nr_strength.get()
        self.nr_strength_label.config(text=f"{val:.0%}")

    def _update_device_info(self) -> None:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if self.use_gpu.get():
                self.device_info_label.config(
                    text=f"{gpu_name} ({gpu_mem:.1f} GB)", fg=self.colors["accent"]
                )
            else:
                self.device_info_label.config(
                    text=f"GPU available: {gpu_name} ({gpu_mem:.1f} GB)",
                    fg=self.colors["text_muted"],
                )
        else:
            if self.use_gpu.get():
                self.device_info_label.config(
                    text="No GPU detected — will use CPU", fg=self.colors["warning"]
                )
            else:
                self.device_info_label.config(
                    text="CPU processing", fg=self.colors["text_muted"]
                )

    def _select_video(self) -> None:
        filetypes = [
            ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm"),
            ("MP4 files", "*.mp4"),
            ("MKV files", "*.mkv"),
            ("AVI files", "*.avi"),
            ("MOV files", "*.mov"),
            ("All files", "*.*"),
        ]
        filepath = filedialog.askopenfilename(
            title="Select Video File", filetypes=filetypes
        )
        if filepath:
            self.video_paths = [filepath]
            self.current_video_path = filepath
            filename = os.path.basename(filepath)
            self.video_path_var.set(filename)
            self.process_btn.config(state=tk.NORMAL)
            self._log_status(f"Loaded: {filename}")
            self._load_before_waveform(filepath)

    def _add_batch(self) -> None:
        filetypes = [
            ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm"),
            ("All files", "*.*"),
        ]
        filepaths = filedialog.askopenfilenames(
            title="Select Videos for Batch Processing", filetypes=filetypes
        )
        if filepaths:
            new = [p for p in filepaths if p not in self.video_paths]
            self.video_paths.extend(new)
            if not self.current_video_path and self.video_paths:
                self.current_video_path = self.video_paths[0]
                self._load_before_waveform(self.current_video_path)
            self.video_path_var.set(
                f"{os.path.basename(self.video_paths[0])} "
                f"(+{len(self.video_paths) - 1} more)" if len(self.video_paths) > 1
                else os.path.basename(self.video_paths[0])
            )
            self.process_btn.config(state=tk.NORMAL)
            self._log_status(
                f"Batch queue: {len(self.video_paths)} file(s)"
            )

    def _load_before_waveform(self, video_path: str) -> None:
        """Extract audio in background and show in the Before waveform."""
        def _do():
            if not self.processor:
                return
            temp_audio = os.path.join(
                os.path.dirname(video_path),
                f".temp_preview_{os.getpid()}.wav",
            )
            try:
                success, msg = self.processor.extract_audio(video_path, temp_audio)
                if success and os.path.exists(temp_audio):
                    # Load waveform data BEFORE deleting the file
                    import soundfile as sf
                    audio_data, _ = sf.read(temp_audio, dtype="float32")
                    if audio_data.ndim > 1:
                        audio_data = audio_data.mean(axis=1)
                    # Update waveform on main thread
                    self.root.after(0, lambda: self._set_before_waveform_data(audio_data))
                # Clean up temp file
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)
            except Exception as e:
                pass
        threading.Thread(target=_do, daemon=True).start()

    def _set_before_waveform_data(self, audio_data) -> None:
        """Set waveform data and redraw the Before canvas."""
        self.waveform_before.waveform_data = audio_data
        self.waveform_before.draw_waveform()

    # ------------------------------------------------------------------
    # Processor init / reload
    # ------------------------------------------------------------------

    def _initialize_processor(self) -> None:
        self._log_status("Initializing DeepFilterNet2 model...")
        self._update_model_status("Starting model download...", 0)
        self.root.update_idletasks()

        def model_progress_callback(stage: str, percent: float):
            self.root.after(0, lambda: self._update_model_status(stage, percent))

        def init_thread():
            try:
                device = "cuda" if self.use_gpu.get() and torch.cuda.is_available() else "cpu"
                self.processor = AudioProcessor(
                    device=device, progress_callback=model_progress_callback
                )
                self.root.after(0, lambda: self._log_status(f"Model loaded on {self.processor.device.upper()}"))
                self.root.after(0, lambda: self._log_status("Ready to process videos"))
                self.root.after(0, self._set_model_ready)
            except Exception as e:
                msg = str(e)
                self.root.after(0, lambda m=msg: self._log_status(f"Initialization error: {m}", error=True))
                self.root.after(0, lambda m=msg: self._update_model_status(f"Error: {m}", 0, error=True))

        threading.Thread(target=init_thread, daemon=True).start()

    def _reinitialize_processor(self) -> None:
        self.process_btn.config(state=tk.DISABLED)
        self.select_btn.config(state=tk.DISABLED)
        self._show_model_status_panel()
        self._update_model_status("Reloading model...", 0)

        def model_progress_callback(stage: str, percent: float):
            self.root.after(0, lambda: self._update_model_status(stage, percent))

        def reload_thread():
            try:
                device = "cuda" if self.use_gpu.get() and torch.cuda.is_available() else "cpu"
                self.processor = AudioProcessor(
                    device=device, progress_callback=model_progress_callback
                )
                self.root.after(0, lambda: self._log_status(f"Model reloaded on {device.upper()}"))
                self.root.after(0, self._set_model_ready)
                self.root.after(0, self._enable_controls)
            except Exception as e:
                msg = str(e)
                self.root.after(0, lambda m=msg: self._log_status(f"Reload error: {m}", error=True))
                self.root.after(0, lambda m=msg: self._update_model_status(f"Error: {m}", 0, error=True))
                self.root.after(0, self._enable_controls)

        threading.Thread(target=reload_thread, daemon=True).start()

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _start_processing(self) -> None:
        if not self.video_paths:
            messagebox.showerror("Error", "Please select a video file first")
            return
        if not self.processor:
            messagebox.showerror("Error", "Model not initialized yet")
            return

        self.process_btn.config(state=tk.DISABLED)
        self.select_btn.config(state=tk.DISABLED)
        self.batch_btn.config(state=tk.DISABLED)
        self._processing_start_time = time.time()
        threading.Thread(target=self._process_batch, daemon=True).start()

    def _process_batch(self) -> None:
        total = len(self.video_paths)
        for idx, vpath in enumerate(self.video_paths):
            self.root.after(
                0,
                lambda i=idx, t=total: self.batch_label.config(
                    text=f"File {i + 1}/{t}"
                ),
            )
            self._process_single(vpath, idx, total)

        self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.select_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.batch_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.batch_label.config(text=""))

        if total > 1:
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Batch Complete",
                    f"All {total} videos processed successfully!",
                ),
            )

    def _process_single(self, video_path: str, file_idx: int, total_files: int) -> None:
        temp_dir = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        extracted_audio = os.path.join(temp_dir, f".temp_audio_{base_name}_{os.getpid()}.wav")
        enhanced_audio = os.path.join(temp_dir, f".temp_enhanced_{base_name}_{os.getpid()}.wav")
        static_removed = os.path.join(temp_dir, f".temp_static_{base_name}_{os.getpid()}.wav")
        post_processed = os.path.join(temp_dir, f".temp_postproc_{base_name}_{os.getpid()}.wav")

        # Output filename
        custom = self.custom_output_name.get().strip()
        if custom:
            out_name = custom if custom.lower().endswith(".mp4") else custom + ".mp4"
            # Suffix for batch
            if total_files > 1:
                stem, ext = os.path.splitext(out_name)
                out_name = f"{stem}_{file_idx + 1}{ext}"
        else:
            out_name = f"cleaned_{base_name}.mp4"

        output_video = os.path.join(temp_dir, out_name)
        self.output_folder = temp_dir

        preset = self.selected_preset.get()
        nr_strength = self.nr_strength.get()

        def pcb(message: str):
            self.root.after(0, lambda m=message: self._log_status(f"[Enhance] {m}"))

        def spb(message: str):
            self.root.after(0, lambda m=message: self._log_status(f"[StaticNR] {m}"))

        def ppb(message: str):
            self.root.after(0, lambda m=message: self._log_status(f"[PostProc] {m}"))

        def upd(percent: float):
            # Scale per-file progress into overall batch progress
            overall = ((file_idx + percent / 100) / total_files) * 100
            self.root.after(0, lambda p=overall: self._update_progress(p))
            self._update_eta(overall)

        try:
            self._update_status(f"[{file_idx + 1}/{total_files}] Starting: {os.path.basename(video_path)}")
            upd(5)

            self._update_status("Extracting audio from video...")
            success, msg = self.processor.extract_audio(video_path, extracted_audio)
            if not success:
                raise Exception(msg)
            upd(18)

            self._update_status("AI speech enhancement (DeepFilterNet2)...")
            success, msg = self.processor.enhance_audio(extracted_audio, enhanced_audio, pcb, nr_strength=nr_strength)
            if not success:
                raise Exception(msg)
            upd(45)

            self._update_status("Removing static/background noise...")
            static_prop = PRESETS[preset]["static_prop_decrease"]
            success, msg = self.processor.remove_static_noise(
                enhanced_audio, static_removed, prop_decrease=static_prop, progress_callback=spb
            )
            if not success:
                raise Exception(msg)
            upd(65)

            self._update_status(f"Post-processing [{preset} preset]...")
            success, msg = self.processor.post_process_audio(
                static_removed, post_processed,
                preset=preset,
                nr_strength_override=nr_strength,
                progress_callback=ppb,
            )
            if not success:
                raise Exception(msg)
            upd(82)

            self._update_status("Muxing enhanced audio back to video...")
            success, msg = self.processor.mux_video(video_path, post_processed, output_video)
            if not success:
                raise Exception(msg)
            upd(100)

            # Show after waveform for last (or only) file
            if file_idx == total_files - 1 and os.path.exists(post_processed):
                after_path = post_processed  # still exists before cleanup
                self.root.after(0, lambda p=after_path: self.waveform_after.set_waveform(p))

            self._update_status("Processing complete!")
            self._log_status(f"Output saved: {out_name}", success=True)

            self.root.after(0, lambda: self.open_folder_btn.config(state=tk.NORMAL))

            if total_files == 1:
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Success",
                        f"Video processing completed!\n\nOutput: {out_name}",
                    ),
                )

        except Exception as e:
            error_msg = str(e)
            self._update_status(f"Processing failed: {error_msg}")
            self._log_status(f"ERROR: {error_msg}", error=True)
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Processing Error", f"Failed to process video:\n\n{error_msg}"
                ),
            )

        finally:
            for tmp in [extracted_audio, enhanced_audio, static_removed, post_processed]:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------

    def _update_progress(self, percent: float) -> None:
        self.progress_bar.delete("all")

        width = self.progress_bar.winfo_width()
        height = self.progress_bar.winfo_height()
        if width <= 1:
            width = 400
        if height <= 1:
            height = 20

        self.progress_bar.create_rectangle(
            0, 0, width, height, fill=self.colors["bg"], outline=self.colors["border"]
        )
        fill_width = int(width * percent / 100)
        self.progress_bar.create_rectangle(
            0, 0, fill_width, height, fill=self.colors["accent"], outline=""
        )

        self.progress_var.set(percent)
        self.progress_label.config(text=f"{percent:.0f}%")

    def _update_eta(self, percent: float) -> None:
        """Update the ETA label based on elapsed time and current progress."""
        if self._processing_start_time is None or percent <= 5:
            self.root.after(0, lambda: self.eta_label.config(text=""))
            return
        elapsed = time.time() - self._processing_start_time
        if percent >= 100:
            elapsed_str = f"{elapsed:.0f}s"
            self.root.after(0, lambda s=elapsed_str: self.eta_label.config(text=f"Done in {s}"))
            return
        rate = elapsed / percent  # seconds per percent
        remaining = rate * (100 - percent)
        if remaining < 60:
            eta_str = f"~{remaining:.0f}s remaining"
        else:
            eta_str = f"~{remaining / 60:.1f}min remaining"
        self.root.after(0, lambda s=eta_str: self.eta_label.config(text=s))

    def _update_status(self, message: str) -> None:
        self.root.after(0, lambda m=message: self.status_label.config(text=m))

    def _log_status(self, message: str, error: bool = False, success: bool = False) -> None:
        def _do():
            self.status_text.config(state=tk.NORMAL)
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            color = (
                self.colors["error"] if error
                else self.colors["accent"] if success
                else self.colors["text_muted"]
            )
            self.status_text.insert(tk.END, f"[{ts}] ", "timestamp")
            self.status_text.insert(tk.END, f"{message}\n", "message")
            self.status_text.tag_config("timestamp", foreground=self.colors["text_muted"])
            self.status_text.tag_config("message", foreground=color)
            self.status_text.see(tk.END)
            self.status_text.config(state=tk.DISABLED)

        self.root.after(0, _do)

    def _clear_log(self) -> None:
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete("1.0", tk.END)
        self.status_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Model status panel helpers
    # ------------------------------------------------------------------

    def _update_model_status(self, message: str, percent: float, error: bool = False) -> None:
        self.model_status_label.config(
            text=message,
            fg=self.colors["error"] if error else self.colors["text"],
        )
        self._update_model_progress(percent)

    def _update_model_progress(self, percent: float) -> None:
        self.model_progress_bar.delete("all")
        width = self.model_progress_bar.winfo_width()
        height = self.model_progress_bar.winfo_height()
        if width <= 1:
            width = 400
        if height <= 1:
            height = 16

        self.model_progress_bar.create_rectangle(
            0, 0, width, height, fill=self.colors["bg"], outline=self.colors["border"]
        )
        fill_width = int(width * percent / 100)
        self.model_progress_bar.create_rectangle(
            0, 0, fill_width, height, fill=self.colors["accent"], outline=""
        )
        self.model_progress_var.set(percent)
        self.model_percent_label.config(text=f"{percent:.0f}%")

    def _set_model_ready(self) -> None:
        self.model_status_label.config(
            text="Model initialized and ready", fg=self.colors["accent"]
        )
        self.model_percent_label.config(text="100%")
        self.root.after(1500, self._hide_model_status_panel)

    def _show_model_status_panel(self) -> None:
        self.model_status_frame.pack(fill=tk.X, pady=(0, 12), before=self._video_select_frame)

    def _hide_model_status_panel(self) -> None:
        self.model_status_frame.pack_forget()

    def _enable_controls(self) -> None:
        if self.video_paths:
            self.process_btn.config(state=tk.NORMAL)
        self.select_btn.config(state=tk.NORMAL)
        self.batch_btn.config(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Open output folder — cross-platform
    # ------------------------------------------------------------------

    def _open_output_folder(self) -> None:
        folder = self.output_folder
        if not folder or not os.path.exists(folder):
            messagebox.showwarning("Warning", "Output folder not found")
            return
        try:
            if sys.platform == "win32":
                os.startfile(folder)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", folder])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", folder])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {e}")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    root = tk.Tk()

    if sys.platform == "win32":
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    app = VideoSpeechEnhancerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
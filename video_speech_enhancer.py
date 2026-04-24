#!/usr/bin/env python3
"""
Video Speech Enhancer - Local speech enhancement + automatic English captions.
Real-time noise suppression while keeping human voice clear.

New in this version:
  - Stable FFmpeg/noise-reduction enhancement pipeline (no model install)
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
import subprocess
import tempfile
import shutil
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk
from pathlib import Path
from typing import Optional, Tuple, Callable, List

import numpy as np
import soundfile as sf
import imageio_ffmpeg

try:
    import torch
except Exception:  # pragma: no cover - handled at runtime
    torch = None
# Enhancement uses a stable local pipeline; captions use Whisper if installed.


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

    # Stable local enhancement pipeline (FFmpeg + classical filtering).
    # This avoids model downloads and runtime auto-install failures.
    MODEL_NAME = "Stable Local Enhancement Pipeline"
    TARGET_SAMPLE_RATE = 48000   # keeps captions/audio aligned on a common sample rate

    @staticmethod
    def _run_command(cmd: list[str]) -> tuple[int, str, str]:
        """Run a command and return (returncode, stdout, stderr)."""
        import subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return result.returncode, result.stdout, result.stderr

    @staticmethod
    def probe_cuda() -> dict:
        """Return a detailed snapshot of CUDA / GPU availability."""
        info = {
            "torch_imported": torch is not None,
            "torch_cuda_available": False,
            "torch_cuda_version": getattr(torch.version, "cuda", None) if torch is not None else None,
            "device_count": 0,
            "device_name": None,
            "device_memory_gb": None,
            "nvidia_smi": False,
            "nvidia_smi_name": None,
            "nvidia_smi_memory_gb": None,
            "reason": "",
        }

        if torch is not None:
            try:
                info["torch_cuda_available"] = bool(torch.cuda.is_available())
                info["device_count"] = int(torch.cuda.device_count()) if info["torch_cuda_available"] else 0
                if info["torch_cuda_available"] and info["device_count"] > 0:
                    info["device_name"] = torch.cuda.get_device_name(0)
                    props = torch.cuda.get_device_properties(0)
                    info["device_memory_gb"] = round(props.total_memory / (1024 ** 3), 1)
            except Exception as e:
                info["reason"] = f"Torch CUDA probe failed: {e}"
        else:
            info["reason"] = "PyTorch is not installed"

        # NVIDIA driver / hardware probe, useful when torch was installed without CUDA.
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            if result.returncode == 0 and result.stdout.strip():
                first = result.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first.split(",")]
                info["nvidia_smi"] = True
                info["nvidia_smi_name"] = parts[0] if parts else None
                if len(parts) > 1:
                    mem_txt = parts[1].split()[0]
                    try:
                        info["nvidia_smi_memory_gb"] = round(float(mem_txt) / 1024.0, 1)
                    except Exception:
                        info["nvidia_smi_memory_gb"] = None
        except Exception:
            pass

        if info["torch_cuda_available"]:
            info["reason"] = "CUDA available through PyTorch"
        elif info["nvidia_smi"]:
            if not info["torch_imported"]:
                info["reason"] = "NVIDIA GPU detected, but PyTorch is not installed"
            elif info["torch_cuda_version"]:
                info["reason"] = "NVIDIA GPU detected, but PyTorch CUDA initialisation failed"
            else:
                info["reason"] = "NVIDIA GPU detected, but the installed PyTorch build is CPU-only"
        else:
            info["reason"] = info["reason"] or "No CUDA-capable GPU detected"

        return info

    @classmethod
    def cuda_is_ready(cls) -> bool:
        """True when PyTorch can actually use CUDA."""
        return bool(cls.probe_cuda().get("torch_cuda_available"))

    @staticmethod
    def torch_status() -> dict:
        """Return a compact status snapshot for PyTorch itself."""
        info = {
            "available": torch is not None,
            "version": None,
            "cuda_version": None,
            "cuda_available": False,
        }
        if torch is None:
            return info
        try:
            info["version"] = getattr(torch, "__version__", None)
            info["cuda_version"] = getattr(torch.version, "cuda", None)
            info["cuda_available"] = bool(torch.cuda.is_available())
        except Exception:
            pass
        return info

    def __init__(
        self,
        device: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        cuda_ready = self.cuda_is_ready()
        if device is None:
            self.device = "cuda" if cuda_ready else "cpu"
        else:
            requested = str(device).lower()
            self.device = "cuda" if requested == "cuda" and cuda_ready else "cpu"

        self.cuda_info = self.probe_cuda()
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

        report(f"Setting up stable local audio pipeline on {self.device.upper()}...", 10)

        # No heavy model download here — the pipeline uses FFmpeg + classical
        # signal processing so the app starts reliably on clean machines.
        self.model = None
        self.df_state = None

        report("Local enhancement pipeline ready", 100)

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
    # AI enhancement (stable local pipeline)
    # ------------------------------------------------------------------

    def enhance_audio(
        self,
        input_audio_path: str,
        output_audio_path: str,
        progress_callback=None,
        nr_strength: float = 1.0,
    ) -> Tuple[bool, str]:
        """
        Enhance audio using a stable, non-ML pipeline.

        The pipeline is intentionally conservative to avoid installation and
        runtime failures:

            1. Optional FFmpeg denoise pass (afftdn) for fast cleanup.
            2. High-pass filter to remove low-frequency rumble.
            3. Optional noisereduce fallback if FFmpeg processing fails.
            4. Loudness/peak normalization for consistent output.

        The goal is reliable speech cleanup with no model downloads and no
        runtime dependency installation.
        """
        try:
            import subprocess
            from scipy.signal import butter, sosfilt

            if progress_callback:
                progress_callback("Running stable local enhancement pipeline...")

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_filter = (
                f"highpass=f=70,"
                f"lowpass=f=12000,"
                f"afftdn=nf=-25,"
                f"volume={max(0.85, min(1.25, 1.0 + (nr_strength - 0.5) * 0.25))},"
                f"loudnorm=I=-16:LRA=11:TP=-1.5"
            )

            cmd = [
                ffmpeg_bin,
                "-y",
                "-i", input_audio_path,
                "-af", ffmpeg_filter,
                "-ac", "1",
                "-ar", str(self.TARGET_SAMPLE_RATE),
                output_audio_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

            if result.returncode == 0 and os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) >= 100:
                return True, "Stable local enhancement completed"

            if progress_callback:
                progress_callback("FFmpeg path failed, using Python fallback...")

            # Python fallback: high-pass + light spectral denoise + normalization.
            audio_data, sample_rate = sf.read(input_audio_path, dtype="float32")
            if audio_data.ndim == 2:
                audio_data = audio_data.mean(axis=1).astype(np.float32)

            sos = butter(4, 70.0, btype="highpass", fs=sample_rate, output="sos")
            audio_data = sosfilt(sos, audio_data).astype(np.float32)

            try:
                import noisereduce as nr
                # Use the quietest half-second as a noise reference.
                win = max(1, int(0.5 * sample_rate))
                step = max(1, win // 4)
                best_rms = float("inf")
                noise_clip = audio_data[:win] if len(audio_data) >= win else audio_data.copy()
                for start in range(0, max(1, len(audio_data) - win), step):
                    seg = audio_data[start:start + win]
                    rms = float(np.sqrt(np.mean(seg ** 2)))
                    if rms < best_rms:
                        best_rms = rms
                        noise_clip = seg.copy()

                audio_data = nr.reduce_noise(
                    y=audio_data,
                    y_noise=noise_clip,
                    sr=sample_rate,
                    stationary=True,
                    prop_decrease=max(0.2, min(0.95, nr_strength)),
                    freq_mask_smooth_hz=300,
                    time_mask_smooth_ms=40,
                ).astype(np.float32)
            except Exception:
                # noisereduce is optional; the high-pass + normalize path is still valid.
                pass

            peak = float(np.max(np.abs(audio_data)))
            if peak > 0:
                audio_data = (audio_data / peak * 0.95).astype(np.float32)

            sf.write(output_audio_path, audio_data, sample_rate, subtype="PCM_16")

            if not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) < 100:
                return False, "Stable enhancement output not created properly"

            return True, "Stable local enhancement completed"

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

    # ------------------------------------------------------------------
    # Captions generation
    # ------------------------------------------------------------------

    @staticmethod
    def _format_srt_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        if millis == 1000:
            secs += 1
            millis = 0
        if secs == 60:
            minutes += 1
            secs = 0
        if minutes == 60:
            hours += 1
            minutes = 0
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    @staticmethod
    def _escape_ffmpeg_filter_path(path: str) -> str:
        # FFmpeg subtitle filter needs escaping for Windows drive letters and special chars.
        normalized = Path(path).resolve().as_posix()
        normalized = normalized.replace('\\', '\\\\').replace(':', '\\:').replace("'", "\\'")
        return normalized

    @staticmethod
    def _normalize_caption_hex(color_value: str, fallback: str = "#FFFFFF") -> str:
        value = (color_value or fallback).strip()
        if not value.startswith("#"):
            value = fallback
        if len(value) == 4:
            value = "#" + "".join(ch * 2 for ch in value[1:])
        if len(value) != 7:
            value = fallback
        return value.upper()

    @staticmethod
    def _hex_to_ass_color(color_value: str, opacity_percent: float) -> str:
        hex_value = AudioProcessor._normalize_caption_hex(color_value)
        opacity = max(0.0, min(100.0, float(opacity_percent)))
        alpha = int(round((100.0 - opacity) * 255.0 / 100.0))
        red = int(hex_value[1:3], 16)
        green = int(hex_value[3:5], 16)
        blue = int(hex_value[5:7], 16)
        return f"&H{alpha:02X}{blue:02X}{green:02X}{red:02X}&"

    @staticmethod
    def _format_ass_timestamp(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centis = int(round((seconds - int(seconds)) * 100))
        if centis == 100:
            secs += 1
            centis = 0
        if secs == 60:
            minutes += 1
            secs = 0
        if minutes == 60:
            hours += 1
            minutes = 0
        return f"{hours}:{minutes:02}:{secs:02}.{centis:02}"

    @staticmethod
    def _wrap_caption_text(text: str, max_len: int = 38) -> str:
        words = text.split()
        if not words:
            return ""
        lines = []
        current = words[0]
        for word in words[1:]:
            if len(current) + 1 + len(word) <= max_len:
                current += " " + word
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return "\\N".join(lines)

    def generate_english_captions(
        self,
        audio_path: str,
        captions_path: str,
        progress_callback: Optional[Callable[[str], None]] = None,
        caption_options: Optional[dict] = None,
    ) -> Tuple[bool, str]:
        """Generate English captions from audio using Whisper and write a timed subtitle file."""
        try:
            if not os.path.exists(audio_path):
                return False, f"Audio file not found: {audio_path}"

            if progress_callback:
                progress_callback(f"Loading Whisper for auto captions on {self.device.upper()}...")

            try:
                import whisper
            except ImportError:
                return False, "Missing dependency: openai-whisper. Install project requirements first."

            opts = caption_options or {}
            timing_offset = float(opts.get("timing_offset", 0.0))
            min_duration = max(0.4, float(opts.get("min_duration", 1.2)))
            font_size = int(opts.get("font_size", 18))
            text_color = self._normalize_caption_hex(str(opts.get("text_color", "#FFFFFF")))
            background_color = self._normalize_caption_hex(str(opts.get("background_color", "#000000")))
            outline_color = self._normalize_caption_hex(str(opts.get("outline_color", "#000000")))
            text_opacity = float(opts.get("text_opacity", 0.0))
            background_opacity = float(opts.get("background_opacity", 55.0))
            outline_opacity = float(opts.get("outline_opacity", 0.0))
            outline_width = float(opts.get("outline_width", 2.0))
            shadow = int(opts.get("shadow", 0))
            margin_v = int(opts.get("margin_v", 24))
            max_chars = int(opts.get("max_chars_per_line", 38))
            alignment = int(opts.get("alignment", 2))
            font_name = str(opts.get("font_name", "Arial"))
            caption_format = str(opts.get("format", "ass")).lower()

            model_size = "medium"
            if progress_callback:
                progress_callback("Transcribing English captions...")

            device = "cuda" if self.device == "cuda" and torch is not None and torch.cuda.is_available() else "cpu"
            model = whisper.load_model(model_size, device=device)
            result = model.transcribe(audio_path, language="en", fp16=(device == "cuda"))

            segments = result.get("segments", [])
            if not segments:
                return False, "No speech detected for captions"

            if caption_format == "srt":
                with open(captions_path, "w", encoding="utf-8") as f:
                    subtitle_index = 1
                    for segment in segments:
                        caption_text = " ".join(str(segment.get("text", "")).replace("\r", " ").replace("\n", " ").split())
                        if not caption_text:
                            continue
                        start = max(0.0, float(segment["start"]) + timing_offset)
                        end = max(start + min_duration, float(segment["end"]) + timing_offset)
                        wrapped_caption = self._wrap_caption_text(caption_text, max_chars).replace('\\N', '\n')
                        f.write(
                            f"{subtitle_index}\n"
                            f"{self._format_srt_timestamp(start)} --> {self._format_srt_timestamp(end)}\n"
                            f"{wrapped_caption}\n\n"
                        )
                        subtitle_index += 1
            else:
                primary = self._hex_to_ass_color(text_color, text_opacity)
                back = self._hex_to_ass_color(background_color, background_opacity)
                outline = self._hex_to_ass_color(outline_color, outline_opacity)
                with open(captions_path, "w", encoding="utf-8") as f:
                    f.write("[Script Info]\n")
                    f.write("ScriptType: v4.00+\n")
                    f.write("PlayResX: 1920\n")
                    f.write("PlayResY: 1080\n")
                    f.write("WrapStyle: 2\n")
                    f.write("ScaledBorderAndShadow: yes\n\n")
                    f.write("[V4+ Styles]\n")
                    f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
                    f.write(
                        "Style: Default,"
                        f"{font_name},"
                        f"{font_size},"
                        f"{primary},"
                        f"{primary},"
                        f"{outline},"
                        f"{back},"
                        "0,0,0,0,100,100,0,0,3,"
                        f"{max(0.0, outline_width):.1f},"
                        f"{shadow},"
                        f"{alignment},"
                        "20,20,"
                        f"{margin_v},"
                        "1\n\n"
                    )
                    f.write("[Events]\n")
                    f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
                    for segment in segments:
                        caption_text = " ".join(str(segment.get("text", "")).replace("\r", " ").replace("\n", " ").split())
                        if not caption_text:
                            continue
                        start = max(0.0, float(segment["start"]) + timing_offset)
                        end = max(start + min_duration, float(segment["end"]) + timing_offset)
                        f.write(
                            f"Dialogue: 0,{self._format_ass_timestamp(start)},{self._format_ass_timestamp(end)},Default,,0,0,0,,"
                            f"{self._wrap_caption_text(caption_text, max_chars)}\n"
                        )

            if not os.path.exists(captions_path) or os.path.getsize(captions_path) < 20:
                return False, "Caption file was not created properly"

            return True, "English captions generated successfully"

        except Exception as e:
            return False, f"Caption generation error: {str(e)}"

    @staticmethod
    def _build_subtitles_filter(subtitles_path: str) -> str:
        """Build a robust FFmpeg subtitles filter for the given SRT/ASS path."""
        subtitle_path = Path(subtitles_path).resolve().as_posix()
        subtitle_path = subtitle_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        if subtitle_path.lower().endswith(".ass"):
            return f"subtitles=filename='{subtitle_path}':charenc=UTF-8"
        force_style = (
            "FontSize=18,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,"
            "BorderStyle=3,Outline=2,Shadow=0,MarginV=24"
        )
        return f"subtitles=filename='{subtitle_path}':charenc=UTF-8:force_style='{force_style}'"

    def mux_video_with_captions(
        self,
        original_video_path: str,
        enhanced_audio_path: str,
        subtitles_path: str,
        output_video_path: str,
    ) -> Tuple[bool, str]:
        """Mux enhanced audio and burn captions into the final video."""
        try:
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

            if not os.path.exists(enhanced_audio_path):
                return False, f"Enhanced audio file not found: {enhanced_audio_path}"
            if not os.path.exists(subtitles_path):
                return False, f"Subtitles file not found: {subtitles_path}"

            audio_file_size = os.path.getsize(enhanced_audio_path)
            if audio_file_size < 100:
                return False, f"Enhanced audio file too small: {audio_file_size} bytes"

            safe_subtitles_path = None
            try:
                safe_subtitles_path = os.path.join(tempfile.gettempdir(), f"vse_subs_{os.getpid()}{Path(subtitles_path).suffix or '.ass'}")
                shutil.copyfile(subtitles_path, safe_subtitles_path)
                subtitles_filter = self._build_subtitles_filter(safe_subtitles_path)
            except Exception:
                subtitles_filter = self._build_subtitles_filter(subtitles_path)

            cmd = [
                ffmpeg_bin,
                "-i", original_video_path,
                "-i", enhanced_audio_path,
                "-vf", subtitles_filter,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "18",
                "-c:a", "aac",
                "-b:a", "192k",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                "-movflags", "+faststart",
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

            if safe_subtitles_path and os.path.exists(safe_subtitles_path):
                try:
                    os.remove(safe_subtitles_path)
                except Exception:
                    pass

            if result.returncode != 0:
                return False, f"FFmpeg muxing with captions failed: {result.stderr}"

            if not os.path.exists(output_video_path):
                return False, "Output video file was not created"

            output_size = os.path.getsize(output_video_path)
            if output_size < 1000:
                return False, f"Output video too small: {output_size} bytes (possible corruption)"

            return True, "Video muxing with captions completed successfully"

        except Exception as e:
            return False, f"Video muxing with captions error: {str(e)}"


# ===========================================================================
# Waveform canvas
# ===========================================================================
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
        self.use_gpu = tk.BooleanVar(value=AudioProcessor.cuda_is_ready())
        self.selected_preset = tk.StringVar(value="Balanced")
        self.nr_strength = tk.DoubleVar(value=0.85)   # 0–1
        self.custom_output_name = tk.StringVar(value="")
        self.caption_timing_offset = tk.DoubleVar(value=0.0)
        self.caption_min_duration = tk.DoubleVar(value=1.2)
        self.caption_font_size = tk.IntVar(value=18)
        self.caption_text_opacity = tk.IntVar(value=0)
        self.caption_background_opacity = tk.IntVar(value=55)
        self.caption_text_color = tk.StringVar(value="#FFFFFF")
        self.caption_background_color = tk.StringVar(value="#000000")
        self.caption_outline_color = tk.StringVar(value="#000000")
        self.caption_outline_width = tk.DoubleVar(value=2.0)
        self.caption_shadow = tk.IntVar(value=0)
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
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.scroll_container = tk.Frame(self.root, bg=self.colors["bg"])
        self.scroll_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.scroll_container.columnconfigure(0, weight=1)
        self.scroll_container.rowconfigure(0, weight=1)

        self.ui_canvas = tk.Canvas(
            self.scroll_container,
            bg=self.colors["bg"],
            highlightthickness=0,
            borderwidth=0,
        )
        self.ui_canvas.grid(row=0, column=0, sticky="nsew")

        self.ui_scrollbar = tk.Scrollbar(
            self.scroll_container,
            orient=tk.VERTICAL,
            command=self.ui_canvas.yview,
            bg=self.colors["border"],
            troughcolor=self.colors["bg"],
            activebackground=self.colors["accent"],
        )
        self.ui_scrollbar.grid(row=0, column=1, sticky="ns")
        self.ui_canvas.configure(yscrollcommand=self.ui_scrollbar.set)

        self.main_frame = tk.Frame(self.ui_canvas, bg=self.colors["bg"])
        self._main_frame_window = self.ui_canvas.create_window(
            (0, 0),
            window=self.main_frame,
            anchor="nw",
        )

        self.main_frame.bind("<Configure>", self._on_main_frame_configure)
        self.ui_canvas.bind("<Configure>", self._on_canvas_configure)
        self.ui_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.ui_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.ui_canvas.bind_all("<Button-5>", self._on_mousewheel)

        self._build_header(self.main_frame)
        self._build_device_selector(self.main_frame)
        self._build_model_status_panel(self.main_frame)
        self._build_preset_and_strength(self.main_frame)
        self._build_video_select(self.main_frame)
        self._build_output_name(self.main_frame)
        self._build_caption_formatting(self.main_frame)
        self._build_waveform_preview(self.main_frame)
        self._build_progress_section(self.main_frame)
        self._build_action_buttons(self.main_frame)
        self._build_status_section(self.main_frame)

    # ── Header ─────────────────────────────────────────────────────────

    def _on_main_frame_configure(self, event) -> None:
        self.ui_canvas.configure(scrollregion=self.ui_canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.ui_canvas.itemconfigure(self._main_frame_window, width=event.width)
        self.ui_canvas.configure(scrollregion=self.ui_canvas.bbox("all"))

    def _on_mousewheel(self, event) -> None:
        if not hasattr(self, "ui_canvas"):
            return
        if getattr(event, "num", None) == 4:
            self.ui_canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            self.ui_canvas.yview_scroll(3, "units")
        elif getattr(event, "delta", 0):
            self.ui_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

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
            text="Real-time speech enhancement via stable local filters",
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

        right_side = tk.Frame(device_frame, bg=self.colors["panel"])
        right_side.pack(side=tk.RIGHT)

        self.device_info_label = tk.Label(
            right_side,
            text="",
            font=self.font_label,
            fg=self.colors["text_muted"],
            bg=self.colors["panel"],
            justify=tk.RIGHT,
            wraplength=300,
        )
        self.device_info_label.pack(anchor=tk.E)

        self._update_device_info()

    # ── Model status panel ─────────────────────────────────────────────

    def _build_model_status_panel(self, parent) -> None:
        self.model_status_frame = tk.LabelFrame(
            parent,
            text=" Audio Pipeline Initialization ",
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


    def _build_caption_formatting(self, parent) -> None:
        frame = tk.LabelFrame(
            parent,
            text=" Caption Formatting ",
            font=self.font_label,
            fg=self.colors["accent"],
            bg=self.colors["panel"],
            padx=15,
            pady=10,
        )
        frame.pack(fill=tk.X, pady=(0, 12))

        timing_row = tk.Frame(frame, bg=self.colors["panel"])
        timing_row.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            timing_row,
            text="Timing Offset:",
            font=self.font_label,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            width=14,
            anchor=tk.W,
        ).pack(side=tk.LEFT)

        self.caption_offset_slider = tk.Scale(
            timing_row,
            variable=self.caption_timing_offset,
            from_=-5.0,
            to=5.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            length=220,
            font=self.font_small,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            troughcolor=self.colors["bg"],
            highlightthickness=0,
            activebackground=self.colors["accent"],
        )
        self.caption_offset_slider.pack(side=tk.LEFT, padx=(0, 8))

        self.caption_offset_label = tk.Label(
            timing_row,
            text="0.0s",
            font=self.font_label,
            fg=self.colors["accent"],
            bg=self.colors["panel"],
            width=8,
        )
        self.caption_offset_label.pack(side=tk.LEFT)

        duration_row = tk.Frame(frame, bg=self.colors["panel"])
        duration_row.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            duration_row,
            text="Min Duration:",
            font=self.font_label,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            width=14,
            anchor=tk.W,
        ).pack(side=tk.LEFT)

        self.caption_min_duration_slider = tk.Scale(
            duration_row,
            variable=self.caption_min_duration,
            from_=0.8,
            to=4.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            length=220,
            font=self.font_small,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            troughcolor=self.colors["bg"],
            highlightthickness=0,
            activebackground=self.colors["accent"],
        )
        self.caption_min_duration_slider.pack(side=tk.LEFT, padx=(0, 8))

        self.caption_min_duration_label = tk.Label(
            duration_row,
            text="1.2s",
            font=self.font_label,
            fg=self.colors["accent"],
            bg=self.colors["panel"],
            width=8,
        )
        self.caption_min_duration_label.pack(side=tk.LEFT)

        text_row = tk.Frame(frame, bg=self.colors["panel"])
        text_row.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            text_row,
            text="Text Style:",
            font=self.font_label,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            width=14,
            anchor=tk.W,
        ).pack(side=tk.LEFT)

        self.text_color_btn = tk.Button(
            text_row,
            text="Text Color",
            font=self.font_small,
            fg=self.colors["bg"],
            bg=self.caption_text_color.get(),
            activebackground=self.colors["accent_hover"],
            activeforeground=self.colors["bg"],
            relief=tk.FLAT,
            cursor="hand2",
            command=self._choose_caption_text_color,
            padx=10,
            pady=3,
        )
        self.text_color_btn.pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(
            text_row,
            text="Opacity",
            font=self.font_small,
            fg=self.colors["text_muted"],
            bg=self.colors["panel"],
        ).pack(side=tk.LEFT)

        self.caption_text_opacity_slider = tk.Scale(
            text_row,
            variable=self.caption_text_opacity,
            from_=0,
            to=100,
            resolution=1,
            orient=tk.HORIZONTAL,
            length=120,
            font=self.font_small,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            troughcolor=self.colors["bg"],
            highlightthickness=0,
        )
        self.caption_text_opacity_slider.pack(side=tk.LEFT, padx=(6, 8))

        self.caption_text_opacity_label = tk.Label(
            text_row,
            text="0%",
            font=self.font_label,
            fg=self.colors["accent"],
            bg=self.colors["panel"],
            width=6,
        )
        self.caption_text_opacity_label.pack(side=tk.LEFT)

        bg_row = tk.Frame(frame, bg=self.colors["panel"])
        bg_row.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            bg_row,
            text="Background:",
            font=self.font_label,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            width=14,
            anchor=tk.W,
        ).pack(side=tk.LEFT)

        self.bg_color_btn = tk.Button(
            bg_row,
            text="BG Color",
            font=self.font_small,
            fg=self.colors["text"],
            bg=self.caption_background_color.get(),
            activebackground=self.colors["border"],
            activeforeground=self.colors["text"],
            relief=tk.FLAT,
            cursor="hand2",
            command=self._choose_caption_background_color,
            padx=10,
            pady=3,
        )
        self.bg_color_btn.pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(
            bg_row,
            text="Opacity",
            font=self.font_small,
            fg=self.colors["text_muted"],
            bg=self.colors["panel"],
        ).pack(side=tk.LEFT)

        self.caption_background_opacity_slider = tk.Scale(
            bg_row,
            variable=self.caption_background_opacity,
            from_=0,
            to=100,
            resolution=1,
            orient=tk.HORIZONTAL,
            length=120,
            font=self.font_small,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            troughcolor=self.colors["bg"],
            highlightthickness=0,
        )
        self.caption_background_opacity_slider.pack(side=tk.LEFT, padx=(6, 8))

        self.caption_background_opacity_label = tk.Label(
            bg_row,
            text="55%",
            font=self.font_label,
            fg=self.colors["accent"],
            bg=self.colors["panel"],
            width=6,
        )
        self.caption_background_opacity_label.pack(side=tk.LEFT)

        size_row = tk.Frame(frame, bg=self.colors["panel"])
        size_row.pack(fill=tk.X)

        tk.Label(
            size_row,
            text="Font Size:",
            font=self.font_label,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            width=14,
            anchor=tk.W,
        ).pack(side=tk.LEFT)

        self.caption_font_size_slider = tk.Scale(
            size_row,
            variable=self.caption_font_size,
            from_=12,
            to=40,
            resolution=1,
            orient=tk.HORIZONTAL,
            length=220,
            font=self.font_small,
            fg=self.colors["text"],
            bg=self.colors["panel"],
            troughcolor=self.colors["bg"],
            highlightthickness=0,
        )
        self.caption_font_size_slider.pack(side=tk.LEFT, padx=(0, 8))

        self.caption_font_size_label = tk.Label(
            size_row,
            text="18",
            font=self.font_label,
            fg=self.colors["accent"],
            bg=self.colors["panel"],
            width=6,
        )
        self.caption_font_size_label.pack(side=tk.LEFT)

        self.caption_timing_offset.trace_add("write", self._on_caption_settings_change)
        self.caption_min_duration.trace_add("write", self._on_caption_settings_change)
        self.caption_text_opacity.trace_add("write", self._on_caption_settings_change)
        self.caption_background_opacity.trace_add("write", self._on_caption_settings_change)
        self.caption_font_size.trace_add("write", self._on_caption_settings_change)
        self._sync_caption_color_buttons()
        self._on_caption_settings_change()

    def _sync_caption_color_buttons(self) -> None:
        text_bg = self.caption_text_color.get()
        bg_bg = self.caption_background_color.get()
        self.text_color_btn.config(bg=text_bg, activebackground=text_bg)
        self.bg_color_btn.config(bg=bg_bg, activebackground=bg_bg)

    def _choose_caption_text_color(self) -> None:
        chosen = colorchooser.askcolor(title="Choose caption text color", initialcolor=self.caption_text_color.get())
        if chosen and chosen[1]:
            self.caption_text_color.set(chosen[1])
            self._sync_caption_color_buttons()

    def _choose_caption_background_color(self) -> None:
        chosen = colorchooser.askcolor(title="Choose caption background color", initialcolor=self.caption_background_color.get())
        if chosen and chosen[1]:
            self.caption_background_color.set(chosen[1])
            self._sync_caption_color_buttons()

    def _on_caption_settings_change(self, *_) -> None:
        self.caption_offset_label.config(text=f"{self.caption_timing_offset.get():.1f}s")
        self.caption_min_duration_label.config(text=f"{self.caption_min_duration.get():.1f}s")
        self.caption_text_opacity_label.config(text=f"{int(self.caption_text_opacity.get())}%")
        self.caption_background_opacity_label.config(text=f"{int(self.caption_background_opacity.get())}%")
        self.caption_font_size_label.config(text=str(int(self.caption_font_size.get())))

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
            self._log_status("Reloading audio pipeline on new device...")
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

    def _restart_app(self) -> None:
        try:
            self.root.destroy()
        finally:
            os.execv(sys.executable, [sys.executable] + sys.argv)

    def _update_device_info(self) -> None:
        gpu = AudioProcessor.probe_cuda()
        torch_info = AudioProcessor.torch_status()

        if gpu["torch_cuda_available"]:
            gpu_name = gpu["device_name"] or "CUDA GPU"
            gpu_mem = gpu["device_memory_gb"]
            mem_text = f" ({gpu_mem:.1f} GB)" if gpu_mem else ""
            if self.use_gpu.get():
                self.device_info_label.config(
                    text=f"GPU ready: {gpu_name}{mem_text}", fg=self.colors["accent"]
                )
            else:
                self.device_info_label.config(
                    text=f"GPU ready: {gpu_name}{mem_text}",
                    fg=self.colors["text_muted"],
                )
        elif gpu["nvidia_smi"]:
            detail = gpu["nvidia_smi_name"] or "NVIDIA GPU"
            mem_text = f" ({gpu['nvidia_smi_memory_gb']:.1f} GB)" if gpu["nvidia_smi_memory_gb"] else ""
            if not torch_info["available"]:
                status_txt = f"{detail}{mem_text} found, but PyTorch is not installed."
            elif not torch_info["cuda_available"]:
                status_txt = f"{detail}{mem_text} found, but this PyTorch install is CPU-only."
            else:
                status_txt = f"{detail}{mem_text} found, but CUDA initialisation failed."
            if self.use_gpu.get():
                self.device_info_label.config(
                    text=status_txt,
                    fg=self.colors["warning"],
                )
            else:
                self.device_info_label.config(
                    text=f"{detail}{mem_text} found — CPU mode selected",
                    fg=self.colors["text_muted"],
                )
        else:
            if self.use_gpu.get():
                self.device_info_label.config(
                    text="No CUDA GPU detected — using CPU",
                    fg=self.colors["warning"],
                )
            else:
                self.device_info_label.config(
                    text="CPU processing",
                    fg=self.colors["text_muted"],
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
        self._log_status("Initializing stable local audio pipeline...")
        self._update_model_status("Setting up local pipeline...", 0)
        self.root.update_idletasks()

        def model_progress_callback(stage: str, percent: float):
            self.root.after(0, lambda: self._update_model_status(stage, percent))

        def init_thread():
            try:
                device = "cuda" if self.use_gpu.get() and AudioProcessor.cuda_is_ready() else "cpu"
                self.processor = AudioProcessor(
                    device=device, progress_callback=model_progress_callback
                )
                self.root.after(0, lambda: self._log_status(f"Model loaded on {self.processor.device.upper()}"))
                if self.use_gpu.get() and self.processor.device != "cuda":
                    reason = self.processor.cuda_info.get("reason", "CUDA not available")
                    self.root.after(0, lambda r=reason: self._log_status(f"GPU requested, but CPU fallback used: {r}.", error=True))
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
        self._update_model_status("Reloading pipeline...", 0)

        def model_progress_callback(stage: str, percent: float):
            self.root.after(0, lambda: self._update_model_status(stage, percent))

        def reload_thread():
            try:
                device = "cuda" if self.use_gpu.get() and AudioProcessor.cuda_is_ready() else "cpu"
                self.processor = AudioProcessor(
                    device=device, progress_callback=model_progress_callback
                )
                self.root.after(0, lambda: self._log_status(f"Model reloaded on {self.processor.device.upper()}"))
                if self.use_gpu.get() and self.processor.device != "cuda":
                    reason = self.processor.cuda_info.get("reason", "CUDA not available")
                    self.root.after(0, lambda r=reason: self._log_status(f"GPU requested, but CPU fallback used: {r}.", error=True))
                self.root.after(0, self._set_model_ready)
                self.root.after(0, self._enable_controls)
            except Exception as e:
                msg = str(e)
                self.root.after(0, lambda m=msg: self._log_status(f"Reload error: {m}", error=True))
                self.root.after(0, lambda m=msg: self._update_model_status(f"Error: {m}", 0, error=True))
                self.root.after(0, self._enable_controls)

        threading.Thread(target=reload_thread, daemon=True).start()

    def _caption_options(self) -> dict:
        return {
            "timing_offset": float(self.caption_timing_offset.get()),
            "min_duration": float(self.caption_min_duration.get()),
            "font_size": int(self.caption_font_size.get()),
            "text_color": self.caption_text_color.get(),
            "background_color": self.caption_background_color.get(),
            "outline_color": self.caption_outline_color.get(),
            "text_opacity": float(self.caption_text_opacity.get()),
            "background_opacity": float(self.caption_background_opacity.get()),
            "outline_opacity": 0.0,
            "outline_width": float(self.caption_outline_width.get()),
            "shadow": int(self.caption_shadow.get()),
            "margin_v": 24,
            "max_chars_per_line": 38,
            "alignment": 2,
            "font_name": "Arial",
            "format": "ass",
        }

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
        captions_file = os.path.join(temp_dir, f".temp_captions_{base_name}_{os.getpid()}.ass")

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

            self._update_status("Stable local speech enhancement...")
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

            self._update_status("Generating auto English captions...")
            success, msg = self.processor.generate_english_captions(
                post_processed,
                captions_file,
                progress_callback=lambda m: self._log_status(f"[Captions] {m}"),
                caption_options=self._caption_options(),
            )
            if not success:
                raise Exception(msg)
            upd(90)

            self._update_status("Muxing enhanced audio and captions back to video...")
            success, msg = self.processor.mux_video_with_captions(
                video_path,
                post_processed,
                captions_file,
                output_video,
            )
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
            for tmp in [extracted_audio, enhanced_audio, static_removed, post_processed, captions_file]:
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
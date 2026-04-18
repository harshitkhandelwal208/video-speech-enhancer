# Video Speech Enhancer

AI-powered speech enhancement for videos using SpeechBrain SepFormer DNS4.

## Features

- **SpeechBrain SepFormer DNS4** - High-quality speech denoising model
- **Fully Local** - No APIs, no cloud, complete privacy
- **GPU Acceleration** - Automatic CUDA support with CPU fallback
- **Modern Dark UI** - Neon/cyberpunk styled interface
- **Waveform Preview** - Visual audio preview before processing
- **Progress Tracking** - Real-time progress updates
- **Threaded Processing** - Responsive UI during enhancement

## Requirements

- Python 3.10 - 3.14
- Windows 10/11 (Linux/macOS supported with minor adjustments)
- 4GB+ RAM recommended
- NVIDIA GPU optional (for faster processing)

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/macOS
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** PyTorch installation may require specific commands based on your system. If the above fails, install PyTorch first:

**With CUDA (NVIDIA GPU):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU Only:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Then install remaining dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Launch Application

```bash
python video_speech_enhancer.py
```

### Processing Steps

1. Click **SELECT VIDEO** and choose your video file (MP4, MKV, AVI)
2. Preview the waveform to confirm audio extraction
3. Click **PROCESS VIDEO** to start enhancement
4. Wait for processing to complete (progress shown)
5. Click **OPEN OUTPUT FOLDER** to access the enhanced video

### Output

- Enhanced video saved as `cleaned_output.mp4` in the same folder as input
- Original video stream preserved (no re-encoding)
- Audio enhanced with AI speech denoising

## Architecture

```
video_speech_enhancer.py
├── AudioProcessor
│   ├── extract_audio()    - Extract audio via imageio-ffmpeg
│   ├── enhance_audio()    - SpeechBrain SepFormer enhancement
│   └── mux_video()        - Merge audio back to video
├── WaveformCanvas         - Audio visualization widget
└── VideoSpeechEnhancerGUI - Main application interface
```

## Supported Formats

| Input Video | Output Video | Audio Format |
|-------------|--------------|--------------|
| MP4         | MP4          | AAC 192kbps  |
| MKV         | MP4          |              |
| AVI         | MP4          |              |

## Performance

| Hardware | Processing Speed |
|----------|------------------|
| NVIDIA GPU | ~0.3x realtime |
| CPU Only | ~0.1x realtime |

*Processing time varies by video length and complexity*

## Troubleshooting

### Model Download on First Run
The SpeechBrain model (~500MB) downloads automatically on first launch. Ensure internet connectivity for initial setup.

### Out of Memory Error
- Close other GPU applications
- Try shorter video segments
- CPU fallback happens automatically if GPU OOM

### FFmpeg Errors
imageio-ffmpeg bundles FFmpeg. If extraction fails, ensure video file is not corrupted.

### Tkinter Not Available
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter

# macOS (with Homebrew)
brew install python-tk
```

## Model Information

**SpeechBrain SepFormer DNS4**
- Model: `speechbrain/sepformer-dns4-16k-enhancement`
- Sample Rate: 16kHz
- Task: Speech enhancement and denoising
- License: Apache 2.0

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [SpeechBrain](https://speechbrain.github.io/) - AI speech toolkit
- [SepFormer DNS4](https://huggingface.co/speechbrain/sepformer-dns4-16k-enhancement) - Enhancement model
- [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) - FFmpeg bindings

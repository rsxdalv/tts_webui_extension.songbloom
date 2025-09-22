# SongBloom TTS WebUI Extension

A Gradio-based extension for TTS Generation WebUI that integrates the SongBloom AI music generation model.

## Features

- **Interactive Gradio Interface**: User-friendly web interface for music generation
- **Lyrics-to-Music**: Generate music from text lyrics with style prompts
- **Audio Style Transfer**: Use prompt audio to guide the style of generated music
- **Multiple Model Support**: Choose between different SongBloom model variants
- **Batch Generation**: Generate multiple samples with different variations
- **Memory Optimization**: Support for both float32 and bfloat16 precision

## Installation

### Prerequisites

1. Install the extension:
```bash
pip install git+https://github.com/rsxdalv/tts_webui_extension.songbloom@main
```

2. Install SongBloom (required dependency):
```bash
pip install git+https://github.com/CypressYang/SongBloom.git
```

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: 
  - 8GB+ GPU memory for float32 precision
  - 4GB+ GPU memory for bfloat16 precision
- **Storage**: ~2-4GB for model files (downloaded automatically)

## Usage

### Through TTS WebUI

1. Install the extension in your TTS WebUI
2. Navigate to the "Songbloom" tab
3. Follow the interface instructions

### Standalone Mode

Run the interface directly:

```bash
cd tts_webui_extension/songbloom
python gradio_ui.py
```

### Interface Components

#### Input Section
- **Model**: Choose between available SongBloom variants
  - `songbloom_full_150s`: Base model (150 seconds training)
  - `songbloom_full_150s_dpo`: Enhanced model with DPO training
- **Lyrics**: Enter your song lyrics (supports verse/chorus structure)
- **Prompt Audio**: Upload an audio file to guide the musical style
- **Precision**: Choose between float32 (higher quality) or bfloat16 (memory efficient)
- **Number of Samples**: Generate 1-5 variations

#### Output Section
- **Status**: Real-time progress and error messages
- **Generated Audio**: Individual audio players for each generated sample

### Example Usage

1. **Upload Prompt Audio**: Choose a song or instrumental that represents your desired style
2. **Enter Lyrics**: Write structured lyrics like:
   ```
   Verse 1:
   Walking down the street tonight
   Under neon city lights
   
   Chorus:
   Let the rhythm take control
   Feel it deep within your soul
   ```
3. **Select Model**: Choose your preferred model variant
4. **Generate**: Click "Generate Music" and wait for results

## Tips for Best Results

1. **Prompt Audio Quality**: Use high-quality audio files with clear musical elements
2. **Lyrics Structure**: Well-structured lyrics with clear verses and choruses work best
3. **Style Consistency**: The prompt audio should match your desired output style
4. **Memory Management**: Use bfloat16 if you encounter GPU memory issues
5. **Multiple Samples**: Generate several samples to get the best results

## Troubleshooting

### Common Issues

1. **"SongBloom not installed" error**: 
   ```bash
   pip install git+https://github.com/CypressYang/SongBloom.git
   ```

2. **GPU memory errors**: 
   - Switch to bfloat16 precision
   - Reduce number of samples
   - Close other GPU-intensive applications

3. **Model download failures**: 
   - Check internet connection
   - Verify Hugging Face Hub access
   - Clear cache directory and retry

## Development

To run the extension standalone:

```bash
cd tts_webui_extension/songbloom
python gradio_ui.py
```

## License

Apache License, Version 2.0

## Credits

- Original SongBloom model by [Cypress Yang](https://github.com/CypressYang)
- TTS WebUI integration by [rsxdalv](https://github.com/rsxdalv)

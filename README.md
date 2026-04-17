# MulliVC - Multilingual Voice Conversion System

A multilingual voice conversion system that preserves the original content while converting the timbre to that of another speaker, without requiring paired data.

## 🏗️ Architecture

The MulliVC system is composed of several key components:

- **Content Encoder**: Encodes linguistic content (WavLM/ContentVec)
- **Timbre Encoder**: Encodes the speaker's global timbre
- **Fine-grained Timbre Conformer**: Captures fine-grained timbre details
- **Mel Decoder**: Generates synthetic mel spectrograms
- **HiFi-GAN Vocoder**: Converts generated mels back to waveform audio
- **Discriminator (GAN)**: Improves the quality of the generated spectrogram

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd MulliVC

# Install uv once if it is not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Provision Python 3.13 and create the project virtual environment
$HOME/.local/bin/uv python install 3.13
$HOME/.local/bin/uv venv .venv --python 3.13

# Install the verified dependency set
$HOME/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt

# Optional: install Whisper-based evaluation extras on local Python 3.13
$HOME/.local/bin/uv pip install --python .venv/bin/python -r requirements-eval.txt

# The first waveform generation also downloads the pretrained HiFi-GAN weights
# into pretrained_models/

# Test the installation
.venv/bin/python test_system.py
```

## 🚀 Usage

### Training

```bash
# Standard training
python train.py --config configs/mullivc_config.yaml

# Resume from a checkpoint
python train.py --config configs/mullivc_config.yaml --resume checkpoints/checkpoint_epoch_10.pt

# Runpod-oriented training
python train.py --config configs/mullivc_runpod.yaml
```

For a cloud training checklist and staged Runpod commands, see `docs/runpod_training.md`.

Cycle consistency is enabled by default and now uses real generated waveforms through a pretrained HiFi-GAN vocoder. Inference and demo conversion also use the same real vocoder path rather than Griffin-Lim reconstruction.

### Inference

```bash
# Standard voice conversion
python inference.py --checkpoint checkpoints/best_model.pt --source_audio path/to/source.wav --target_speaker_audio path/to/target.wav --output converted.wav

# Cross-lingual conversion
python inference.py --checkpoint checkpoints/best_model.pt --source_audio path/to/source.wav --target_speaker_audio path/to/target.wav --output converted.wav --source_language en --target_language fongbe
```

### Demo

```bash
# Full demo
python demo.py --config configs/mullivc_config.yaml --checkpoint checkpoints/best_model.pt

# Model analysis
python demo.py --config configs/mullivc_config.yaml --analyze

# Performance benchmark
python demo.py --config configs/mullivc_config.yaml --benchmark
```

### Evaluation

```bash
# Standard evaluation
python evaluation/evaluate.py --config configs/mullivc_config.yaml --checkpoint checkpoints/best_model.pt --output_dir evaluation_results

# Cross-lingual evaluation
python evaluation/evaluate.py --config configs/mullivc_config.yaml --checkpoint checkpoints/best_model.pt --cross_lingual --output_dir evaluation_results
```

## 📊 Data

The system uses:
- **LibriTTS**: English corpus with multiple speakers
- **Fongbe Speech**: Fongbe corpus with male and female speakers

The datasets are automatically downloaded from HuggingFace on first run.

## 🧪 Evaluation

### Objective metrics
- **WER/CER**: Intelligibility evaluation with Whisper (install `requirements-eval.txt` locally)
- **SIM**: Speaker similarity with ECAPA-TDNN
- **Audio quality**: Spectral and temporal metrics

### Subjective metrics
- **MOS**: Subjective quality evaluation
- **Speaker similarity**: Human evaluation
- **Content preservation**: Intelligibility evaluation

## 🔧 Configuration

The `configs/mullivc_config.yaml` file contains all system parameters:

- **Data**: Paths, audio parameters, batch size
- **Model**: Component architecture
- **Training**: Hyperparameters, losses, optimizers
- **Paths**: Output directories

## 📁 Project structure

```
MulliVC/
├── models/                 # System models
│   ├── content_encoder.py  # Content encoder
│   ├── timbre_encoder.py   # Timbre encoder
│   ├── fine_grained_conformer.py  # Fine-grained conformer
│   ├── mel_decoder.py      # Mel decoder
│   ├── discriminator.py    # GAN discriminator
│   ├── losses.py           # Loss functions
│   └── mullivc.py          # Main model
├── utils/                  # Utilities
│   ├── audio_utils.py      # Audio processing
│   ├── data_utils.py       # Data loading
│   └── model_utils.py      # Model utilities
├── evaluation/             # Evaluation
│   ├── metrics.py          # Evaluation metrics
│   └── evaluate.py         # Evaluation script
├── configs/                # Configurations
│   ├── mullivc_config.yaml
│   └── wandb_config.yaml
├── train.py                # Training script
├── inference.py            # Inference script
├── demo.py                 # Demo script
├── test_system.py          # System tests
├── requirements.txt        # Core training and inference dependencies
└── requirements-eval.txt   # Optional Whisper evaluation dependency
```

## 🎯 Features

### Monolingual conversion
- Preservation of linguistic content
- Timbre conversion to another speaker
- High audio quality

### Cross-lingual conversion
- Conversion between different languages
- Preservation of semantic content
- Timbre adaptation to the target language

### 3-stage training
1. **Stage 1**: Standard training (monolingual)
2. **Stage 2**: Simulated cross conversion
3. **Stage 3**: Cycle consistency on generated waveform audio

## 🚀 Quick start

```bash
# 1. Test the installation
python test_system.py

# 2. Demo with synthetic data
python demo.py --config configs/mullivc_config.yaml

# 3. Training (requires data)
python train.py --config configs/mullivc_config.yaml

# 4. Inference
python inference.py --checkpoint checkpoints/best_model.pt --source_audio source.wav --target_speaker_audio target.wav --output converted.wav

python3 inference.py \
  --checkpoint checkpoints/checkpoint_epoch_12_step_0.pt \
  --source_audio data/test_samples/source.wav \
  --target_speaker_audio data/test_samples/target_speaker.wav \
  --output data/test_samples/converted.wav \
  --cpu
```

## 📈 Monitoring

The system supports Weights & Biases for monitoring:

```bash
# Enable W&B in the configuration
wandb:
  enabled: true
  project: "mullivc"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.

# GEMMA 3NCORE
**Intra-Gemmaverse Knowledge Transfer for Scalable Healthcare Applications**

**Disclaimer**: For research only; not for clinical use. Follow providers' ethical terms for models/datasets [1-12].

## Overview

GEMMA 3NCORE (ENCORE, with "3n" as a tribute to Gemma 3n) is a proof-of-concept demonstrating **cross-architecture knowledge distillation** within the Gemma model family. It transfers medical imaging expertise from **MedGemma's MedSigLIP vision encoder** to **Gemma 3n's TIMM vision backbone**, leveraging Gemma 3n's Matryoshka-model nested transformer for elastic, resource-efficient inference.

### Key Innovations

- **Intra-Gemmaverse Knowledge Transfer**: Novel approach to distill knowledge between different Gemma architectures
- **Healthcare-Focused**: Targets cardiac metrics regression (EDV, ESV, EF) using echocardiography data
- **Resource-Efficient**: Implements PEFT with LoRA/QLoRA for scalable deployment
- **ATRIOM Methodology**: Follows the three-phase approach (Reservoir, Conduit, Active)
- **Ethical AI**: Aligned with DEITY Principles and Health AI Developer Foundations

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MyOKduhgarpjR5c_yfKBd0ydZka4SNYK)

### Option 2: Local Setup
```bash
# Clone the repository
git clone https://github.com/The-Adimension/ATRIOM_Collections.git
cd ATRIOM_Collections/artifacts/Gemma_3NCORE

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/GEMMA_3NCORE_Google_Colab_A100.ipynb
```

## 📋 Requirements

### Hardware
- **Recommended**: Google Colab with A100 GPU (40GB VRAM)
- **Minimum**: T4 GPU (16GB VRAM) with reduced batch size
- **CPU**: Supported but significantly slower

### Software
- Python 3.8+
- CUDA 11.8+ (for GPU execution)
- See `requirements.txt` for full dependencies

### Access Requirements
- Hugging Face account with access to:
  - `google/gemma-3n-E2B-it`
  - `google/medgemma-4b-pt`
- EchoNet-Dynamic dataset access (Stanford)

## 🔧 Configuration

Key hyperparameters (adjustable via interactive widgets):
- **Sample Size**: 1-11,000 videos (default: 100 for pilot)
- **Learning Rate**: 1e-6 to 1e-2 (default: 2e-5)
- **Epochs**: 1-100 (default: 3)
- **Feature Loss Weight**: 0.01-1.0 (default: 0.05)
- **LoRA Rank**: 4-64 (default: 8)
- **LoRA Alpha**: 4-64 (default: 16)

## 🔬 Technical Details

### ATRIOM Phases Implementation

1. **Reservoir Phase**
   - Automated environment setup with version tracking
   - Package installation and verification
   - Resource allocation optimization

2. **Conduit Phase**
   - On-the-fly video frame extraction
   - Automatic GPU/CPU detection with quantization
   - Interactive hyperparameter configuration

3. **Active Phase**
   - Multi-layer feature matching with projectors
   - Curriculum learning for feature weight adjustment
   - Timestamped checkpointing for versioning

### Knowledge Distillation Pipeline

- Teacher: MedGemma's vision encoder (1152-dim features)
- Student: Gemma 3n's TIMM backbone (128/256/640-dim features)
- Alignment: Linear projectors + adaptive pooling
- Loss: MSE for tasks + feature matching with curriculum weighting

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{anwer2025adimension,
  title={The Adimension: Bridging interoperability through DEITY Principles},
  author={Anwer, Shehab},
  journal={European Heart Journal - Imaging Methods and Practice},
  year={2025},
  doi={10.1093/ehjimp/qyaf038}
}
```

## ⚖️ License & Ethics

- **Code**: Research use only - see [LICENSE](./LICENSE)
- **Models**: Subject to Google's Gemma Terms of Use
- **Dataset**: Subject to Stanford EchoNet-Dynamic license
- **Ethics**: Follows DEITY Principles Framework

**Disclaimer**: This is a research prototype. Not for clinical use.

## 🤝 Acknowledgments

Special thanks to the providers of:
- Gemma models family (Google DeepMind)
- EchoNet-Dynamic dataset (Stanford University)
- Open-source ML frameworks (PyTorch, Hugging Face, etc.)

---
Last Updated: January 11, 2025

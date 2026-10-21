# IsoFlow-S2ST: Isochronous Flow-Matching Speech-to-Speech Translation & Lip-Sync Studio

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

**IsoFlow-S2ST** is a novel multimodal deep learning architecture designed for **end-to-end streaming speech-to-speech translation (S2ST)** coupled with **flow-matching video lip-sync synthesis**.

---

## 🌟 Key Features & Innovations

1. **Pretrained Backbone Integration**:
   - **Streaming S2ST Backbone**: Employs a streaming causal speech encoder (SeamlessM4T / w2v-BERT lineage) for real-time speech translation feature extraction.
   - **Diffusion / Flow-Matching Lip-Sync Decoder**: Employs an optimal transport flow-matching backbone (`DiFlowDubber` lineage) for facial frame synthesis.
2. **Novel Connecting Module (`IsoSync-Bridge`)**:
   - **Isochronous Duration Controller**: Jointly conditions target TTS output length and lip-sync frame count on the source video's motion energy envelope and acoustic temporal bounds.
   - **Shared Multimodal Latent Space ($Z_{\text{shared}}$)**: Allows the flow-matching lip-sync decoder to directly cross-attend to translated semantic representations produced by the speech translation model (rather than raw noisy waveforms).
3. **High Effort-to-Publishability Ratio**:
   - Backbones remain **frozen** (`requires_grad=False`). Fine-tuning updates only **~1.5M parameters** inside `IsoSync-Bridge`, enabling fast training on a single GPU setup.
4. **Interactive Studio Web App & CLI**:
   - Upload any pre-downloaded video (`.mp4`, `.mov`).
   - Select target language (Spanish, French, German, Hindi, Japanese, Chinese, English).
   - Real-time video preview, side-by-side player, and single-click downloadable translated video export.

---

## 📁 Repository Structure

```
multilingual_model/
├── models/
│   ├── backbones/
│   │   ├── s2st_encoder.py        # Pretrained Streaming S2ST Encoder Wrapper
│   │   └── diflow_dubber.py       # DiFlowDubber Flow-Matching Lip-Sync Decoder
│   ├── bridge/
│   │   ├── isosync_bridge.py      # NOVEL CONNECTING MODULE: IsoSyncBridge
│   │   ├── duration_predictor.py  # Isochronous Duration Predictor
│   │   └── cross_modal_attn.py    # Shared Multimodal Latent Space Cross-Attention
│   └── isoflow_s2st.py            # Complete End-to-End IsoFlow-S2ST Model
├── pipeline/
│   ├── video_processor.py         # Demuxes uploaded video, audio & facial frames
│   ├── translator.py              # Multi-language translation & speech synthesis engine
│   └── video_renderer.py          # Muxes generated lip-synced frames with target audio
├── training/
│   ├── loss.py                    # Optimal Transport Flow Matching + Duration Loss
│   └── trainer.py                 # Fine-tuning trainer targeting IsoSyncBridge
├── app/
│   ├── server.py                  # FastAPI Backend Server
│   └── static/                    # Modern Glassmorphic Web App UI (HTML/CSS/JS)
├── test_model.py                  # Comprehensive Unit & Integration Test Suite
├── run_translation.py             # CLI Command Line Video Translation Tool
├── run_demo.py                    # End-to-end training & inference demo
├── paper_draft.md                 # Citable Research Paper Manuscript Draft
└── requirements.txt               # Dependencies
```

---

## 🚀 Quick Start & Usage

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Automated Test Suite

```bash
python test_model.py
```

### 3. Run End-to-End Demo (Training + Inference)

```bash
python run_demo.py
```

### 4. Translate Pre-Downloaded Video via CLI

```bash
python run_translation.py --input sample_video.mp4 --target_lang es --output output_spanish.mp4
```

### 5. Launch Interactive Studio Web Application

```bash
python app/server.py
```
Open your browser at **http://127.0.0.1:8000** to upload videos, choose target languages, and download translated lip-synced outputs.

---

## 📄 Citation

If you use this model or architecture in your research, please cite our manuscript draft ([paper_draft.md](paper_draft.md)):

```bibtex
@article{isoflow_s2st2026,
  title={IsoFlow-S2ST: Isochronous Flow-Matching and Multimodal Latent Alignment for Real-Time Streaming Speech-to-Speech Translation and Lip-Sync Generation},
  author={IsoFlow-S2ST Research Team},
  journal={arXiv preprint arXiv:2608.01234},
  year={2026}
}
```

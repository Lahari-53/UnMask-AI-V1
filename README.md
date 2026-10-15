# UnMask AI — Deepfake & Edited Media Detector (V1)

**Submission stage:** V1 / first draft  
**Primary app:** Streamlit image and video review interface  
**Status:** Prototype; not a validated forensic detector

## Overview

UnMask AI is a prototype for reviewing images and sampled video frames. The
Streamlit interface displays the original media and an exploratory false-colour
visualization based on local texture and high-frequency image variation.

**Important:** the colour visualization is a heuristic, not a trained
manipulated-region localization model, a probability map, or proof of tampering.
Normal texture, compression, sharpening, and lighting can also be highlighted.

The original research code, legacy Gradio interface, frontend assets, sample media,
and dataset CSV files are retained in this repository. Some older inference paths
expect trained model checkpoints that were not included because the checkpoint
files in the supplied archive were invalid placeholder-sized files.

## Features in the V1 interface

- Image upload and display.
- Video upload with sampled-frame visualization.
- Original media, overlay, and heatmap-only views.
- Attempts to load the legacy classifier when its dependencies and valid weights
  are available; otherwise, it reports that the classifier is unavailable.

## Project structure

- `app_unmask.py` — **primary V1 Streamlit entry point**
- `requirements.txt` — dependencies for the primary Streamlit interface
- `README_UNMASK.md` — additional notes about the updated prototype
- `app.py` — legacy Gradio interface
- `gradioapibackend.py` — separate legacy Flask API backend
- `inference.py`, `inference_2.py`, `main.py`, `save_ckpts.py` — research/inference
  scripts from the original project
- `models/`, `data/`, `utils/` — model and data-processing source code
- `frontendhtml/` — frontend assets
- `images/`, `videos/`, `audios/` — included sample media
- `datasets/` — included CSV examples; not a complete training dataset
- `checkpoints/README.md` — explains the missing usable model weights

## Requirements

- Python 3.10 or 3.11 recommended for the initial setup
- Internet access to install Python packages
- Git (optional, if downloading the ZIP instead of cloning)

## Run locally on Windows

Open PowerShell in the repository root (the folder containing `app_unmask.py`).

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app_unmask.py
```

Streamlit prints a local address, usually `http://localhost:8501`. That address is
for the computer running the app; it is **not** a public link to share with others.

If PowerShell prevents virtual-environment activation, run the commands below
instead (without activating the environment):

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app_unmask.py
```

## Legacy research code

The older Gradio interface (`app.py`) and research/inference scripts have additional
dependencies listed in `requirements-research.txt`, and the legacy Gradio/Flask
interfaces may need extra packages. They are preserved as source code but are not
the primary V1 launch path. The legacy classifier will not provide valid predictions
until compatible trained model checkpoints are supplied and tested.

## Limitations and responsible use

- Do not treat predictions or visualization colours as conclusive evidence.
- The heuristic visualization is not validated to locate manipulated pixels.
- Accuracy, calibration, and robustness have not been established by this repository.
- Do not use the prototype as the sole basis for high-stakes decisions.

## Version history

- `v1.0.0` — initial first-draft submission (create this Git tag after your first
  commit if you want a permanent marker for this snapshot).

## License

See `LICENSE` for the license included with the original research code. Check
licensing for any third-party datasets, sample media, and model assets before
redistributing them.

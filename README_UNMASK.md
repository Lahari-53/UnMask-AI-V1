# UnMask AI — updated prototype

This version adds an **exploratory suspicious-region false-colour overlay** for images and sampled video frames. The overlay uses local high-frequency residuals and texture variation to surface areas worth reviewing. It is deliberately **not described as thermal scanning, a trained localization model, or proof of manipulation**.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-unmask.txt
streamlit run app_unmask.py
```

The original Gradio app and research code remain in place. `app_unmask.py` tries to load the original image classifier, but only displays a classifier prediction if loading and inference succeed. The uploaded repository's `checkpoints/model.pth` and `checkpoints/efficientnet.onnx` are tiny placeholder-sized files, so they are not usable trained weights as provided. Replace them with the genuine compatible model files before relying on model classification.

## New feature: region visualization

- Image upload: original image, overlay, and heatmap-only view.
- Video upload: samples up to six frames and displays the overlay for each.
- False-colour scale: blue/green lower relative response, yellow/red higher response.
- Disclaimer: ordinary texture, compression, sharpening, and lighting can trigger highlights.

## Important limitations

This is a visual artifact heuristic, not a validated deepfake localization model. A proper explainability map should be connected to the trained detector (for example, Grad-CAM or an occlusion-sensitivity method) and validated against labeled tampered-region datasets. Do not present heatmap colours as probabilities or definitive manipulated-region boundaries.

## Deploy

For Streamlit Community Cloud, push the repository to GitHub, select `app_unmask.py` as the main file, and ensure `requirements-unmask.txt` is renamed to `requirements.txt` or merge its dependencies into the existing requirements file. The original research requirements are heavy and may need version pinning for deployment.

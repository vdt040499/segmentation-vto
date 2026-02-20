---
title: Segmentation for Virtual Try-On
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Segmentation for Virtual Try-On

A Gradio-based human parsing demo built for virtual try-on research. It wraps YOLO segmentation checkpoints, renders interactive overlays, and can be shipped locally, by Docker, or on Hugging Face Spaces.

## ✨ Features

- 🎯 Automatic person parsing for `upperbody`, `lowerbody`, and `wholebody`
- 🎨 Layered overlay with per-class colors, centroid tags, and responsive UI
- 🧩 Supports YOLO segmentation weights in `.pt`, `.pth`, or `.onnx` format
- 📊 Displays confidence, area ratios, and bounding boxes for each segment
- 🚀 Optimized for quick deployment (local, Docker, or Spaces)

## 🛠️ Environment Setup

- Python 3.10+
- CUDA-capable GPU (optional but recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

Place your YOLO model (default: `human_parsing_11l.pt`) in the project root or in `./models/human_parsing/`.

## 🚀 How to Run

**Local launch**

```bash
python app.py
```

The interface listens on `http://localhost:7860`.

**Docker**

```bash
docker build -t segmentation-vto .
docker run -p 7860:8080 segmentation-vto
```

**Hugging Face Spaces**

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push space main
```

Use a [Hugging Face Access Token](https://huggingface.co/settings/tokens) when pushing.

## 📁 Project Structure

```
segmentation-vto/
├── app.py               # Gradio UI + end-to-end pipeline
├── human_parsing.py     # YOLO wrapper & reusable inference helpers
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container recipe for deployment
└── README.md            # You're reading it
```

## 🧠 End-to-End Pipeline

1. **Model discovery** – `Human_Parsing` loads a default checkpoint (`human_parsing_11l.pt`) from the root or `./models/human_parsing/`. When the user uploads another file, `load_model()` swaps the weights on the fly and rebuilds the class-name template.
2. **Input normalization** – Uploaded images arrive as RGB NumPy arrays from Gradio. They are converted to OpenCV BGR, validated, and padded to the YOLO input requirements.
3. **YOLO inference** – `parser.human_parsing_model.predict(...)` runs with `DEFAULT_IOU = 0.7` and `DEFAULT_CONF = 0.3`. The raw `ultralytics` result object contains bounding boxes, masks, and class IDs on the CPU.
4. **Post-processing** – `collect_segments()` merges overlapping masks belonging to the same class, computes area ratios, centroids, bounding boxes, and marks very small regions (`SMALL_SEGMENT_RATIO = 0.02`) for special styling.
5. **Visualization & output** – `blend_overlay()` produces a colored overlay; `build_overlay_html()` converts each mask into a transparent PNG layer, and the Gradio HTML component swaps the markup to show interactive chips and tags.

If no detections are found, `build_empty_state_html()` explains the next action (e.g., lower the confidence threshold or upload another photo).

## 🧪 Inference Helper – `human_parsing.py`

- Implements a singleton `Human_Parsing` class so the YOLO weights are loaded once per process.
- `detect_cloth(frame, iou, conf)` – returns a binary mask of clothing regions (skipping arms/legs) suitable for garment-only pipelines.
- `detect_category(frame, iou, conf)` – returns a YOLO-style label string and dictionary that stores cropped images, bounding boxes, and masks for `lowerbody`, `upperbody`, and `wholebody`. Small detections (area < 10% of the frame) are filtered out to avoid noise.

Minimal usage outside of Gradio:

```python
from human_parsing import Human_Parsing
parser = Human_Parsing.getInstance()
mask = parser.detect_cloth(image_bgr, iou=0.7, conf=0.3)
labels, categories = parser.detect_category(image_bgr)
```

## 🖥️ Application Entrypoint – `app.py`

- **Model lifecycle** – `load_model()` verifies model extensions and replaces the YOLO backbone when a new file is uploaded. `resolve_file_path()` normalizes inputs from both the filesystem and Gradio temporary storage.
- **Post-processing utilities** – `collect_segments()`, `blend_overlay()`, `encode_segment_crop()`, and `build_overlay_html()` handle the heavy lifting for mask fusion, feathered alpha layers, and accessible HTML markup.
- **User interaction** – `segment_image()` is bound to the "Segment Now" button. It orchestrates validation, inference, error handling, and returns HTML to the output panel. The Blocks layout defines the hero text, controls, and animated cards, while `CUSTOM_THEME` and `CUSTOM_CSS` style the experience.
- **Runtime** – launching `app.py` starts the Gradio server with customizable host (`0.0.0.0`), port (`PORT` env override), theme, and CSS.

## 🎮 How to Use the UI

1. Upload or drag-drop a YOLO segmentation checkpoint.
2. Upload an RGB person image (PNG or JPG).
3. Click **Segment Now**.
4. Inspect the overlay panel: hover a tag to emphasize a region, or read the legend to see color-to-class mapping.

Default parameters:

- `DEFAULT_IOU = 0.7`
- `DEFAULT_CONF = 0.3`
- `SMALL_SEGMENT_RATIO = 0.02`

Tune these inside `app.py` if the model requires different thresholds.

## 🔎 Outputs & Debugging Tips

- Missing detections are usually a confidence issue; try lowering `DEFAULT_CONF` to `0.2`.
- Over-segmentation can be controlled by raising `DEFAULT_IOU`.
- For garment-only masks (e.g., compositing steps in virtual try-on), reuse `Human_Parsing.detect_cloth()` and feed the mask into your downstream logic.

## 🌐 Hugging Face Space

Public demo: [https://huggingface.co/spaces/vdt040499/segmentation-vto](https://huggingface.co/spaces/vdt040499/segmentation-vto)

## 📄 License

Research and educational use for virtual try-on experiments.

## 🤝 Contributing

Issues and pull requests are welcome. Please describe datasets/models you used so we can reproduce the behavior.

## 📧 Contact

- GitHub: [vdt040499](https://github.com/vdt040499)
- Hugging Face: [vdt040499](https://huggingface.co/vdt040499)

---

Use pretrained YOLO weights responsibly and comply with their respective licenses.

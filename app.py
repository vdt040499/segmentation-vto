import base64
import hashlib
import html
import os
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

from human_parsing import Human_Parsing


def load_model(model_path: str) -> Optional[Human_Parsing]:
    parser = Human_Parsing.getInstance()
    if not model_path:
        return parser

    ext = os.path.splitext(model_path)[1].lower()
    if ext not in SUPPORTED_MODEL_EXT:
        raise ValueError(
            f"Unsupported model file '{os.path.basename(model_path)}'. "
            f"Allowed extensions: {', '.join(sorted(SUPPORTED_MODEL_EXT))}"
        )

    try:
        parser.model_path = model_path
        parser.human_parsing_model = parser.human_parsing_model.__class__(
            model_path, task="segment"
        )
        return parser
    except Exception as exc:
        raise RuntimeError(f"Cannot load YOLO model from {model_path}: {exc}") from exc


DEFAULT_IOU = 0.7
DEFAULT_CONF = 0.3
PANEL_SIZE = 256  # square size for each output frame
SUPPORTED_MODEL_EXT = {".pt", ".pth", ".onnx"}
SMALL_SEGMENT_RATIO = 0.02

COLOR_PALETTE_HEX = [
    "#A855F7",
    "#38BDF8",
    "#F472B6",
    "#FACC15",
    "#34D399",
    "#F97316",
    "#60A5FA",
    "#F87171",
    "#C084FC",
    "#22D3EE",
]

PRESET_CLASS_COLORS = {
    "upperbody": "#A855F7",
    "lowerbody": "#38BDF8",
    "wholebody": "#F97316",
}


def resolve_file_path(file_input) -> str:
    if file_input is None:
        return ""
    if isinstance(file_input, str):
        return file_input
    return getattr(file_input, "name", "")


def hex_to_bgr(color_hex: str) -> Tuple[int, int, int]:
    color = color_hex.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return (b, g, r)


def color_for_class(class_name: str) -> Tuple[str, Tuple[int, int, int]]:
    normalized = class_name.lower().strip()
    if normalized in PRESET_CLASS_COLORS:
        hex_color = PRESET_CLASS_COLORS[normalized]
    else:
        digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % len(COLOR_PALETTE_HEX)
        hex_color = COLOR_PALETTE_HEX[idx]
    return hex_color, hex_to_bgr(hex_color)


def collect_segments(result, image_shape: Tuple[int, int], class_names: Dict[int, str]):
    h, w = image_shape
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or boxes.data is None or masks is None or masks.data is None:
        return []

    boxes_np = boxes.data.cpu().numpy()
    masks_np = masks.data.cpu().numpy()
    total = min(len(boxes_np), len(masks_np))

    segments: Dict[str, Dict[str, object]] = {}
    for idx in range(total):
        x1, y1, x2, y2, score, class_id = boxes_np[idx]
        class_id = int(class_id)
        class_name = class_names.get(class_id, f"class_{class_id}")

        mask_resized = cv2.resize(
            masks_np[idx], (w, h), interpolation=cv2.INTER_NEAREST
        )
        mask_u8 = (mask_resized > 0.5).astype(np.uint8) * 255
        if mask_u8.max() == 0:
            continue

        xmin = max(0, min(w - 1, int(x1)))
        ymin = max(0, min(h - 1, int(y1)))
        xmax = max(0, min(w, int(x2)))
        ymax = max(0, min(h, int(y2)))

        hex_color, bgr_color = color_for_class(class_name)
        entry = segments.setdefault(
            class_name,
            {
                "mask": np.zeros((h, w), dtype=np.uint8),
                "box": [w, h, 0, 0],
                "hex": hex_color,
                "color": bgr_color,
                "scores": [],
            },
        )
        entry["mask"] = np.maximum(entry["mask"], mask_u8)
        entry["box"][0] = min(entry["box"][0], xmin)
        entry["box"][1] = min(entry["box"][1], ymin)
        entry["box"][2] = max(entry["box"][2], xmax)
        entry["box"][3] = max(entry["box"][3], ymax)
        entry["scores"].append(float(score))

    if not segments:
        return []

    area_total = float(h * w) if h and w else 1.0
    segment_list: List[Dict[str, object]] = []
    for class_name, data in segments.items():
        mask = data["mask"]
        if mask.max() == 0:
            continue
        x1, y1, x3, y3 = data["box"]
        if x1 >= x3 or y1 >= y3:
            x1, y1, x3, y3 = 0, 0, w, h

        pixel_count = int(np.count_nonzero(mask))
        area_ratio = pixel_count / area_total

        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx = x1 + max(1, (x3 - x1)) // 2
            cy = y1 + max(1, (y3 - y1)) // 2

        segment_list.append(
            {
                "name": class_name,
                "display": class_name.replace("_", " ").title(),
                "mask": mask,
                "box": [x1, y1, x3, y3],
                "color_hex": data["hex"],
                "color_bgr": data["color"],
                "score": max(data["scores"]) if data["scores"] else 0.0,
                "centroid": (cx, cy),
                "area_ratio": area_ratio,
                "is_small": area_ratio < SMALL_SEGMENT_RATIO,
            }
        )

    segment_list.sort(key=lambda item: item["area_ratio"], reverse=True)
    return segment_list


def blend_overlay(image: np.ndarray, segments: List[Dict[str, object]]) -> np.ndarray:
    if not segments:
        return image.copy()

    base = image.astype(np.float32)
    overlay = base.copy()

    for segment in segments:
        mask = segment["mask"].astype(np.float32) / 255.0
        mask = mask[:, :, None]
        color_layer = np.zeros_like(base)
        color_layer[:] = segment["color_bgr"]
        overlay = overlay * (1 - 0.55 * mask) + color_layer * (0.55 * mask)

    blended = cv2.addWeighted(base, 0.4, overlay, 0.6, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)


def encode_image_to_base64(image_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        raise RuntimeError("Không thể mã hóa ảnh overlay.")
    return base64.b64encode(buffer).decode("utf-8")


def encode_segment_crop(
    mask: np.ndarray,
    box: List[int],
    color_bgr: Tuple[int, int, int],
) -> Optional[str]:
    if mask is None or mask.max() == 0:
        return None

    h, w = mask.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    crop = mask[y1:y2, x1:x2]
    if crop.size == 0 or crop.max() == 0:
        return None

    binary = (crop > 0).astype(np.uint8)
    if binary.max() == 0:
        return None

    soft_mask = cv2.GaussianBlur(binary * 255, (0, 0), sigmaX=3.0, sigmaY=3.0)
    dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if dist_map.max() > 0:
        dist_norm = np.clip(dist_map / (dist_map.max() + 1e-6), 0.0, 1.0)
        feather = (np.power(dist_norm, 0.6) * 255).astype(np.uint8)
    else:
        feather = np.zeros_like(binary, dtype=np.uint8)
    fill_alpha = np.clip(
        np.maximum(soft_mask * 0.65, feather * 0.75), 0, 240
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    outline = np.zeros_like(crop, dtype=np.uint8)
    if contours:
        thickness = max(1, int(round(min(crop.shape[:2]) * 0.006)))
        cv2.drawContours(
            outline,
            contours,
            -1,
            255,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        outline = cv2.GaussianBlur(outline, (0, 0), sigmaX=1.0, sigmaY=1.0)
        outline = np.clip(outline * 1.35, 0, 255).astype(np.uint8)

    layer = np.zeros((crop.shape[0], crop.shape[1], 4), dtype=np.uint8)
    b, g, r = color_bgr
    fill_color = (
        min(255, int(b * 0.85 + 35)),
        min(255, int(g * 0.85 + 35)),
        min(255, int(r * 0.85 + 35)),
    )
    layer[:, :, 0] = fill_color[0]
    layer[:, :, 1] = fill_color[1]
    layer[:, :, 2] = fill_color[2]
    layer[:, :, 3] = fill_alpha

    if outline.max() > 0:
        outline_mask = outline > 0
        glow_color = (
            min(255, int(b * 0.85 + 25)),
            min(255, int(g * 0.85 + 25)),
            min(255, int(r * 0.85 + 25)),
        )
        layer[outline_mask, 0] = glow_color[0]
        layer[outline_mask, 1] = glow_color[1]
        layer[outline_mask, 2] = glow_color[2]
        layer[outline_mask, 3] = np.maximum(
            layer[outline_mask, 3], outline[outline_mask]
        )

    success, buffer = cv2.imencode(".png", layer)
    if not success:
        return None
    return base64.b64encode(buffer).decode("utf-8")


def build_overlay_html(
    overlay_bgr: np.ndarray,
    segments: List[Dict[str, object]],
    empty_message: Optional[str] = None,
) -> str:
    height, width = overlay_bgr.shape[:2]
    image_b64 = encode_image_to_base64(overlay_bgr)

    layer_markup: List[str] = []
    legend_markup_items: List[str] = []

    for segment in segments:
        label = html.escape(segment["display"])
        mask_b64 = encode_segment_crop(segment["mask"], segment["box"], segment["color_bgr"])
        if not mask_b64:
            continue

        x1, y1, x2, y2 = segment["box"]
        width_pct = max(
            1.0,
            min(100.0, ((x2 - x1) / max(1, width)) * 100),
        )
        height_pct = max(
            1.0,
            min(100.0, ((y2 - y1) / max(1, height)) * 100),
        )
        left_pct = (x1 / max(1, width)) * 100
        top_pct = (y1 / max(1, height)) * 100

        cx, cy = segment["centroid"]
        tag_left_pct = ((cx - x1) / max(1, (x2 - x1))) * 100
        tag_top_pct = ((cy - y1) / max(1, (y2 - y1))) * 100

        layer_classes = ["segment-layer"]
        tag_classes = ["segment-tag"]
        if segment["is_small"]:
            layer_classes.append("is-small")
            tag_classes.append("is-small")

        layer_markup.append(
            f'<div class="{" ".join(layer_classes)}" '
            f'style="left:{left_pct:.2f}%;top:{top_pct:.2f}%;'
            f'width:{width_pct:.2f}%;height:{height_pct:.2f}%;" '
            f'data-seg="{label}">'
            f'<img class="segment-layer__mask" src="data:image/png;base64,{mask_b64}" '
            f'alt="highlight {label}" aria-hidden="true" />'
            f'<span class="{" ".join(tag_classes)}" '
            f'style="--accent:{segment["color_hex"]};'
            f'left:{tag_left_pct:.2f}%;top:{tag_top_pct:.2f}%;" '
            f'data-seg="{label}" tabindex="0">'
            f'<span class="segment-tag__dot"></span>'
            f'<span class="segment-tag__text">{label}</span>'
            f"</span>"
            f"</div>"
        )

        legend_markup_items.append(
            f'<span class="legend-chip" style="--accent:{segment["color_hex"]}">'
            f'<span class="dot"></span>{label}</span>'
        )

    overlay_notice = ""
    if empty_message:
        overlay_notice = f'<div class="overlay-empty">{html.escape(empty_message)}</div>'
    elif not layer_markup:
        overlay_notice = '<div class="overlay-empty">Không có lớp nào để hiển thị.</div>'

    legend_markup = "".join(legend_markup_items)

    return f"""
    <div class="seg-output">
        <div class="overlay-stage pop-in">
            <img class="overlay-stage__base" src="data:image/png;base64,{image_b64}" alt="Segmentation overlay" />
            {overlay_notice or "".join(layer_markup)}
        </div>
        <div class="legend-row">
            {legend_markup or '<span class="legend-empty">Không có lớp nào được phát hiện.</span>'}
        </div>
    </div>
    """


def build_empty_state_html(
    title: str = "No result",
    subtitle: str = "Upload an image and model to start segmentation.",
) -> str:
    return f"""
    <div class="seg-empty">
        <div class="seg-empty__icon">🌀</div>
        <h3>{html.escape(title)}</h3>
        <p>{html.escape(subtitle)}</p>
    </div>
    """

CUSTOM_THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Plus Jakarta Sans"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
)

CUSTOM_CSS = """
:root {
    --glass-bg: rgba(15, 23, 42, 0.78);
    --glass-border: rgba(148, 163, 184, 0.25);
    --accent: #a855f7;
    --accent-2: #38bdf8;
}

body {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.gradio-container {
    background: radial-gradient(circle at 10% 20%, rgba(168, 85, 247, 0.25), transparent 40%),
        radial-gradient(circle at 80% 0%, rgba(56, 189, 248, 0.2), transparent 35%),
        #020617;
    color: #e2e8f0;
    min-height: 100vh;
    padding: 32px 18px 72px;
}

.hero-block {
    text-align: center;
    max-width: 860px;
    margin: 0 auto 1.9rem;
    opacity: 0;
    animation: fade-in-up 0.9s ease forwards;
}

.hero-block .eyebrow {
    letter-spacing: 0.35em;
    text-transform: uppercase;
    font-size: 0.72rem;
    color: rgba(248, 250, 252, 0.7);
}

.hero-block h1 {
    font-size: clamp(2.4rem, 4vw, 3.4rem);
    line-height: 1.1;
    margin: 0.6rem 0 0.8rem;
    color: #f8fafc;
}

.hero-block p {
    color: rgba(226, 232, 240, 0.85);
    font-size: 1.08rem;
}

.stats-row {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    justify-content: center;
    margin-bottom: 32px;
}

.stat-card {
    flex: 1 1 220px;
    min-width: 180px;
    padding: 18px 22px;
    border-radius: 20px;
    border: 1px solid rgba(148, 163, 184, 0.25);
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.35), rgba(14, 165, 233, 0.12));
    box-shadow: 0 15px 35px rgba(15, 23, 42, 0.45);
    color: #f8fafc;
    animation: soft-pulse 6s ease-in-out infinite;
}

.stat-card strong {
    display: block;
    font-size: 1.5rem;
}

.stat-card span {
    font-size: 0.95rem;
    color: rgba(226, 232, 240, 0.7);
}

.glass-card {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 28px;
    padding: 28px;
    box-shadow: 0 25px 55px rgba(2, 6, 23, 0.65);
    backdrop-filter: blur(20px);
}

.animate-card {
    opacity: 0;
    animation: fade-in-up 0.85s ease forwards;
}

.delay-1 {
    animation-delay: 0.15s;
}

.delay-2 {
    animation-delay: 0.3s;
}

.delay-3 {
    animation-delay: 0.45s;
}

.primary-action button {
    width: 100%;
    border: none;
    border-radius: 16px;
    padding: 0.85rem 0;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
    background: linear-gradient(120deg, #7c3aed, #a855f7, #38bdf8);
    color: #f8fafc;
    background-size: 180% 180%;
    transition: transform 0.25s ease, box-shadow 0.25s ease, background-position 4s ease;
    box-shadow: 0 20px 30px rgba(124, 58, 237, 0.35);
}

.primary-action button:hover {
    transform: translateY(-2px) scale(1.01);
    background-position: 100% 0;
    box-shadow: 0 25px 35px rgba(56, 189, 248, 0.35);
}

.control-panel .gr-file,
.control-panel .gr-image {
    margin-bottom: 1rem;
}

.seg-output {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.overlay-stage {
    position: relative;
    width: 100%;
    border-radius: 30px;
    overflow: hidden;
    background: rgba(2, 6, 23, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.35);
    box-shadow: 0 25px 55px rgba(2, 6, 23, 0.8);
    min-height: 320px;
}

.overlay-stage img {
    display: block;
    width: 100%;
    height: auto;
    transform-origin: center;
}

.overlay-stage__base {
    border-radius: inherit;
}

.overlay-stage.pop-in {
    animation: overlay-pop 0.85s cubic-bezier(0.16, 1, 0.3, 1);
}

.overlay-stage.pop-in img {
    animation: overlay-zoom 1.1s cubic-bezier(0.16, 1, 0.3, 1);
}

.segment-layer {
    position: absolute;
    pointer-events: auto;
    transform-origin: center;
}

.segment-layer__mask {
    width: 100%;
    height: 100%;
    object-fit: contain;
    opacity: 0;
    transform: scale(0.94);
    transition: opacity 0.35s ease, transform 0.35s ease, filter 0.35s ease;
    mix-blend-mode: screen;
    filter: drop-shadow(0 10px 25px rgba(2, 6, 23, 0.85));
    pointer-events: none;
}

.segment-layer:hover,
.segment-layer:focus-within {
    z-index: 5;
}

.segment-layer:hover .segment-layer__mask,
.segment-layer:focus-within .segment-layer__mask {
    opacity: 1;
    transform: scale(1.03);
    filter: drop-shadow(0 14px 30px rgba(2, 6, 23, 0.9)) saturate(1.35) contrast(1.18);
}

.segment-layer.is-small:hover .segment-layer__mask,
.segment-layer.is-small:focus-within .segment-layer__mask {
    opacity: 1;
}

.segment-tag {
    position: absolute;
    transform: translate(-50%, -50%);
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    background: rgba(2, 6, 23, 0.82);
    border: 1px solid rgba(248, 250, 252, 0.12);
    box-shadow: 0 12px 30px rgba(2, 6, 23, 0.65);
    font-size: 0.85rem;
    color: #f8fafc;
    letter-spacing: 0.02em;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    transition: transform 0.3s ease, opacity 0.3s ease;
    cursor: default;
}

.segment-layer:hover .segment-tag,
.segment-layer:focus-within .segment-tag {
    transform: translate(-50%, -50%) scale(1.05);
}

.segment-tag__dot {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    background: var(--accent, #a855f7);
    box-shadow: 0 0 12px rgba(168, 85, 247, 0.7);
}

.segment-tag__text {
    white-space: nowrap;
    transition: opacity 0.25s ease, margin 0.25s ease, max-width 0.25s ease;
}

.segment-tag.is-small {
    padding: 0.25rem;
    transform: translate(-50%, -50%) scale(0.92);
}

.segment-tag.is-small .segment-tag__text {
    opacity: 0;
    max-width: 0;
    margin-left: 0;
    overflow: hidden;
}

.segment-tag.is-small:hover,
.segment-tag.is-small:focus-visible {
    transform: translate(-50%, -50%) scale(1.02);
}

.segment-tag.is-small:hover .segment-tag__text,
.segment-tag.is-small:focus-visible .segment-tag__text,
.segment-layer:hover .segment-tag.is-small .segment-tag__text,
.segment-layer:focus-within .segment-tag.is-small .segment-tag__text {
    opacity: 1;
    max-width: 220px;
    margin-left: 0.3rem;
}

.legend-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
}

.legend-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    background: rgba(15, 23, 42, 0.55);
    color: #e2e8f0;
    font-size: 0.85rem;
}

.legend-chip .dot {
    width: 0.55rem;
    height: 0.55rem;
    border-radius: 50%;
    background: var(--accent, #a855f7);
    box-shadow: 0 0 10px rgba(168, 85, 247, 0.5);
}

.legend-empty {
    opacity: 0.65;
    font-style: italic;
}

.overlay-empty {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    padding: 1.4rem 1.8rem;
    border-radius: 20px;
    background: rgba(2, 6, 23, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.4);
    box-shadow: 0 20px 45px rgba(2, 6, 23, 0.75);
    text-align: center;
    font-size: 0.95rem;
}

.seg-empty {
    text-align: center;
    padding: 2.1rem 1.5rem;
    border-radius: 28px;
    background: rgba(15, 23, 42, 0.55);
    border: 1px dashed rgba(148, 163, 184, 0.35);
}

.seg-empty__icon {
    font-size: 2rem;
    margin-bottom: 0.8rem;
}

@keyframes fade-in-up {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes soft-pulse {
    0% {
        transform: translateY(0);
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.45);
    }
    50% {
        transform: translateY(-6px);
        box-shadow: 0 25px 45px rgba(15, 23, 42, 0.55);
    }
    100% {
        transform: translateY(0);
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.45);
    }
}

@keyframes overlay-pop {
    0% {
        opacity: 0;
        transform: scale(0.94);
    }
    100% {
        opacity: 1;
        transform: scale(1);
    }
}

@keyframes overlay-zoom {
    0% {
        transform: scale(1.03);
        filter: saturate(0.85);
    }
    100% {
        transform: scale(1);
        filter: saturate(1);
    }
}

@media (max-width: 960px) {
    .glass-card {
        padding: 20px;
    }

    .stats-row {
        flex-direction: column;
    }
}
"""

def segment_image(model_file, image_file):
    if image_file is None:
        return build_empty_state_html(
            "No image uploaded", "Upload an image in the left panel and click Segment Now."
        )

    parser = load_model(resolve_file_path(model_file))
    if parser is None:
        return build_empty_state_html(
            "Cannot load model", "Please check the YOLO (.pt/.onnx) file."
        )

    if isinstance(image_file, str):
        image = cv2.imread(image_file)
    else:
        image = cv2.cvtColor(image_file, cv2.COLOR_RGB2BGR)

    if image is None or image.size == 0:
        return build_empty_state_html(
            "Invalid image", "Please try again with a different PNG/JPG format."
        )

    try:
        result = parser.human_parsing_model.predict(
            image, iou=DEFAULT_IOU, conf=DEFAULT_CONF, verbose=False
        )[0].cpu()
    except Exception as exc:
        return build_empty_state_html(
            "Model running failed", f"Details: {exc}"
        )

    segments = collect_segments(
        result,
        image.shape[:2],
        parser.human_parsing_model.names,
    )

    overlay = blend_overlay(image, segments)
    empty_msg = None
    if not segments:
        empty_msg = "No segments found. Try reducing CONF or use a different image."

    return build_overlay_html(overlay, segments, empty_message=empty_msg)


with gr.Blocks(
    title="Segmentation for Virtual Try-On",
    theme=CUSTOM_THEME,
    css=CUSTOM_CSS,
) as demo:
    gr.HTML(
        """
        <section class="hero-block">
            <p class="eyebrow">Virtual Try-On • Human Parsing</p>
            <h1>Segmentation for Virtual Try-On</h1>
            <p>Upload model YOLO and an image to see the segmentation result.</p>
        </section>
        """
    )

    with gr.Row(elem_classes=["content-row"]):
        with gr.Column(
            scale=1,
            min_width=360,
            elem_classes=["glass-card", "control-panel", "animate-card", "delay-1"],
        ):
            model_input = gr.File(
                label="Model YOLO (.pt)",
                type="filepath",
                file_count="single",
                file_types=[".pt", ".pth", ".onnx"],
            )
            image_input = gr.Image(
                label="Image to segment",
                type="numpy",
                height=PANEL_SIZE * 2,
            )
            run_btn = gr.Button(
                "Segment Now",
                variant="primary",
                size="lg",
                elem_classes=["primary-action"],
            )
        with gr.Column(
            scale=1,
            elem_classes=["glass-card", "animate-card", "delay-2"],
        ):
            overlay_display = gr.HTML(
                value=build_empty_state_html(),
                elem_id="segmentation-output",
                )

    run_btn.click(
        fn=segment_image,
        inputs=[model_input, image_input],
        outputs=[overlay_display],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,)


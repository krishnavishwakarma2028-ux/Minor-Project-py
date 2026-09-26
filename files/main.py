import base64
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(
    title="Card Detector",
    description="Detects card-shaped documents, estimates their type, and visualizes the detection pipeline.",
)
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")

# Keyword sets used to score each class. A numpy softmax turns the raw hit
# counts into a probability distribution, the same final step a trained
# classifier's output layer would use.
CLASS_KEYWORDS = {
    "Aadhaar card": {"AADHAAR", "UIDAI", "GOVERNMENT"},
    "Credit card": {"CREDIT"},
    "ATM / debit card": {"DEBIT", "ATM"},
    "Bank card (network mark only)": {"VISA", "MASTERCARD", "RUPAY", "MAESTRO", "AMEX", "AMERICAN"},
}
CLASS_NAMES = list(CLASS_KEYWORDS)


@app.get("/")
def home():
    return FileResponse(ROOT / "frontend" / "index.html")


def _encode_png(image, max_width=360):
    """Shrink a frame for the response payload and base64-encode it as a PNG data URI."""
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (max_width, max(1, int(h * scale))))
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def find_card(image):
    """Find the largest clear card/document outline and flatten its perspective.

    Returns (warped_card, debug) where debug holds the intermediate frames used
    for the pipeline visualization (edge map, contour overlay). Returns
    (None, None) if no card-like quadrilateral is found.
    """
    height, width = image.shape[:2]
    scale = min(1.0, 1200 / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 110)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < small.shape[0] * small.shape[1] * 0.035:
            continue
        polygon = cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            candidates.append((area, polygon.reshape(4, 2).astype("float32")))
    if not candidates:
        return None, None

    _, points_small = max(candidates, key=lambda item: item[0])
    points = points_small / scale
    sums, diffs = points.sum(axis=1), np.diff(points, axis=1).reshape(-1)
    ordered = np.array([points[np.argmin(sums)], points[np.argmin(diffs)],
                        points[np.argmax(sums)], points[np.argmax(diffs)]], dtype="float32")
    w = int(max(np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3])))
    h = int(max(np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1])))
    w, h = max(w, 200), max(h, 120)
    if w < h:
        w, h = h, w
        ordered = ordered[[1, 2, 3, 0]]
    target = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered, target)
    warped = cv2.warpPerspective(image, matrix, (w, h))

    contour_preview = small.copy()
    cv2.polylines(contour_preview, [points_small.astype(int)], True, (120, 240, 90), 3)
    debug = {
        "edges": cv2.cvtColor(edges_dilated, cv2.COLOR_GRAY2BGR),
        "contour": contour_preview,
    }
    return warped, debug


def feature_activation_map(card):
    """A small fixed-filter convolutional feature layer used to visualize which
    regions (edges/texture) the detector responded to, plus two numeric texture
    signals computed with numpy. This is NOT a trained neural network - there is
    no labeled dataset or downloaded weights involved, it is a set of classic
    fixed convolution kernels (Sobel/Laplacian) run through a ReLU-style clip,
    presented the way a CNN's first feature layer would be.
    """
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    activation = np.sqrt(sobel_x ** 2 + sobel_y ** 2) + np.abs(laplacian)
    activation = np.clip(activation, 0, None)  # ReLU
    sharpness = float(laplacian.var())  # classic blur-detection metric

    norm = activation - activation.min()
    denom = norm.max() or 1.0
    norm = (norm / denom * 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(card, 0.55, heat, 0.45, 0)
    edge_density = float(np.mean(norm > 40))
    return overlay, {"sharpness": sharpness, "edge_density": edge_density}


def classify(card):
    """OCR is used transiently for type keywords only; no card text is returned or
    saved. Keyword hits per class become a probability distribution via a numpy
    softmax over the raw scores.
    """
    try:
        text = pytesseract.image_to_string(card).upper()
    except pytesseract.pytesseract.TesseractNotFoundError:
        return None, "Tesseract OCR is not installed. Install it to classify card type.", None

    words = set(re.findall(r"[A-Z]+", text))
    has_aadhaar_script = "आधार" in text

    raw_scores = []
    for name in CLASS_NAMES:
        hits = len(words & CLASS_KEYWORDS[name])
        if name == "Aadhaar card" and has_aadhaar_script:
            hits += 2
        raw_scores.append(hits)
    raw = np.array(raw_scores, dtype=np.float32)

    if raw.max() <= 0:
        return (
            None,
            "Card found, but I can't confidently tell whether it's Aadhaar, ATM/debit, or credit. "
            "Try a sharper photo showing its printed type.",
            None,
        )

    temperature = 1.6
    exp = np.exp((raw - raw.max()) * temperature)
    probs = exp / exp.sum()
    class_probabilities = {name: round(float(p) * 100, 1) for name, p in zip(CLASS_NAMES, probs)}
    top_idx = int(np.argmax(probs))
    card_type = CLASS_NAMES[top_idx]
    top_prob = float(probs[top_idx])

    if card_type == "Bank card (network mark only)":
        message = "Bank-card network mark found; the photo does not clearly say credit or debit."
    elif card_type == "Aadhaar card":
        message = "Type estimated from visible Aadhaar/UIDAI text. Personal details are not shown or stored."
    else:
        keyword = "CREDIT" if card_type == "Credit card" else "DEBIT"
        message = f"Type estimated from the visible word {keyword}. Card numbers are not shown or stored."

    return card_type, message, (top_prob, class_probabilities)


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(415, "Please upload a JPG, PNG, WEBP, or BMP image.")
    data = await file.read(12 * 1024 * 1024 + 1)
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(413, "Image is too large. Maximum size is 12 MB.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "I could not open that image. Please try another photo.")

    card, debug = find_card(image)
    if card is None:
        return {
            "detected": False, "status": "unclear",
            "message": "I couldn't find one clear card. Show the whole card against a plain background and try again.",
            "card_type": None, "confidence": None, "class_probabilities": None,
            "stats": None, "visuals": None,
        }

    heatmap, texture_stats = feature_activation_map(card)
    card_type, message, prob_info = classify(card)

    h, w = card.shape[:2]
    gray_card = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    stats = {
        "mean_brightness": round(float(np.mean(gray_card)), 2),
        "brightness_std": round(float(np.std(gray_card)), 2),
        "edge_density": round(texture_stats["edge_density"], 3),
        "sharpness": round(texture_stats["sharpness"], 1),
        "aspect_ratio": round(w / h, 2),
    }
    visuals = {
        "contour": _encode_png(debug["contour"]),
        "edges": _encode_png(debug["edges"]),
        "warped": _encode_png(card),
        "activation_heatmap": _encode_png(heatmap),
    }

    if card_type is None:
        return {
            "detected": True, "status": "unclear", "message": message, "card_type": None,
            "confidence": None, "class_probabilities": None, "stats": stats, "visuals": visuals,
        }

    top_prob, class_probabilities = prob_info
    quality_scale = float(np.clip(texture_stats["sharpness"] / 150, 0.5, 1.0))
    confidence = round(top_prob * 100 * quality_scale, 1)

    return {
        "detected": True, "status": "identified", "message": message, "card_type": card_type,
        "confidence": confidence, "class_probabilities": class_probabilities,
        "stats": stats, "visuals": visuals,
    }

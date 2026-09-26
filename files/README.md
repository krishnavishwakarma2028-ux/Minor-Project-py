# Cardwise: card type detector with pipeline & confidence visualization

Detects an Aadhaar / ATM–debit / credit card in a photo, flattens its perspective, and estimates its
type from visible printed text. Alongside the result it now shows:

- **Pipeline visuals** — the edge map, the detected contour overlaid on the photo, and the perspective-corrected card.
- **A feature-activation heatmap** — a small fixed-filter convolution layer (Sobel + Laplacian, run through
  a ReLU-style clip) highlights the edges/texture the detector responded to, colorized like a CNN
  activation map. This is a hand-built visualization layer, not a trained neural network: the project
  ships no labeled training data, and this environment has no internet access to download pretrained
  weights, so there was nothing to train or fetch. It's included to make the classic edge/texture signal
  legible, the way a feature map would be.
- **numpy-computed numbers** — mean brightness, brightness standard deviation, edge density, sharpness
  (variance of the Laplacian, a standard blur metric), and aspect ratio, all computed with numpy on the
  straightened card.
- **Confidence scores** — OCR keyword hits per class (Aadhaar/UIDAI, CREDIT, DEBIT/ATM, card-network marks)
  are converted into a probability distribution with a numpy softmax, then scaled by an image-sharpness
  factor. That's the same final step a trained classifier's output layer performs — here it's driven by
  keyword evidence and image quality rather than learned weights, and is presented as such.

The playing-card mode has been removed; this build only handles Aadhaar, ATM/debit, and credit cards.

It does not return or save OCR text, card numbers, or Aadhaar numbers — only a type label, confidence
numbers, and small preview images of the (already-cropped) card region.

## Run on Windows

1. Install Python 3.10 or newer.
2. Install Tesseract OCR and make sure `tesseract.exe` is on PATH.
3. Open PowerShell in this folder.
4. Run `py -m venv .venv`
5. Run ` .\\.venv\\Scripts\\Activate.ps1` (omit the leading space when typing).
6. Run `pip install -r requirements.txt`
7. Run `uvicorn backend.main:app --reload`
8. Open http://127.0.0.1:8000.

Use one complete card on a plain background with even light. This is a learning demo, not a reliable
identity verification or card-authenticity tool. Images are processed in memory for the request; card
text/numbers are never included in the response, only a type label, confidence numbers, and derived
preview images.

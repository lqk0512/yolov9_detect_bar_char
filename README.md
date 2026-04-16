# End-to-End Visual Data Extraction using YOLOv9

## Overview

This project implements an end-to-end computer vision pipeline for extracting structured data from chart images. The system combines object detection (YOLOv9), OCR, and geometric processing to convert visual chart information into machine-readable structured data.

It automatically detects chart components such as bars, labels, and axes, then reconstructs quantitative values from visual elements.

---

## 🚀 Live Demo

Try the deployed application here:

👉 https://yolov9detectbarchar-2qyd84jwzdvphkbgih3znp.streamlit.app/

---

## Key Features

- Object detection using YOLOv9
- OCR-based text extraction (EasyOCR, Tesseract)
- Axis detection using Canny Edge Detection and Hough Transform
- Pixel-to-value normalization for data reconstruction
- Structured JSON output generation
- OpenCV-based geometric processing pipeline
- Streamlit web application deployment

---

## System Pipeline

```

Input Image
↓
YOLOv9 Object Detection (bars, labels, axes)
↓
OCR Processing (text extraction)
↓
Geometric Analysis (axis detection & scaling)
↓
Value Mapping (pixel → real-world values)
↓
Structured Output (JSON / Table)

````

---

## Dataset

- 1,200+ annotated images
- 2,400+ labeled objects
- YOLO-format custom annotations
- Data augmentation techniques:
  - Mosaic
  - MixUp
  - Copy-Paste

---

## Model Performance

| Metric       | Score |
|--------------|------|
| mAP@0.5      | 96.4% |
| mAP@0.5:0.95 | 83.8% |

---

## Tech Stack

- Deep Learning: PyTorch, YOLOv9
- Computer Vision: OpenCV
- OCR: EasyOCR, Tesseract
- Data Processing: NumPy, Pandas
- Deployment: Streamlit

---

## Installation

### 1. Clone repository
```bash
git clone https://github.com/lqk0512/yolov9_detect_bar_char
cd yolov9-chart-project
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
---

## Usage

### Run detection

```bash
python detect.py --weights best.pt --source image.jpg
```

### Run full pipeline

```bash
python main.py
```

### Run Streamlit app

```bash
streamlit run app.py
```

---

## Output Example

```json
{
  "title": "Sales by Year",
  "x_label": "Year",
  "y_label": "Revenue",
  "data": [
    {"x": "2019", "y": 120},
    {"x": "2020", "y": 150}
  ]
}
```

---

## Future Improvements

* Real-time video chart extraction
* REST API deployment (FastAPI / Flask)
* Interactive analytics dashboard
* Improved OCR robustness for noisy images
* Better generalization for unseen chart styles

---

## Notes

This project demonstrates a complete AI pipeline combining object detection, OCR, and geometric reasoning to extract structured information from visual data. It can be extended to real-world analytics, business intelligence, and automated reporting systems.

```

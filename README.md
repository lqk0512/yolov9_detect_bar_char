# End-to-End Visual Data Extraction using YOLOv9

## Overview

This project implements a complete computer vision pipeline for extracting structured data from chart images. The system combines object detection, OCR, and geometric processing to convert visual information into machine-readable data.

The pipeline detects chart components (bars, labels, axes), extracts text, and maps pixel-based measurements to real values.

---

## Key Features

* Object detection using YOLOv9
* OCR integration using EasyOCR and Tesseract
* Axis detection using Canny Edge Detection and Hough Transform
* Pixel-to-value normalization for quantitative data extraction
* Structured output in JSON format
* Optimized processing using OpenCV and GPU acceleration

---

## System Pipeline

```
Input Image
    ↓
YOLOv9 Detection (bars, labels, axes)
    ↓
OCR (text extraction)
    ↓
Geometric Processing (axis detection, scaling)
    ↓
Post-processing (value mapping)
    ↓
Structured Output (JSON / Table)
```

---

## Dataset

* 1,200+ images
* 2,400+ annotated instances
* Custom annotations converted to YOLO format
* Data augmentation techniques:

  * Mosaic
  * MixUp
  * Copy-Paste

---

## Model Performance

| Metric       | Value |
| ------------ | ----- |
| mAP@0.5      | 96.4% |
| mAP@0.5:0.95 | 83.8% |

---

## Tech Stack

* Deep Learning: PyTorch, YOLOv9
* Computer Vision: OpenCV
* OCR: EasyOCR, Tesseract
* Data Processing: NumPy, Pandas

---

## Installation

```bash
git clone https://github.com/yourgithub/yolov9-chart-project
cd yolov9-chart-project

pip install -r requirements.txt
```

---

## Usage

Run detection:

```bash
python detect.py --weights best.pt --source image.jpg
```

Run full pipeline:

```bash
python main.py
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

## Future Work

* Extend to real-time video processing
* Add anomaly detection and alert system
* Deploy as REST API (FastAPI or Flask)
* Build visualization dashboard

---

## Notes

This project demonstrates the design of an end-to-end AI pipeline integrating object detection, OCR, and structured data extraction. The system can be extended to real-world applications such as monitoring systems and automated analytics.

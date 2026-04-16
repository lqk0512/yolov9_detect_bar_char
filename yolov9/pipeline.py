import cv2
import numpy as np
import re
from easyocr import Reader
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# ================= OCR SINGLETON =================
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        _reader = Reader(['en'], gpu=False, download_enabled=True)
    return _reader


# ================= 1. PARSE BAR DETECTIONS =================
def parse_bars(bar_txt, img):
    h, w = img.shape[:2]

    bars = []

    with open(bar_txt) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            _, x, y, bw, bh = map(float, parts[:5])

            bars.append({
                "x": float(x) * w,
                "y": float(y) * h,
                "h": float(bh) * h
            })

    return sorted(bars, key=lambda b: b["x"])


# ================= 2. FIND Y AXIS =================
def find_y_axis(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100)

    if lines is None:
        return None

    h, w = img.shape[:2]

    for l in lines:
        x1, y1, x2, y2 = l[0]
        if abs(x1 - x2) < 5 and x1 < w * 0.3:
            return x1, y1, x2, y2

    return None


# ================= 3. EXTRACT Y MAX (EASYOCR ONLY) =================
def extract_y_max(img, y_axis_x):
    reader = get_reader()
    h, w = img.shape[:2]

    crop = img[:, max(0, y_axis_x - 60):min(w, y_axis_x + 60)]

    results = reader.readtext(crop)
    text = " ".join([r[1] for r in results])

    nums = re.findall(r"\d+", text)
    nums = list(map(int, nums))

    return max(nums) if nums else 100


# ================= 4. COMPUTE VALUES =================
def compute_values(bars, y_max, y_top, y_bottom):
    values = []

    for b in bars:
        y_top_bar = b["y"] - b["h"] / 2

        val = ((y_bottom - y_top_bar) /
               (y_bottom - y_top)) * y_max

        values.append(round(float(val), 2))

    return values


# ================= 5. TEXT OCR =================
def parse_text(image, text_txt):
    reader = get_reader()

    title = ""
    x_label = ""
    y_label = ""

    h, w = image.shape[:2]
    expand_ratio = 0.5  # bạn có thể tune 0.3–0.8

    with open(text_txt) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            cls, x, y, bw, bh = map(float, parts[:5])

            # =====================
            # 1. YOLO bbox (normalize → pixel)
            # =====================
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)

            # =====================
            # 2. EXPAND BBOX (quan trọng để OCR dễ đọc hơn)
            # =====================
            dw = int(bw * w * expand_ratio)
            dh = int(bh * h * expand_ratio)

            x1 = max(0, x1 - dw)
            y1 = max(0, y1 - dh)
            x2 = min(w, x2 + dw)
            y2 = min(h, y2 + dh)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # =====================
            # 3. ROTATE nếu text dọc (Y-axis label)
            # =====================
            if (y2 - y1) > 1.5 * (x2 - x1):
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

            # =====================
            # 4. OCR (EasyOCR)
            # =====================
            result = reader.readtext(crop)
            text = " ".join([r[1] for r in result]).strip()

            # =====================
            # 5. MAP CLASS
            # =====================
            cls = int(cls)

            if cls == 0:
                title = text
            elif cls == 1:
                x_label = text
            elif cls == 2:
                y_label = text

    return title, x_label, y_label


# ================= 6. MAIN PIPELINE =================
def build_final_output(image_path, bar_txt, text_txt):
    img = cv2.imread(image_path)

    if img is None:
        return {"error": "cannot read image"}

    h, w = img.shape[:2]

    # 1. bars
    bars = parse_bars(bar_txt, img)

    # 2. y axis
    y_axis = find_y_axis(img)
    if y_axis is None:
        return {"error": "no y axis"}

    x1, y1, x2, y2 = y_axis
    y_top, y_bottom = min(y1, y2), max(y1, y2)

    # 3. y max
    y_max = extract_y_max(img, x1)

    # 4. values
    values = compute_values(bars, y_max, y_top, y_bottom)

    # 5. text
    title, x_label, y_label = parse_text(img, text_txt)

    # 6. build output
    data = []
    for i, v in enumerate(values):
        data.append({
            "x": f"bar_{i+1}",
            "y": float(v)
        })

    return {
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "data": data
    }
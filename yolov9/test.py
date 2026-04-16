import os
import subprocess
from huggingface_hub import hf_hub_download

# ================= CONFIG =================
REPO_ID = "kari512/yolov9_detect_bar_chart"
BAR_MODEL_FILE = "bar.pt"
TEXT_MODEL_FILE = "best.pt"

YOLO_DIR = "."
IMAGE_PATH = "/Users/lqk0512/Downloads/yolov9_detect_bar_chart/data/chart1.jpg"

WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ================= DOWNLOAD FROM HF =================
print("Downloading BAR model...")
bar_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=BAR_MODEL_FILE,
)

print("Downloading TEXT model...")
text_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=TEXT_MODEL_FILE,
)
print("BAR model:", bar_path)
print("TEXT model:", text_path)

# ================= RUN DETECT (BAR) =================
print("\nRunning BAR detection...")

subprocess.run(f"""
python3 {YOLO_DIR}/detect.py \
--weights {bar_path} \
--source {IMAGE_PATH} \
--save-txt --save-conf \
--project runs/detect \
--name exp_bar --exist-ok
""", shell=True)

# ================= RUN DETECT (TEXT) =================
print("\nRunning TEXT detection...")

subprocess.run(f"""
python3 {YOLO_DIR}/detect.py \
--weights {text_path} \
--source {IMAGE_PATH} \
--save-txt --save-conf \
--project runs/detect \
--name exp_text --exist-ok
""", shell=True)

print("\nDONE ALL DETECTION ✔")
print("Check: runs/detect/exp_bar and exp_text")
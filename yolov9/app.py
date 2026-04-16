import streamlit as st
import cv2
import numpy as np
import os
import subprocess
from huggingface_hub import hf_hub_download
from pipeline import build_final_output

st.set_page_config(page_title="YOLO Chart Detect", layout="centered")

st.title("📊 YOLO Chart Detection Demo")

REPO_ID = "kari512/yolov9_detect_bar_chart"

import shutil

def clean_temp_files():
    for path in [
        "runs/bar/exp",
        "runs/text/exp"
    ]:
        if os.path.exists(path):
            shutil.rmtree(path)

# ================= DOWNLOAD MODEL =================
@st.cache_resource
def load_models():
    bar = hf_hub_download(REPO_ID, "bar.pt")
    text = hf_hub_download(REPO_ID, "best.pt")
    return bar, text


bar_model, text_model = load_models()


# ================= RUN DETECT =================
def run_detect(image_path):
    os.makedirs("runs/bar", exist_ok=True)
    os.makedirs("runs/text", exist_ok=True)

    subprocess.run([
        "python3", "detect.py",
        "--weights", bar_model,
        "--source", image_path,
        "--save-txt",
        "--save-conf",
        "--project", "runs/bar",
        "--name", "exp",
        "--exist-ok"
    ])

    subprocess.run([
        "python3", "detect.py",
        "--weights", text_model,
        "--source", image_path,
        "--save-txt",
        "--save-conf",
        "--project", "runs/text",
        "--name", "exp",
        "--exist-ok"
    ])


# ================= UI =================
uploaded = st.file_uploader("Upload chart image", type=["jpg", "png", "jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Input Image", channels="BGR")

    img_path = "temp.jpg"
    cv2.imwrite(img_path, img)

    if st.button("Run Detection 🚀"):

        with st.spinner("Running YOLO..."):
            clean_temp_files()
            run_detect(img_path)

        st.success("Done!")

        base = "temp"

        bar_txt = f"runs/bar/exp/labels/{base}.txt"
        text_txt = f"runs/text/exp/labels/{base}.txt"

        st.subheader("📊 BAR DETECTION OUTPUT")

        if os.path.exists(bar_txt):
            with open(bar_txt) as f:
                st.code(f.read())
        else:
            st.error("No bar output found")

        st.subheader("📝 TEXT DETECTION OUTPUT")

        if os.path.exists(text_txt):
            with open(text_txt) as f:
                st.code(f.read())
        else:
            st.error("No text output found")

        result = build_final_output(
            img_path,
            bar_txt,
            text_txt
        )

        st.json(result)
        st.table(result["data"])
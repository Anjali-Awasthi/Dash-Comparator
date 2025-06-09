import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pytesseract
from PIL import Image
import io
from difflib import unified_diff

st.set_page_config(page_title="Dashboard Comparator", layout="wide")
st.title("Dashboard Comparator")

uploaded1 = st.file_uploader("Upload First Dashboard Image", type=["png", "jpg", "jpeg"], key="1")
uploaded2 = st.file_uploader("Upload Second Dashboard Image", type=["png", "jpg", "jpeg"], key="2")

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (800, 600))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img, gray

def compare_images(gray1, gray2):
    (score, diff) = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")
    _, thresh = cv2.threshold(cv2.absdiff(gray1, gray2), 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return score, diff, contours

def highlight_differences(img, contours):
    for c in contours:
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img

def ocr_text(img):
    return pytesseract.image_to_string(img)

if uploaded1 and uploaded2:
    img1_color, gray1 = preprocess(uploaded1.read())
    img2_color, gray2 = preprocess(uploaded2.read())

    # Compare
    score, diff, contours = compare_images(gray1, gray2)

    st.markdown(f"###  SSIM Similarity Score: `{score:.4f}`")

    # Show Original Images
    col1, col2 = st.columns(2)
    col1.image(img1_color, caption="First Dashboard", use_column_width=True)
    col2.image(img2_color, caption="Second Dashboard", use_column_width=True)

    # Highlight Differences
    img_diff_highlighted = highlight_differences(img2_color.copy(), contours)
    st.image(img_diff_highlighted, caption="🔴 Highlighted Differences", use_column_width=True)

    # OCR Comparison
    st.markdown("###  Text Difference (OCR-based)")

    text1 = ocr_text(img1_color)
    text2 = ocr_text(img2_color)

    st.markdown("**Text from First Dashboard**")
    st.code(text1, language="text")

    st.markdown("**Text from Second Dashboard**")
    st.code(text2, language="text")

    # Text diff
    st.markdown("**🔁 Text Differences**")
    diff_lines = list(unified_diff(text1.splitlines(), text2.splitlines(), lineterm=""))
    if diff_lines:
        st.code("\n".join(diff_lines), language="diff")
    else:
        st.success(" No text differences found!")

else:
    st.info("Please upload two dashboard screenshots to compare.")


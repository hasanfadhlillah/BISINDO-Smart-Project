# Aplikasi Web - Streamlit

import streamlit as st
import cv2
import numpy as np
import utils
from PIL import Image

st.set_page_config(page_title="BISINDO-Smart Live", page_icon="🖐️", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; }
    div[data-testid="stMetricValue"] { font-size: 24px; }
    </style>
    """, unsafe_allow_html=True)

# Load Model
try:
    utils.load_trained_model()
except Exception as e:
    st.error(f"❌ Error Load Model: {e}")

st.title("🖐️ BISINDO-Smart Real-Time Translator")
st.markdown("### Sistem Penerjemah Bahasa Isyarat Indonesia (BISINDO)")
st.info("Aplikasi ini menggabungkan **Pengolahan Citra Digital** (HSV, Morfologi,& Shape/Contour Extraction) dengan **Deep Learning** (CNN MobileNetV2).")

# Real-time Tuning - Sider Kalibrasi HSV
with st.sidebar:
    st.sidebar.title("⚙️ Kalibrasi Kulit (HSV)")
    st.sidebar.info("Atur slider agar tangan di kotak kecil (mask) terlihat putih bersih.")
    
    # Kita pakai session state agar nilai tidak reset saat ambil foto
    if 'h_min' not in st.session_state: st.session_state['h_min'] = 0
    if 's_min' not in st.session_state: st.session_state['s_min'] = 40
    if 'v_min' not in st.session_state: st.session_state['v_min'] = 60
    if 'h_max' not in st.session_state: st.session_state['h_max'] = 30
    if 's_max' not in st.session_state: st.session_state['s_max'] = 255
    if 'v_max' not in st.session_state: st.session_state['v_max'] = 255

    h_min = st.slider("Hue Min", 0, 179, st.session_state['h_min'])
    s_min = st.slider("Saturation Min", 0, 255, st.session_state['s_min'])
    v_min = st.slider("Value Min", 0, 255, st.session_state['v_min'])
    h_max = st.slider("Hue Max", 0, 179, st.session_state['h_max'])
    s_max = st.slider("Saturation Max", 0, 255, st.session_state['s_max'])
    v_max = st.slider("Value Max", 0, 255, st.session_state['v_max'])

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("📸 Ambil Gambar")
    # Input Kamera Native Streamlit
    img_file_buffer = st.camera_input("Arahkan tangan ke kamera lalu klik 'Take Photo'")

with col2:
    st.subheader("🧠 Hasil Analisis")
    
    if img_file_buffer is not None:
        # Konversi Buffer -> OpenCV Image
        bytes_data = img_file_buffer.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Flip Horizontal (Cermin)
        frame = cv2.flip(frame, 1)
        
        # Define ROI (Kotak Biru)
        h, w, _ = frame.shape
        box_size = 400
        start_x, start_y = max(0, (w - box_size) // 2), max(0, (h - box_size) // 2)
        end_x, end_y = min(w, start_x + box_size), min(h, start_y + box_size)
        
        roi = frame[start_y:end_y, start_x:end_x]
        
        # 1. PREPROCESSING (HSV + Morfologi)
        mask = utils.preprocess_image(
            roi, 
            h_min, s_min, v_min, 
            h_max, s_max, v_max
        )
        
        # 2. PREDIKSI
        label, conf, box = utils.predict_gesture(roi, mask)
        
        # 3. VISUALISASI
        # Tampilkan Masking
        st.image(mask, caption="Binary Mask (Segmentasi HSV + Morfologi)", width=250)
        
        # Tampilkan Hasil ROI dengan Kotak Hijau
        roi_display = roi.copy()
        if label:
            x1, y1, x2, y2 = box
            cv2.rectangle(roi_display, (x1, y1), (x2, y2), (0, 255, 0), 3)
            st.success(f"Terdeteksi: Huruf **{label}**")
            
            # Meteran Keyakinan
            st.progress(int(conf * 100))
            st.caption(f"Confidence Score: {conf*100:.2f}%")
        else:
            st.warning("Tangan tidak terdeteksi. Coba atur pencahayaan atau slider HSV.")
            
        # Tampilkan ROI Asli
        st.image(roi_display, channels="BGR", caption="Region of Interest (ROI)")

    else:
        st.info("Silakan ambil foto untuk memulai deteksi.")

st.divider()
st.caption("Made with ❤️ by SunShine Team")
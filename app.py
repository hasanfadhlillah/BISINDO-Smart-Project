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

col_control, col_cam, col_result = st.columns([1, 2, 1])

with col_control:
    st.header("⚙️ Kalibrasi")
    st.write("Geser slider sampai area tangan di kanan menjadi **PUTIH** dan latar belakang **HITAM**.")
    
    # Gunakan Session State untuk menyimpan nilai slider
    if 'h_min' not in st.session_state: st.session_state['h_min'] = 0
    if 's_min' not in st.session_state: st.session_state['s_min'] = 20 # Diturunkan biar lebih sensitif
    if 'v_min' not in st.session_state: st.session_state['v_min'] = 50 # Diturunkan biar lebih sensitif
    if 'h_max' not in st.session_state: st.session_state['h_max'] = 30
    if 's_max' not in st.session_state: st.session_state['s_max'] = 255
    if 'v_max' not in st.session_state: st.session_state['v_max'] = 255

    h_min = st.slider("Hue Min", 0, 179, st.session_state['h_min'])
    s_min = st.slider("Saturation Min", 0, 255, st.session_state['s_min'])
    v_min = st.slider("Value Min", 0, 255, st.session_state['v_min'])
    h_max = st.slider("Hue Max", 0, 179, st.session_state['h_max'])
    s_max = st.slider("Saturation Max", 0, 255, st.session_state['s_max'])
    v_max = st.slider("Value Max", 0, 255, st.session_state['v_max'])

with col_cam:
    st.header("📸 Kamera")
    img_file_buffer = st.camera_input("Ambil Foto Gestur Tangan")

with col_result:
    st.header("🧠 Analisis")

if img_file_buffer is not None:
    # Konversi ke OpenCV
    bytes_data = img_file_buffer.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Flip
    frame = cv2.flip(frame, 1)
    
    # ROI Otomatis (Tengah)
    h, w, _ = frame.shape
    box_size = 400
    start_x = max(0, (w - box_size) // 2)
    start_y = max(0, (h - box_size) // 2)
    end_x = min(w, start_x + box_size)
    end_y = min(h, start_y + box_size)
    
    roi = frame[start_y:end_y, start_x:end_x]
    
    # 1. PREPROCESSING (Selalu dijalankan)
    mask = utils.preprocess_image(roi, h_min, s_min, v_min, h_max, s_max, v_max)
    
    # 2. PREDIKSI
    label, conf, box = utils.predict_gesture(roi, mask)
    
    with col_result:
        # Tampilkan Masker
        st.image(mask, caption="Binary Mask (Putih=Tangan)", use_container_width=True)
        
        # Berikan feedback ke user
        if np.sum(mask) < 1000:
            st.warning("⚠️ **Masker terlalu gelap!**")
            st.markdown("Turunkan **Saturation Min** atau **Value Min** di sebelah kiri.")
        elif np.sum(mask) > (mask.shape[0]*mask.shape[1] * 0.8):
            st.warning("⚠️ **Masker terlalu putih!**")
            st.markdown("Naikkan **Saturation Min** agar background hilang.")
            
    with col_cam:
        roi_display = roi.copy()
        
        if label:
            x1, y1, x2, y2 = box
            # Gambar kotak hijau
            cv2.rectangle(roi_display, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            st.success(f"Huruf Terdeteksi: **{label}**")
            st.progress(int(conf * 100))
            st.caption(f"Confidence: {conf*100:.2f}%")
        else:
            st.error("❌ Objek Tidak Dikenali sebagai Tangan")
        
        st.image(roi_display, channels="BGR", caption="Hasil Deteksi", use_container_width=True)

else:
    with col_result:
        st.info("Menunggu input kamera...")

st.divider()
st.caption("Made with ❤️ by SunShine Team")
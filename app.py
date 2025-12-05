import streamlit as st
import cv2
import numpy as np
import utils
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(page_title="BISINDO-Smart", page_icon="🖐️", layout="wide")

# Load Model
try:
    utils.load_trained_model()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'bisindo_smart_model.keras' ada. Error: {e}")

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🖐️ BISINDO-Smart Translator")
st.markdown("### Sistem Penerjemah Bahasa Isyarat Indonesia (BISINDO) Berbasis AI")
st.info("Aplikasi ini menggabungkan **Pengolahan Citra Digital** (HSV & Morfologi) dengan **Deep Learning** (CNN MobileNetV2).")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚙️ Kalibrasi")
    st.write("Sesuaikan slider agar tangan terlihat putih bersih di 'Mask View'.")
    
    h_min = st.slider("Hue Min", 0, 179, 0)
    s_min = st.slider("Saturation Min", 0, 255, 40)
    v_min = st.slider("Value Min", 0, 255, 60)
    h_max = st.slider("Hue Max", 0, 179, 30)
    s_max = st.slider("Saturation Max", 0, 255, 255)
    v_max = st.slider("Value Max", 0, 255, 255)

with col2:
    st.header("📸 Kamera")
    # Input Kamera
    img_file_buffer = st.camera_input("Ambil Foto Gestur Tangan")

    if img_file_buffer is not None:
        # Convert ke OpenCV Format
        bytes_data = img_file_buffer.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Tentukan ROI (Area Tengah)
        h, w, _ = frame.shape
        start_x, start_y = (w - 400) // 2, (h - 400) // 2
        roi = frame[start_y:start_y+400, start_x:start_x+400]
        
        # 1. Preprocessing (Pakai Utils)
        mask = utils.preprocess_image(roi, h_min, s_min, v_min, h_max, s_max, v_max)
        
        # 2. Prediksi (Pakai Utils)
        label, conf, box = utils.predict_gesture(roi, mask)
        
        # Visualisasi Hasil
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.image(roi, channels="BGR", caption="Original ROI")
            if label:
                # Gambar kotak di ROI untuk visualisasi
                x1, y1, x2, y2 = box
                cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 0), 3)
                st.image(roi, channels="BGR", caption="Deteksi Tangan")

        with res_col2:
            st.image(mask, caption="Binary Mask (Hasil Segmentasi)")
            
        with res_col3:
            st.subheader("Hasil Prediksi:")
            if label:
                st.metric(label="Huruf", value=label)
                st.metric(label="Confidence", value=f"{conf*100:.2f}%")
                
                if conf > 0.8:
                    st.success("✅ Akurasi Tinggi")
                else:
                    st.warning("⚠️ Akurasi Rendah")
            else:
                st.error("❌ Tangan Tidak Terdeteksi")
# Aplikasi Web - Streamlit

import streamlit as st
import cv2
import numpy as np
import utils
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Konfigurasi Halaman
st.set_page_config(page_title="BISINDO-Smart Live", page_icon="🖐️", layout="wide")

# Load Model
utils.load_trained_model()

st.title("🖐️ BISINDO-Smart Real-Time Translator")
st.markdown("### Sistem Penerjemah Bahasa Isyarat Indonesia (BISINDO)")
st.info("Aplikasi ini menggabungkan **Pengolahan Citra Digital** (HSV & Morfologi) dengan **Deep Learning** (CNN MobileNetV2).")

# Real-time Tuning
st.sidebar.title("⚙️ Kalibrasi Kulit (HSV)")
st.sidebar.info("Atur slider ini agar tangan di kotak kecil (mask) terlihat putih bersih.")

h_min = st.sidebar.slider("Hue Min", 0, 179, 0)
s_min = st.sidebar.slider("Saturation Min", 0, 255, 40)
v_min = st.sidebar.slider("Value Min", 0, 255, 60)
h_max = st.sidebar.slider("Hue Max", 0, 179, 30)
s_max = st.sidebar.slider("Saturation Max", 0, 255, 255)
v_max = st.sidebar.slider("Value Max", 0, 255, 255)

# Video Processor
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.h_min = h_min
        self.h_max = h_max
        
    def transform(self, frame):
        # Konversi dari format WebRTC ke OpenCV
        img = frame.to_ndarray(format="bgr24")
        
        # Flip Horizontal (Cermin)
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # Define ROI (Kotak Biru)
        cv2.rectangle(img, (50, 50), (450, 450), (255, 0, 0), 2)
        roi = img[50:450, 50:450]
        
        # PREPROCESSING (HSV + Morfologi
        # Ambil nilai slider langsung dari variabel global streamlit
        mask = utils.preprocess_image(
            roi, 
            h_min, s_min, v_min, 
            h_max, s_max, v_max
        )
        
        # PREDIKSI
        label, conf, box = utils.predict_gesture(roi, mask)
        
        # VISUALISASI HASIL
        # A. Tampilkan Masking Kecil (Pojok Kanan Atas biar beda dikit)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_small = cv2.resize(mask_bgr, (120, 120))
        img[10:130, w-130:w-10] = mask_small
        cv2.rectangle(img, (w-130, 10), (w-10, 130), (0, 255, 255), 1)
        cv2.putText(img, "Binary Mask", (w-125, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        
        # B. Gambar Kotak Hijau di Tangan
        if label:
            x1, y1, x2, y2 = box
            # Offset +50 karena ROI
            cv2.rectangle(img, (50+x1, 50+y1), (50+x2, 50+y2), (0, 255, 0), 3)
            
            # Tampilkan Teks Hasil
            text = f"{label} ({conf*100:.0f}%)"
            color = (0, 255, 0) if conf > 0.8 else (0, 0, 255)
            
            # Background teks biar kebaca
            cv2.rectangle(img, (50+x1, 50+y1-30), (50+x1+200, 50+y1), (0,0,0), -1)
            cv2.putText(img, text, (50+x1+5, 50+y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI UTAMA
col_video, col_info = st.columns([3, 1])

with col_video:
    st.markdown("#### Live Webcam Feed")
    # Menjalankan WebRTC Streamer
    webrtc_streamer(
        key="bisindo-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_info:
    st.info("💡 **Panduan:**")
    st.markdown("""
    1. Izinkan akses kamera.
    2. Masukkan tangan ke dalam **Kotak Biru**.
    3. Atur slider di kiri sampai tangan di kotak kecil ("Binary Mask") terlihat **putih bersih**.
    4. Lakukan gerakan tangan BISINDO.
    """)
    st.warning("Pastikan cahaya ruangan cukup terang!")

st.divider()
st.caption("Made with ❤️ by SunShine Team")
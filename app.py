# Aplikasi Web - Streamlit

import streamlit as st
import cv2
import numpy as np
import utils
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="BISINDO-Smart Live", page_icon="🖐️", layout="wide")

# Load Model
try:
    utils.load_trained_model()
except Exception as e:
    st.error(f"❌ Error Load Model: {e}")

st.title("🖐️ BISINDO-Smart Real-Time Translator")
st.markdown("### Sistem Penerjemah Bahasa Isyarat Indonesia (BISINDO)")
st.info("Aplikasi ini menggabungkan **Pengolahan Citra Digital** (HSV, Morfologi,& Shape/Contour Extraction) dengan **Deep Learning** (CNN MobileNetV2).")

# Real-time Tuning - Sider Kalibrasi HSV
st.sidebar.title("⚙️ Kalibrasi Kulit (HSV)")
st.sidebar.info("Atur slider agar tangan di kotak kecil (mask) terlihat putih bersih.")

if 'h_min' not in st.session_state: st.session_state['h_min'] = 0
if 's_min' not in st.session_state: st.session_state['s_min'] = 40
if 'v_min' not in st.session_state: st.session_state['v_min'] = 60
if 'h_max' not in st.session_state: st.session_state['h_max'] = 30
if 's_max' not in st.session_state: st.session_state['s_max'] = 255
if 'v_max' not in st.session_state: st.session_state['v_max'] = 255

if 'h_min' not in st.session_state: st.session_state['h_min'] = 0
if 's_min' not in st.session_state: st.session_state['s_min'] = 40
if 'v_min' not in st.session_state: st.session_state['v_min'] = 60
if 'h_max' not in st.session_state: st.session_state['h_max'] = 30
if 's_max' not in st.session_state: st.session_state['s_max'] = 255
if 'v_max' not in st.session_state: st.session_state['v_max'] = 255

# Konfigurasi WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]} 
    ]}
)

# Video Processor
class HandSignProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Konversi WebRTC -> OpenCV
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Flip Horizontal (Cermin)
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # 2. Define ROI (Kotak Biru) - Fixed position
        cv2.rectangle(img, (50, 50), (450, 450), (255, 0, 0), 2)
        roi = img[50:450, 50:450]
        
        # 3. PREPROCESSING (HSV + Morfologi)
        # Mengambil nilai slider global secara real-time
        mask = utils.preprocess_image(
            roi, 
            h_min, s_min, v_min, 
            h_max, s_max, v_max
        )
        
        # 4. PREDIKSI
        label, conf, box = utils.predict_gesture(roi, mask)
        
        # 5. VISUALISASI
        # A. Tampilkan Masking Kecil (Pojok Kanan Atas)
        try:
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_small = cv2.resize(mask_bgr, (120, 120))
            # Tempel mask ke pojok kanan atas
            img[10:130, w-130:w-10] = mask_small
            cv2.rectangle(img, (w-130, 10), (w-10, 130), (0, 255, 255), 1)
            cv2.putText(img, "Binary Mask", (w-125, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        except Exception:
            pass # Hindari crash jika resize gagal

        # B. Gambar Kotak Hijau & Hasil
        if label:
            x1, y1, x2, y2 = box
            # Offset +50 karena ROI dimulai dari (50,50)
            cv2.rectangle(img, (50+x1, 50+y1), (50+x2, 50+y2), (0, 255, 0), 3)
            
            text = f"{label} ({conf*100:.0f}%)"
            color = (0, 255, 0) if conf > 0.8 else (0, 0, 255)
            
            # Background teks hitam biar terbaca
            cv2.rectangle(img, (50+x1, 50+y1-30), (50+x1+200, 50+y1), (0,0,0), -1)
            cv2.putText(img, text, (50+x1+5, 50+y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Kembalikan frame ke WebRTC
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Layout Utama
col_video, col_info = st.columns([3, 1])

with col_video:
    st.markdown("#### Live Webcam Feed")
    # Menjalankan Streamer dengan Settingan Hemat Bandwidth
    webrtc_streamer(
        key="bisindo-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=HandSignProcessor,
        media_stream_constraints={
            "video": {"width": 480, "height": 360, "frameRate": 15}, # Resolusi rendah agar lancar
            "audio": False
        },
        async_processing=True,
    )

with col_info:
    st.info("💡 **Panduan:**")
    st.markdown("""
    1. Izinkan akses kamera pada browser.
    2. Tunggu status berubah jadi "Running".
    3. Masukkan tangan ke dalam **Kotak Biru**.
    4. Atur slider di kiri sampai tangan di kotak kecil ("Binary Mask") terlihat **putih**.
    5. Lakukan gerakan tangan BISINDO.
    """)
    st.warning("Pastikan cahaya ruangan cukup terang!")

st.divider()
st.caption("Made with ❤️ by SunShine Team")
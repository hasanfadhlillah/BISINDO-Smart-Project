import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# KONFIGURASI SISTEM
MODEL_PATH = 'bisindo_smart_model.h5'
IMG_SIZE = 128  

# Label Kelas
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

print("🔄 Loading Model Cerdas...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model Siap!")

# Smoothing Prediksi (Agar huruf tidak kedip-kedip/berubah terlalu cepat)   
prediction_history = deque(maxlen=8)

def nothing(x):
    pass

# SETUP GUI KALIBRASI (Materi Modul 3 Color Space)
cv2.namedWindow("Kalibrasi HSV")
cv2.resizeWindow("Kalibrasi HSV", 400, 300)
# Nilai default ini disetting untuk kulit asia di cahaya ruangan normal
cv2.createTrackbar("H Min", "Kalibrasi HSV", 0, 179, nothing)
cv2.createTrackbar("S Min", "Kalibrasi HSV", 40, 255, nothing)
cv2.createTrackbar("V Min", "Kalibrasi HSV", 60, 255, nothing)
cv2.createTrackbar("H Max", "Kalibrasi HSV", 30, 179, nothing)
cv2.createTrackbar("S Max", "Kalibrasi HSV", 255, 255, nothing)
cv2.createTrackbar("V Max", "Kalibrasi HSV", 255, 255, nothing)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Flip horizontal (Efek Cermin)
    frame = cv2.flip(frame, 1)
    h_frame, w_frame, _ = frame.shape
    
    # Area Scan (Kotak Biru Statis di Tengah)
    cv2.rectangle(frame, (50, 50), (450, 450), (255, 0, 0), 1)
    roi_area = frame[50:450, 50:450]

    # 1. PREPROCESSING (Materi Modul 3 & 10)
    # Konversi ke HSV
    hsv = cv2.cvtColor(roi_area, cv2.COLOR_BGR2HSV)
    
    # Ambil nilai slider realtime
    h_min = cv2.getTrackbarPos("H Min", "Kalibrasi HSV")
    s_min = cv2.getTrackbarPos("S Min", "Kalibrasi HSV")
    v_min = cv2.getTrackbarPos("V Min", "Kalibrasi HSV")
    h_max = cv2.getTrackbarPos("H Max", "Kalibrasi HSV")
    s_max = cv2.getTrackbarPos("S Max", "Kalibrasi HSV")
    v_max = cv2.getTrackbarPos("V Max", "Kalibrasi HSV")

    # Segmentasi Kulit (Thresholding)
    lower_skin = np.array([h_min, s_min, v_min])
    upper_skin = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 2. OPERASI MORFOLOGI (Materi Modul 11)
    kernel = np.ones((5, 5), np.uint8)
    
    # Opening: Hilangkan bintik putih di background
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Closing: Tutup lubang hitam di dalam tangan
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Smoothing
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # 3. FEATURE EXTRACTION: CONTOURS (Materi Modul P17)
    # Cari kontur pada mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    char_detect = ""
    confidence = 0.0
    
    # Jika ada kontur (tangan terdeteksi)
    if len(contours) > 0:
        # Ambil kontur terbesar (asumsi itu tangan)
        c = max(contours, key=cv2.contourArea)
        
        # Filter noise: hanya proses jika area cukup besar
        if cv2.contourArea(c) > 3000:
            # Dapatkan Bounding Box (Kotak Hijau Dinamis)
            x, y, w, h = cv2.boundingRect(c)
            
            # Tambahkan padding agar tidak terlalu ngepas
            pad = 20
            x_roi = max(0, x - pad)
            y_roi = max(0, y - pad)
            w_roi = min(400, w + 2*pad)
            h_roi = min(400, h + 2*pad)

            # Gambar Kotak Hijau yang MENGIKUTI TANGAN (Tracking Sederhana)
            # Offset +50 karena roi_area mulai dari (50,50)
            cv2.rectangle(frame, (50+x_roi, 50+y_roi), (50+x_roi+w_roi, 50+y_roi+h_roi), (0, 255, 0), 2)

            # 4. KLASIFIKASI HYBRID STEP
            # Crop gambar tangan ASLI (RGB), bukan Mask agar akurasi tinggi (tekstur terlihat)
            hand_img_rgb = roi_area[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
            
            if hand_img_rgb.size > 0:
                # Resize ke input model
                img_input = cv2.resize(hand_img_rgb, (IMG_SIZE, IMG_SIZE))
                img_input = np.expand_dims(img_input, axis=0)
                img_input = img_input / 255.0 # Normalisasi
                
                # Prediksi
                preds = model.predict(img_input, verbose=0)
                idx = np.argmax(preds)
                conf = preds[0][idx]
                
                # Masukkan ke history (Stabilizer)
                prediction_history.append(idx)
                
                # Ambil voting terbanyak dari 8 frame terakhir
                final_idx = max(set(prediction_history), key=prediction_history.count)
                
                char_detect = CLASSES[final_idx]
                confidence = conf

    # 5. VISUALISASI DASHBOARD
    # A. Tampilkan Masking (Bukti Preprocessing) di Pojok Kiri
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    frame[10:130, 10:130] = cv2.resize(mask_bgr, (120, 120))
    cv2.rectangle(frame, (10, 10), (130, 130), (0, 255, 255), 2)
    cv2.putText(frame, "Binary Mask", (15, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    # B. Panel Informasi Hasil
    cv2.rectangle(frame, (460, 50), (630, 160), (0, 0, 0), -1) # Background hitam
    cv2.putText(frame, "Hasil:", (470, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    
    color_res = (0, 255, 0) if confidence > 0.8 else (0, 0, 255) # Hijau jika yakin
    cv2.putText(frame, f"{char_detect}", (530, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, color_res, 4)
    
    cv2.putText(frame, f"Conf: {confidence*100:.1f}%", (470, 150), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)

    cv2.imshow("BISINDO-Smart System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
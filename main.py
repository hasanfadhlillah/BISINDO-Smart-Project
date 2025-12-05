import cv2
import utils
from collections import deque

# Load Model
utils.load_trained_model()

# Smoothing
prediction_history = deque(maxlen=8)

def nothing(x): pass

# Setup GUI
cv2.namedWindow("Kalibrasi HSV")
cv2.resizeWindow("Kalibrasi HSV", 400, 300)
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
    frame = cv2.flip(frame, 1)
    
    # Define ROI
    cv2.rectangle(frame, (50, 50), (450, 450), (255, 0, 0), 1)
    roi_area = frame[50:450, 50:450]
    
    # Ambil nilai Trackbar
    h_min = cv2.getTrackbarPos("H Min", "Kalibrasi HSV")
    s_min = cv2.getTrackbarPos("S Min", "Kalibrasi HSV")
    v_min = cv2.getTrackbarPos("V Min", "Kalibrasi HSV")
    h_max = cv2.getTrackbarPos("H Max", "Kalibrasi HSV")
    s_max = cv2.getTrackbarPos("S Max", "Kalibrasi HSV")
    v_max = cv2.getTrackbarPos("V Max", "Kalibrasi HSV")
    
    # 1. Preprocessing (Panggil dari utils)
    mask = utils.preprocess_image(roi_area, h_min, s_min, v_min, h_max, s_max, v_max)
    
    # 2. Prediksi (Panggil dari utils)
    label, conf, box = utils.predict_gesture(roi_area, mask)
    
    final_label = "-"
    
    if label:
        # Smoothing Hasil
        prediction_history.append(label)
        final_label = max(set(prediction_history), key=prediction_history.count)
        
        # Gambar Kotak Hijau Dinamis
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (50+x1, 50+y1), (50+x2, 50+y2), (0, 255, 0), 2)
    
    # Visualisasi Dashboard
    # A. Masking Kecil
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    frame[10:130, 10:130] = cv2.resize(mask_bgr, (120, 120))
    cv2.rectangle(frame, (10, 10), (130, 130), (0, 255, 255), 2)
    cv2.putText(frame, "Binary Mask", (15, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    # B. Panel Hasil
    cv2.rectangle(frame, (460, 50), (630, 160), (0, 0, 0), -1)
    cv2.putText(frame, "Hasil:", (470, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    
    color_res = (0, 255, 0) if conf > 0.8 else (0, 0, 255)
    cv2.putText(frame, f"{final_label}", (530, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, color_res, 4)
    cv2.putText(frame, f"Conf: {conf*100:.1f}%", (470, 150), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)

    cv2.imshow("BISINDO-Smart System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
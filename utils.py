# Fungsi Inti: Preprocessing, Morfologi, Prediksi

import cv2
import numpy as np
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise ImportError("Tidak ada module TFLite maupun TensorFlow.")

# Global Variables
interpreter = None
input_details = None
output_details = None

# Konfigurasi Global
IMG_SIZE = 128
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def load_trained_model(path='bisindo_smart_model.tflite'):
    global interpreter, input_details, output_details
    
    # Load Interpreter
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    
    # Dapatkan info input/output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter

def preprocess_image(roi, h_min, s_min, v_min, h_max, s_max, v_max):
    """
    Melakukan Preprocessing (HSV -> Thresholding -> Morphology)
    Materi: Modul 3 (Color Space) & Modul 11 (Morfologi)
    """
    # 1. Konversi RGB ke HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 2. Segmentasi Kulit (Thresholding)
    lower_skin = np.array([h_min, s_min, v_min])
    upper_skin = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # 3. Operasi Morfologi (Opening & Closing)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Opening: Hilangkan bintik putih di background
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Closing: Tutup lubang hitam di dalam tangan
    mask = cv2.GaussianBlur(mask, (5, 5), 0) # Smoothing
    
    return mask

def predict_gesture(roi, mask):
    """
    Melakukan Prediksi menggunakan Model CNN
    Materi: Featrure Extraction / Klasifikasi Citra
    """
    # Cari Kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 2000: # Filter noise kecil
            # Auto-Crop (Materi P17: Bounding Rect)
            x, y, w, h = cv2.boundingRect(c)
            
            # Padding agar tidak terlalu ngepas
            pad = 20
            h_img, w_img, _ = roi.shape
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)
            
            # Crop gambar RGB asli (Hybrid Approach)
            hand_img = roi[y1:y2, x1:x2]
            
            if hand_img.size > 0:
                # Resize & Normalize
                img_input = cv2.resize(hand_img, (128, 128))
                img_input = np.expand_dims(img_input, axis=0).astype(np.float32) 
                img_input = img_input / 255.0
                
                interpreter.set_tensor(input_details[0]['index'], img_input)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                idx = np.argmax(output_data)
                conf = output_data[0][idx]
                label = CLASSES[idx]
                
                return label, conf, (x1, y1, x2, y2)
    
    return None, 0.0, None
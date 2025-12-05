# Fungsi Inti: Preprocessing, Morfologi, Prediksi

import cv2
import numpy as np

# INISIALISASI VARIABEL GLOBAL (PENTING!)
USING_TFLITE = False
interpreter = None
model = None
input_details = None
output_details = None

try:
    # Coba TFLite Runtime (Untuk Streamlit Cloud / Perangkat Kecil)
    import tflite_runtime.interpreter as tflite
    USING_TFLITE = True
    print("✅ Menggunakan TFLite Runtime (Optimized)")
except ImportError:
    try:
        # Coba TensorFlow Lite bawaan (Untuk Laptop Windows)
        import tensorflow.lite as tflite
        USING_TFLITE = True
        print("✅ Menggunakan TensorFlow Lite (Standard)")
    except ImportError:
        try:
            import tensorflow as tf
            USING_TFLITE = False
            print("✅ Menggunakan TensorFlow Keras (Heavy)")
        except ImportError:
            print("⚠️ Tidak ada library TensorFlow/TFLite)")

CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def load_trained_model(path_tflite='bisindo_smart_model.tflite', path_keras='bisindo_smart_model.keras'):
    global interpreter, model, input_details, output_details
    
    if USING_TFLITE:
        try:
            interpreter = tflite.Interpreter(model_path=path_tflite)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            return interpreter
        except Exception as e:
            print(f"❌ Gagal load TFLite: {e}")
    else:
        if model is None:
            try:
                model = tf.keras.models.load_model(path_keras, compile=False)
                return model
            except Exception as e:
                print(f"❌ Gagal load Keras: {e}")
    
    return None

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
    # Cari Kontur Tangan
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
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
                # Resize ke 128x128
                img_resized = cv2.resize(hand_img, (128, 128))
                
                if USING_TFLITE and interpreter is not None:
                    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)
                    input_data = input_data / 255.0 # Normalisasi 0-1
                    
                    # Set Tensor
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    
                    # Ambil Output
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    
                    idx = np.argmax(output_data[0]) # Ambil index terbesar
                    conf = float(output_data[0][idx]) # Ambil nilai confidence
                    
                elif not USING_TFLITE and model is not None:
                    input_data = np.expand_dims(img_resized, axis=0)
                    input_data = input_data / 255.0
                    
                    preds = model.predict(input_data, verbose=0)
                    idx = np.argmax(preds)
                    conf = float(preds[0][idx])
                
                else:
                    return None, 0.0, None

                # Mapping ke Huruf
                if idx < len(CLASSES):
                    label = CLASSES[idx]
                    return label, conf, (x1, y1, x2, y2)
    
    return None, 0.0, None
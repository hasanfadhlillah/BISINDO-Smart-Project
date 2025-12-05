# 🖐️ BISINDO-Smart Project

**BISINDO-Smart: Penerjemah Bahasa Isyarat Indonesia (BISINDO) Real-Time Menggunakan Pelacakan Tangan Berbasis Operasi Morfologi dan Klasifikasi MobileNetV2.**

Project ini merupakan **Final Project Mata Kuliah Pengolahan Citra Digital dan Visi Komputer** di Fakultas Ilmu Komputer, Universitas Brawijaya. Sistem ini menggabungkan teknik *Classical Computer Vision* (untuk deteksi ROI tangan) dengan *Deep Learning* (untuk klasifikasi huruf).

---

## 🎬 Dokumentasi & Demo

Untuk melihat penjelasan visual dan demonstrasi cara kerja sistem, silakan akses tautan berikut:

- 📢 **Poster Publikasi:** [Lihat Poster (clips.id/Poster-BISINDO-Smart)](https://clips.id/Poster-BISINDO-Smart)
- 🎥 **Video Demo & Penjelasan:** [Tonton di YouTube (clips.id/VideoYoutube-BISINDO-Smart)](https://clips.id/VideoYoutube-BISINDO-Smart)

---

## 📂 Struktur Direktori

Berikut adalah penjelasan fungsi dari setiap file yang ada dalam repositori ini:

```text
BISINDO-Smart-Project/
│
├── 📜 BISINDO_Smart_Hasan_dan_Husain.ipynb  # [TRAINING] Notebook utama untuk melatih model AI (MobileNetV2), augmentasi data, dan evaluasi akurasi.
│
├── 🧠 bisindo_smart_model.tflite            # [MODEL] File model versi ringan (TFLite) untuk deployment di Web/HP.
├── 🧠 bisindo_smart_model.h5                # [MODEL] File model format H5 (backup).
├── 🧠 bisindo_smart_model.keras             # [MODEL] File model format Keras terbaru.
│
├── 🛠️ utils.py                              # [CORE LOGIC] Berisi fungsi inti PCD: Preprocessing (HSV), Morfologi (Opening/Closing), dan Prediksi.
│
├── 💻 main.py                               # [APP LOKAL] Source code untuk aplikasi Desktop (menggunakan OpenCV window). Lebih smooth & real-time.
├── 🌐 app.py                                # [APP WEB] Source code untuk aplikasi Web (menggunakan Streamlit).
│
├── 📄 requirements.txt                      # Daftar pustaka/library yang dibutuhkan untuk menjalankan project.
└── 🖼️ Arsitektur Diagram/                   # Folder berisi diagram alur sistem dan dokumentasi visual.
````

-----

## 🚀 Cara Menjalankan (Replikasi)

Aplikasi ini dapat diakses melalui dua cara: **Web Public** (Praktis) atau **Lokal** (Performa Maksimal).

### 1\. Akses Web (Publik)

Versi ini dapat langsung diakses tanpa instalasi apapun melalui browser.

  * **Link:** [bisindo-smart.streamlit.app](https://www.google.com/search?q=https://bisindo-smart.streamlit.app)
  * *Catatan:* Pastikan memberikan izin akses kamera pada browser.

### 2\. Akses Lokal (Di Komputer)

Versi ini direkomendasikan untuk performa **Real-Time Video** yang lancar tanpa *delay*.

**Langkah-langkah:**

1.  **Clone Repository** atau Download ZIP project ini.
2.  Buka terminal (CMD/Git Bash/Terminal VS Code) di folder project.
3.  **Install Library** yang dibutuhkan:
    ```bash
    pip install -r requirements.txt
    # Atau install manual:
    pip install opencv-python numpy matplotlib tensorflow streamlit
    ```
4.  **Jalankan Aplikasi:**
    ```bash
    python main.py
    ```
5.  Akan muncul jendela kamera OpenCV dengan slider pengaturan HSV untuk kalibrasi pencahayaan.

-----

## ⚠️ Batasan & Catatan Teknis (Web vs Lokal)

Terdapat perbedaan performa yang signifikan antara versi Web (Streamlit) dan Lokal (OpenCV), berikut penjelasannya untuk pertimbangan penilaian:

| Fitur | ✅ Versi Lokal (`main.py`) | ⚠️ Versi Web (`app.py`) |
| :--- | :--- | :--- |
| **Kecepatan Video** | **Real-Time (30+ FPS)**. Sangat lancar karena mengakses hardware kamera secara langsung (*native*). | **Latensi Tinggi (Lag)**. Terasa patah-patah karena Streamlit memproses gambar dengan metode *snapshot* (mengirim frame satu per satu ke server/cloud), bukan *streaming* murni. |
| **Kalibrasi** | **Live Trackbar**. Perubahan slider HSV langsung terlihat efeknya di layar saat itu juga. | **Reloading**. Setiap kali slider digeser, halaman web akan me-*reload* script, membuat proses kalibrasi sedikit lebih lambat. |
| **Kestabilan** | Menggunakan *Temporal Smoothing* (Voting 30 frame) sehingga prediksi huruf sangat stabil. | Prediksi dilakukan per-frame (single shot) karena keterbatasan *state* di web, sehingga hasil prediksi mungkin berkedip (*flickering*). |

**Kesimpulan:** Untuk pengujian akurasi dan pengalaman pengguna terbaik, sangat disarankan menggunakan **Versi Lokal**. Versi Web disediakan untuk kemudahan akses dan demonstrasi portabilitas model.

-----

## 👨‍💻 Tim Pengembang

  * **Muhammad Hasan Fadhlillah** (225150207111026)
  * **Muhammad Husain Fadhlillah** (225150207111027)

*Fakultas Ilmu Komputer, Universitas Brawijaya - 2025*

# 📘 Judul Proyek
*ANALISIS PREDIKSI KONSUMSI BAHAN BAKAR (MPG) MENGGUNAKAN BASELINE, MACHINE LEARNING, DAN DEEP LEARNING*

## 👤 Informasi
- **Nama:** Mohammad Dimas Bahrul Ikhwani 
- **Repo:** https://github.com/Bahrul24/234311017_UAS_DataScience  
- **Video:** [...]  

---

# 1. 🎯 Ringkasan Proyek
Proyek ini bertujuan untuk memprediksi efisiensi bahan bakar kendaraan (MPG - Miles Per Gallon) berdasarkan spesifikasi teknis mobil.

* **Tujuan:** Memprediksi nilai kontinu `mpg` (Regression Task).
* **Pendekatan:** Membangun 3 model perbandingan:
    1.  **Baseline:** Dummy Regressor (Mean Strategy).
    2.  **Machine Learning:** Random Forest Regressor.
    3.  **Deep Learning:** Artificial Neural Network (TensorFlow/Keras).
* **Evaluasi:** Menggunakan metrik MAE (Mean Absolute Error), MSE, dan R² Score. 

---

# 2. 📄 Problem & Goals  
### Problem Statements:
* Diperlukan cara untuk memperkirakan konsumsi bahan bakar mobil baru sebelum diproduksi massal berdasarkan spesifikasi mesin.
* Hubungan antara fitur seperti *horsepower*, *weight*, dan *displacement* terhadap irit-borosnya bensin seringkali tidak linear.
* Dataset memiliki *missing values* pada kolom vital (horsepower) yang perlu ditangani agar tidak merusak model.

### Goals:
* Membangun model regresi yang akurat untuk memprediksi MPG.
* Membandingkan performa antara metode statistik sederhana (Baseline), Machine Learning (Random Forest), dan Deep Learning.
* Memberikan pipeline analisis yang lengkap mulai dari *data cleaning* hingga evaluasi model.

---
## 📁 Struktur Folder
```
project/
│
├── data/                   # Dataset (tidak di-commit, download manual)
│
├── notebooks/              # Jupyter notebooks
│   └── ML_Project.ipynb
│
├── src/                    # Source code
│   
├── models/                 # Saved models
│   ├── model_baseline.pkl
│   ├── model_rf.pkl
│   └── model_cnn.h5
│
├── images/                 # Visualizations
│   └── r
│
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```
---

# 3. 📊 Dataset
- **Sumber:** UCI Machine Learning Repository (Auto MPG Data Set)
- **Jumlah Data:** 398 baris.
- **Tipe:** Tabular (Regresi).

### Fitur Utama
Berikut adalah fitur yang digunakan untuk prediksi:

| **Nama Fitur** | **Deskripsi** |
|---|---|
| **cylinders** | Jumlah silinder pada mesin (3-8). |
| **displacement** | Kapasitas ruang bakar mesin (cu. in.). |
| **horsepower** | Tenaga kuda (indikator kekuatan mesin). |
| **weight** | Berat kendaraan (lbs). Fitur yang sangat berpengaruh. |
| **acceleration** | Waktu tempuh 0-60 mph (detik). |
| **model_year** | Tahun pembuatan mobil (70-82). |
| **origin** | Kode negara asal (1: USA, 2: Europe, 3: Japan). |
| **mpg** | **Target** – Konsumsi bahan bakar (Miles Per Gallon) yang diprediksi. |

---

# 4. 🔧 Data Preparation
- **Cleaning**: Mengidentifikasi nilai `'?'` pada kolom `horsepower`, mengubahnya menjadi NaN, dan melakukan imputasi menggunakan nilai **median**.
- **Feature Selection**: Menghapus kolom `car_name` karena berupa ID unik (text) yang tidak relevan untuk prediksi numerik.
- **Scaling**: Menerapkan StandardScaler (Mean=0, Std=1) untuk menormalisasi fitur numerik (sangat penting untuk Deep Learning).
- **Splitting**: Pembagian dataset menjadi 80% train dan 20% test menggunakan `train_test_split` dengan `random_state = 42`.
- **Balancing**: Tidak diperlukan karena dataset bersifat regresi.

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** [Dummy Regressor - Mean Strategy]
- **Model 2 – Advanced ML:** [Random Forest Regressor]
- **Model 3 – Deep Learning:** [Sequential Neural Network / MLP]

---

# 6. 🧪 Evaluation
**Metrik:** MSE, MAE, R2

| Model | MSE | MAE | R² | Keterangan |
|---|---|---|---|---|
| Baseline (Mean) | 66.85 | 6.53 | -0.005 | Model Terburuk |
| Advanced (Random Forest) | 6.24 | 1.85 | 0.89 | Model Terbaik |
| Deep Learning (MLP) | 7.15 | 1.98 | 0.88 | Kompetitif |

---

# 7. 🏁 Kesimpulan

- **Model terbaik:** Random Forest (Machine Learning)
- **Alasan:**
  - Memiliki **R² Score tertinggi (0.89)**, yang berarti model mampu menjelaskan 89% variasi data.
  - Memiliki error (MAE) terendah dibandingkan model lain.
  - Lebih stabil pada dataset berukuran kecil/menengah (tabular data) dibandingkan Deep Learning.

- **Insight penting:**
  - **Berat Kendaraan (Weight)** dan **Kapasitas Mesin (Displacement)** memiliki korelasi negatif yang kuat dengan MPG (semakin berat mobil, semakin boros).
  - Mobil buatan tahun yang lebih baru (`model_year`) cenderung lebih irit bahan bakar.
  - Deep Learning memberikan hasil yang mendekati Random Forest, namun membutuhkan preprocessing yang lebih ketat (scaling).

---

# 8. 🔮 Future Work

## 📌 Data Improvements
- [x] Mengumpulkan data mobil modern (tahun 2000+)
- [x] Menambah variasi tipe kendaraan (listrik/hybrid)
- [x] Melakukan feature engineering lebih lanjut

## 🤖 Model Enhancements
- [x] Mencoba arsitektur deep learning yang lebih kompleks
- [x] Hyperparameter tuning (GridSearch) pada Random Forest
- [x] Mencoba ensemble methods (XGBoost/LightGBM)
- [ ] Transfer learning dengan model yang lebih besar

## 🚀 Deployment & System
- [x] Membuat API (Flask / FastAPI)
- [x] Membuat web app (Streamlit / Gradio)
- [ ] Containerization dengan Docker
- [ ] Deploy ke cloud (Heroku / GCP / AWS)

## ⚙️ Optimization
- [ ] Model compression (pruning / quantization)
- [x] Improving inference speed
- [x] Reducing model size

---

# 9. 🔁 Reproducibility
Gunakan environment:
**Python Version:**
- Python 3.10+

**Main Libraries & Versions:**
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

**Deep Learning Framework:**
- tensorflow
- keras
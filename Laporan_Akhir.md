# INFORMASI PROYEK

**Judul Proyek:** Analisis Prediksi Konsumsi Bahan Bakar (MPG) Menggunakan Baseline, Machine Learning, dan Deep Learning

**Nama Mahasiswa:** Mohammad Dimas Bahrul Ikhwani  
**NIM:** [Masukkan NIM Anda]  
**Program Studi:** [Masukkan Prodi Anda]  
**Mata Kuliah:** Data Science / Machine Learning  
**Dosen Pengampu:** [Masukkan Nama Dosen]  
**Tahun Akademik:** 2024/2025  
**Link GitHub Repository:** [Masukkan Link Repo Anda]  
**Link Video Pembahasan:** [Masukkan Link Video Anda (Opsional)]  

---

## 1. LEARNING OUTCOMES

Pada proyek ini, mahasiswa diharapkan dapat:
1. ✅ Memahami konteks masalah dan merumuskan problem statement secara jelas.
2. ✅ Melakukan analisis dan eksplorasi data (EDA) secara komprehensif.
3. ✅ Melakukan data preparation yang sesuai dengan karakteristik dataset (Cleaning & Scaling).
4. ✅ Mengembangkan tiga model machine learning yang terdiri dari (**WAJIB**):
   - Model Baseline (Dummy Regressor)
   - Model Machine Learning (Random Forest)
   - Model Deep Learning (Neural Network/MLP)
5. ✅ Menggunakan metrik evaluasi yang relevan (MAE, MSE, R² Score).
6. ✅ Melaporkan hasil eksperimen secara ilmiah dan sistematis.
7. ✅ Mengunggah seluruh kode proyek ke GitHub.
8. ✅ Menerapkan prinsip software engineering dalam pengembangan proyek.

---

## 2. PROJECT OVERVIEW

### 2.1 Latar Belakang
Efisiensi bahan bakar (*Miles Per Gallon* / MPG) adalah salah satu faktor kunci dalam industri otomotif. Produsen mobil perlu mengestimasi konsumsi bahan bakar kendaraan baru sebelum diproduksi secara massal berdasarkan spesifikasi desain mesin (seperti jumlah silinder, kapasitas mesin, dan berat kendaraan).

Namun, hubungan antara spesifikasi teknis tersebut dengan konsumsi bahan bakar seringkali **non-linear** dan kompleks. Oleh karena itu, pendekatan manual seringkali tidak akurat.

Pentingnya proyek ini:
* **Bagi Produsen:** Membantu desain mesin yang lebih efisien.
* **Bagi Konsumen:** Estimasi biaya bahan bakar.

### 2.2 Problem Statement
1.  Bagaimana cara memprediksi nilai konsumsi bahan bakar (MPG) secara akurat berdasarkan fitur teknis kendaraan?
2.  Dataset memiliki *missing values* pada fitur vital (horsepower) yang perlu ditangani.
3.  Perlu diketahui apakah model kompleks (Deep Learning) memberikan performa yang jauh lebih baik dibandingkan model Machine Learning klasik (Random Forest) pada data tabular berukuran kecil.

### 2.3 Goals
1.  Membangun model regresi untuk memprediksi variabel target `mpg` dengan R² Score > 0.80.
2.  Melakukan komparasi performa antara **Baseline**, **Random Forest**, dan **Deep Learning**.
3.  Menentukan model terbaik berdasarkan metrik evaluasi Error (MAE) dan Akurasi (R²).

### 2.4 Solution Approach
Proyek ini menggunakan tiga pendekatan model:
* **Model 1 (Baseline):** `DummyRegressor` (Menggunakan rata-rata/Mean).
* **Model 2 (Advanced ML):** `RandomForestRegressor` (Ensemble Learning).
* **Model 3 (Deep Learning):** `Multi-Layer Perceptron (MLP)` menggunakan TensorFlow/Keras.

---

## 3. DATA UNDERSTANDING

### 3.1 Informasi Dataset
* **Sumber Dataset:** UCI Machine Learning Repository (Auto MPG Data Set).
* **Jumlah Baris:** 398 baris.
* **Jumlah Kolom:** 9 kolom (8 Fitur + 1 Target).
* **Tipe Data:** Tabular.
* **Tipe Tugas:** Regresi (Supervised Learning).

### 3.2 Deskripsi Fitur

| Nama Fitur | Tipe Data | Deskripsi |
| :--- | :--- | :--- |
| **cylinders** | Integer | Jumlah silinder mesin (3, 4, 5, 6, 8). |
| **displacement** | Float | Kapasitas ruang bakar mesin (cu. in.). |
| **horsepower** | Float | Tenaga kuda mesin (indikator kekuatan). |
| **weight** | Float | Berat kendaraan (lbs). |
| **acceleration** | Float | Waktu tempuh 0-60 mph (detik). |
| **model_year** | Integer | Tahun pembuatan mobil (70-82). |
| **origin** | Categorical | Asal negara (1: USA, 2: Europe, 3: Japan). |
| **car_name** | String | Nama mobil (Dihapus karena ID unik). |
| **mpg** | **Float (Target)** | Konsumsi bahan bakar (Miles Per Gallon). |

### 3.3 Kondisi Data
* **Missing Values:** Terdapat 6 nilai `'?'` pada kolom `horsepower`.
* **Outliers:** Terdapat beberapa outlier pada `acceleration` dan `horsepower`, namun masih dalam batas wajar fisika kendaraan.
* **Scale:** Rentang nilai antar fitur sangat berbeda (contoh: `weight` ribuan, `cylinders` satuan), sehingga memerlukan Scaling.

### 3.4 Exploratory Data Analysis (EDA)

**Visualisasi 1: Korelasi Heatmap**
*(Anda dapat memasukkan gambar heatmap dari notebook Anda di sini)*
> **Insight:** Fitur `weight` dan `displacement` memiliki korelasi negatif yang sangat kuat dengan `mpg`. Artinya, semakin berat mobil, semakin boros bahan bakar.

---

## 4. DATA PREPARATION

### 4.1 Data Cleaning
* **Handling Missing Values:** Mengganti nilai `'?'` pada `horsepower` dengan `NaN`, kemudian melakukan imputasi menggunakan nilai **Median**. Median dipilih karena lebih robust terhadap outlier dibandingkan Mean.
* **Drop Columns:** Menghapus kolom `car_name` karena berupa teks unik yang tidak memiliki pola general untuk prediksi.

### 4.2 Data Transformation (Scaling)
* **StandardScaler:** Digunakan untuk menormalisasi fitur numerik (Mean=0, Std=1).
* **Alasan:** Algoritma Deep Learning (Neural Network) sangat sensitif terhadap skala data. Tanpa scaling, proses training akan lambat dan sulit mencapai konvergensi (loss minimum).

### 4.3 Data Splitting
* **Ratio:** 80% Training, 20% Testing.
* **Random State:** 42 (untuk reproducibility).

---

## 5. MODELING

### 5.1 Model 1 — Baseline Model
* **Nama Model:** Dummy Regressor.
* **Strategi:** `strategy="mean"`.
* **Alasan:** Model ini hanya memprediksi nilai rata-rata dari data training untuk semua data baru. Digunakan sebagai patokan terendah. Jika model ML/DL tidak bisa mengalahkan skor ini, maka model tersebut gagal.

### 5.2 Model 2 — ML / Advanced Model
* **Nama Model:** Random Forest Regressor.
* **Hyperparameter:**
    * `n_estimators`: 100 (Jumlah pohon keputusan).
    * `random_state`: 42.
* **Alasan:** Random Forest mampu menangkap hubungan non-linear tanpa perlu scaling data (sebenarnya), tahan terhadap outlier, dan umumnya memberikan performa terbaik untuk data tabular.

### 5.3 Model 3 — Deep Learning Model (WAJIB)
* **Jenis:** Multilayer Perceptron (MLP) untuk Tabular Data.
* **Framework:** TensorFlow / Keras.

#### 5.3.1 Arsitektur Model
| Layer Type | Output Shape | Param # | Aktivasi | Deskripsi |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | (None, 7) | 0 | - | Input Layer (7 Fitur) |
| **Dense** | (None, 64) | 512 | ReLU | Hidden Layer 1 |
| **Dropout** | (None, 64) | 0 | - | Rate = 0.2 (Mencegah Overfitting) |
| **Dense** | (None, 32) | 2,080 | ReLU | Hidden Layer 2 |
| **Dense** | (None, 1) | 33 | Linear | Output Layer (Regresi) |

* **Total Parameters:** 2,625 Trainable params.

#### 5.3.2 Training Configuration
* **Optimizer:** Adam.
* **Loss Function:** MSE (Mean Squared Error).
* **Metrics:** MAE (Mean Absolute Error).
* **Epochs:** 100.
* **Batch Size:** 32.

---

## 6. EVALUATION

### 6.1 Metrik Evaluasi
Proyek ini menggunakan metrik Regresi:
1.  **MAE (Mean Absolute Error):** Rata-rata kesalahan mutlak (lebih mudah diinterpretasikan manusia).
2.  **MSE (Mean Squared Error):** Memberikan penalti lebih besar pada kesalahan prediksi yang jauh (outlier).
3.  **R² Score:** Menjelaskan seberapa baik model mewakili variansi data (Mendekati 1.0 = Sempurna).

### 6.2 Hasil Evaluasi Model

| Model | MAE (Error) | MSE | R² Score | Training Time |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Mean)** | 5.96 | 53.98 | -0.004 | < 0.1s |
| **Random Forest** | **1.58** | **4.58** | **0.91** | ~0.5s |
| **Deep Learning** | 1.85 | 6.16 | 0.88 | ~5.0s |

### 6.3 Analisis Hasil & Perbandingan
1.  **Baseline vs Machine Learning:**
    Model Baseline gagal total (R² negatif), sedangkan Random Forest dan Deep Learning berhasil mencapai R² di atas 0.88. Ini membuktikan bahwa fitur teknis mobil sangat mempengaruhi MPG.
2.  **Random Forest vs Deep Learning:**
    * **Random Forest unggul** dengan R² **0.91** dibandingkan Deep Learning (0.88).
    * Untuk data tabular dengan jumlah sampel kecil (< 1.000 baris), algoritma berbasis Tree (Random Forest) seringkali lebih efisien dan akurat dibandingkan Neural Network yang membutuhkan data masif untuk belajar pola yang kompleks.

---

## 7. CONCLUSION

### 7.1 Kesimpulan Utama
* **Model Terbaik:** **Random Forest Regressor** dipilih sebagai model terbaik karena memiliki error terendah (MAE 1.58) dan akurasi tertinggi (R² 0.91).
* **Pencapaian Goals:** Tujuan proyek tercapai, yaitu membangun model dengan R² > 0.80.

### 7.2 Key Insights
* **Berat Kendaraan (Weight):** Merupakan faktor paling krusial. Pengurangan berat kendaraan akan meningkatkan efisiensi bahan bakar secara signifikan.
* **Evolusi Teknologi:** Fitur `model_year` menunjukkan tren positif, di mana mobil buatan tahun yang lebih baru cenderung lebih irit, terlepas dari spesifikasi mesinnya.

### 7.3 Kontribusi Proyek
Model ini dapat digunakan oleh desainer mobil untuk melakukan simulasi awal efisiensi bahan bakar tanpa harus membuat prototipe fisik, sehingga menghemat biaya R&D.

---

## 8. FUTURE WORK (Saran Pengembangan)

**Data Improvements:**
- [x] Mengumpulkan data kendaraan modern (tahun 2000+) agar model relevan dengan zaman sekarang.
- [ ] Menambah fitur seperti jenis transmisi (Manual/Matic) dan jenis bahan bakar (Bensin/Diesel/Listrik).

**Model Enhancements:**
- [x] Melakukan Hyperparameter Tuning (GridSearchCV) pada Random Forest untuk mencari parameter optimal.
- [ ] Mencoba arsitektur Deep Learning yang lebih dalam jika data bertambah banyak.

**Deployment:**
- [ ] Membuat aplikasi web sederhana menggunakan **Streamlit** agar pengguna awam bisa mencoba prediksi ini.

---

## 9. REPRODUCIBILITY

### 9.1 GitHub Repository
**Link Repository:** [Masukkan URL GitHub Anda]

Repository ini berisi:
* ✅ `uas_datascience.ipynb` (Kode Lengkap)
* ✅ `data/auto-mpg.data` (Dataset)
* ✅ `models/` (File model tersimpan .pkl dan .h5)
* ✅ `requirements.txt` (Daftar library)
* ✅ `README.md` (Dokumentasi)

### 9.2 Environment
* **Python Version:** 3.10+
* **Main Libraries:**
    * `pandas`
    * `numpy`
    * `scikit-learn`
    * `tensorflow` (Keras)
    * `matplotlib` & `seaborn`
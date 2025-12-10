# 📘 ANALISIS PREDIKSI KONSUMSI BAHAN BAKAR (MPG) MENGGUNAKAN BASELINE, MACHINE LEARNING, DAN DEEP LEARNING

## 👤 Informasi
**Nama:** Mohammad Dimas Bahrul Ikhwani  
**NIM:** 234311017  
**Mata Kuliah:** Data Science  
**Tech Stack:** Python, Scikit-Learn, TensorFlow, Pandas  

---

## 1. 🎯 Ringkasan Proyek
Proyek ini bertujuan untuk memprediksi efisiensi bahan bakar kendaraan (MPG - Miles Per Gallon) berdasarkan spesifikasi teknis mobil.

* **Tujuan:** Memprediksi nilai kontinu `mpg` (Regression Task).
* **Pendekatan:** Membangun 3 model perbandingan:
    1.  **Baseline:** Dummy Regressor (Mean Strategy).
    2.  **Machine Learning:** Random Forest Regressor.
    3.  **Deep Learning:** Artificial Neural Network (TensorFlow/Keras).
* **Evaluasi:** Menggunakan metrik MAE (Mean Absolute Error), MSE, dan R² Score.

## 2. 📄 Problem & Goals

### Problem Statements:
* Diperlukan cara untuk memperkirakan konsumsi bahan bakar mobil baru sebelum diproduksi massal berdasarkan spesifikasi mesin.
* Hubungan antara fitur seperti *horsepower*, *weight*, dan *displacement* terhadap irit-borosnya bensin seringkali tidak linear.
* Dataset memiliki *missing values* pada kolom vital (horsepower) yang perlu ditangani agar tidak merusak model.

### Goals:
* Membangun model regresi yang akurat untuk memprediksi MPG.
* Membandingkan performa antara metode statistik sederhana (Baseline), Machine Learning (Random Forest), dan Deep Learning.
* Memberikan pipeline analisis yang lengkap mulai dari *data cleaning* hingga evaluasi model.

## 3. 📁 Struktur Folder

```text
UAS_DataScience_AutoMPG/
│
├── auto-mpg.data           # Dataset Asli (UCI Machine Learning Repository)
├── uas_datascience.ipynb   # Jupyter Notebook (Kode Utama)
├── requirements.txt        # Daftar Library/Dependencies
└── README.md               # Dokumentasi Proyek

4. 📊 Dataset
Sumber: UCI Machine Learning Repository (Auto MPG Data Set).

Jumlah Data: 398 baris.

Tipe: Tabular (Regresi).

Fitur Utama
Dataset ini memiliki 8 kolom, namun fitur car_name dihapus karena berupa ID unik. Berikut 7 fitur yang digunakan:
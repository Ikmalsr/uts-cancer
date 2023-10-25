# Laporan Proyek Machine Learning
### Nama : Ikmal Saepul Rohman 
### Nim : 211351063
### Kelas : Malam B

## Domain Proyek

Pada proyek ini, saya membuat aplikasi untuk menentukan jenis kanker payudara berdasarkan rata rata ukuran sehingga kita bisa menentukan kanker tersebut termasuk dalam golongan ganas atau kanker jinak tanpa harus melakukan test ke rumah sakit

## Business Understanding

Pada dasarnya sel kanker itu terbagi dalam 2 jenis, yakni kanker ganas dan kanker jinak. Hal ini bisa di ketahui berdasarkan rata rata ukuran dan kecenderungan dari sifat kanker tersebut. Maka, adanya aplikasi ini bertujuan untuk menentukan jenis kanker berdasarkan 570 studi kasus dan 30 parimeter dengan algoritma klasifikasi

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Biaya pengecekan sel kanker yang terbilang mahal

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Memudahkan pengguna untuk mengetahui stage kanker payudaranya
- Memungkinkan pengguna untuk melakukan pengecekan mandiri tanpa harus pergi ke rumah sakit


    ### Solution statements
    - Sebagai metode untuk mengetahui klasifikasi kanker melalui algoritma SVC
    - Berdasarkan perhitungan algoritma, tingkat akurasi pada aplikasi machine learning adalah 95.7% sehingga keakurasianya bisa di pertanggung jawabkan

## Data Understanding
Data yang digunakan di dasarkan pada dataset yang di sediakan oleh kaggle dimana di dalamnya terdapat 570 studi kasus dengan 30 parimeter dan 2 klasifikasi

[Cancer Data](https://www.kaggle.com/datasets/erdemtaha/cancer-data).


### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
dari 30 parimeter yang di sediakan, saya hanya mengambil 8 parimeter saja. Dengan 8 parimeter pun tingkat keakurasian nya sudah mencapai 95.7%. Jika di tambakan kembali maka akurasinya juga akan meningkat. Namun parimeter yang saya bawa sudah cukup menunjang karena berdasarkan nilai rata rata nya.
Variable dan tipedata yang di gunakan meliputi :

- radius_mean = float ('Rata Rata Radius Cell Kanker dalam Milimeter (Mm')
- perimeter_mean = float ('Rata Rata Parimeter Cell Kanker (Mm)')
- area_mean = float ('Rata Rata Luas Area Terdampak Cell Kanker (Mm)')
- smoothness_mean = float ('Rata Rata Kerataan Pada Cell Kanker (Mm)')
- compactness_mean = float ('Tingkat kepadatan pada cell Kanker (Mm)')
- concavity_mean = float ('Rata Rata Nilai Kecengungan pada Cell Kanker (Mm)')
- fractal_dimension_mean = float ('Rata Rata Nilai Fractal pada Cell Kanker (Mm)')
- texture_mean = float ('Rata Rata Luas Tekstur area terdampak Cell Kanker (Mm)')


## Data Preparation
Pertama tama kita persiapkan dataset yang akan di pergunakan untuk menjadi model Machine Learning, selanjutnya kita lakukan data preparation dengan memanggil library yang dibutuhkan

```bash
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
```
Selanjutnya kita koneksikan google collab kita dengan kaggle menggunakan token kaggle dengan perintah
```bash
from google.colab import files
files.upload()
```
maka kita akan mengupload file token kaggle kita. dan bisa kita lanjutkan dengan membuat direktori untuk menyimpan file token tersebut
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
selanjutkan kita download data setnya
```bash
!kaggle datasets download -d erdemtaha/cancer-data
```
Jika sudah, kita bisa membuat folder baru untuk menyimpan dan mengekstrak dataset yang sudah kita download
```bash
!mkdir cancer-data
!unzip cancer-data.zip -d cancer-data
!ls cancer-data
```


## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

## Deployment
pada bagian ini anda memberikan link project yang diupload melalui streamlit share. boleh ditambahkan screen shoot halaman webnya.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.


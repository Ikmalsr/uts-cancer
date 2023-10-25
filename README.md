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
Kemudian kita mount data nya dengan perintah
```bash
df = pd.read_csv('cancer-data/Cancer_Data.csv')
```
Jika data sudah di mount, maka kita bisa mencoba memastikan apakah data akan terpanggil atau tidak dengan perintah
```bash
df.head()
```
jika sudah benar maka data 5 teratas akan muncul
Dan jika dilihat ada colom dengan data yang tidak terdefinisi maka kita bisa buang kolom tersebut
```bash
df.drop('Unnamed: 32', axis=1, inplace=True)
```
Maka kolom unnamed: 32 akan terbuang

selanjutnya kita lakukan perhitungan value pada kolom diagnosis dengan perintah

```bash
df['diagnosis'].value_counts()
```
Maka akan muncul
```bash
B    357
M    212
Name: diagnosis, dtype: int64
```
Dimana pada kolom diagnosis terdapat 569 data untuk data B ada 357 entri dan data M 212 entri dengan type data integer
Data B merupakan data Benign atau kanker jinak dan data M merupakan Malignant atau Ganas

Selanjutnya karena model ini hanya bisa menerima inputan berupa angka, maka kita harus rubah dulu variable M dan B dengan perintah
```bash
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
```
Maka jika kita cek value nya akan muncul
```bash
0    357
1    212
Name: diagnosis, dtype: int64
```
dimana data B dan M sudah dirubah menjadi 0 dan 1

Selanjutnya kita bisa lakukan visualisai data dengan perintah
```bash
sns.countplot(x='diagnosis', data=df)
plt.title('Diagnosis Distribution')
plt.show()
```
Maka akan muncul

![alt text](https://github.com/Ikmalsr/uts-cancer/blob/main/diagnosis1.png)
Bisa dilihat berdasarkan visualisai di atas dari studi kasus yang dijalan bisa disimpulkan kebanyakan kanker yang bersifat jinak dibanding yang ganas

selanjutnya kita juga bisa melakukan data visualisasi berupa heatmap dengan perintah

```bash
plt.figure(figsize = (18,9))
sns.heatmap(df.corr(), cmap='GnBu', annot=True)
plt.show()
```
Maka akan muncul
![alt text](https://github.com/Ikmalsr/uts-cancer/blob/main/heatmap.png)
seperti yang dilihat, heatmap berdasarkan data daignosis bernilai 1 dengan parameter yang sudah di koreksi

## Modeling
Selanjutnya jika data preparation sudah selesai maka kita bisa lakukan proses modeling.
Pertama tama yang harus di siapkan adalah nilai X dan Y, diman X menjadi atrribut dan Y menjadi label
```bash
features = ['radius_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','fractal_dimension_mean', 'texture_mean']
X = df[features]
Y = df['diagnosis']
```
Bisa dilihat dimana saya menerapkan 8 attribut pada nilai X dan 1 label pada Y

Selanjutnya kita lakukan scaler karena menggunakan algoritma SVC
```bash
scaler = StandardScaler()
```
Kemudian kita bisa standarkan nilai X
```bash
scaler.fit(X)
```

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


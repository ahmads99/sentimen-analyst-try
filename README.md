# Sentiment Analysis on OVO App Reviews

### Deskripsi
Project ini bertujuan untuk melakukan analisis sentimen terhadap data ulasan aplikasi OVO menggunakan model deep learning. Dengan menggunakan berbagai model seperti LSTM, CNN, dan GRU, proyek ini bertujuan untuk mengklasifikasikan sentimen ulasan (positif, negatif, atau netral) dari pengguna aplikasi OVO. Data yang digunakan adalah ulasan yang telah diproses melalui beberapa tahap preprocessing teks untuk memastikan kualitas data yang optimal.

### Fitur Proyek
- **Preprocessing Data**: Melakukan pembersihan teks dengan menghapus stopwords, tokenisasi, dan lemmatization.
- **Model Machine Learning**: Membangun dan melatih tiga model deep learning (LSTM, CNN, GRU) untuk analisis sentimen.
- **Evaluasi Model**: Menggunakan metrik akurasi untuk menilai performa model pada data latih dan data uji.
- **Optimasi Model**: Melakukan eksperimen dengan berbagai model dan parameter untuk memperoleh hasil terbaik.
- **Visualisasi Hasil**: Menyediakan insight tentang distribusi sentimen dalam data menggunakan visualisasi.

### Struktur Proyek
1. **Data Preprocessing**: Membersihkan data teks dengan teknik seperti stopword removal, tokenisasi, dan lemmatization.
2. **Model Training**:
   - Model LSTM: Menerapkan LSTM untuk menganalisis urutan teks dan menangkap dependensi temporal dalam data.
   - Model CNN: Menerapkan CNN untuk mengekstraksi fitur spasial dari data teks.
   - Model GRU: Menggunakan GRU untuk memanfaatkan informasi urutan secara efisien dengan lebih sedikit parameter.
3. **Evaluasi dan Hasil**:
   - Evaluasi akurasi setiap model pada data latih dan data uji.
   - Perbandingan performa model LSTM, CNN, dan GRU.
4. **Insight**: Melihat hasil analisis sentimen yang dapat digunakan untuk memahami persepsi pengguna terhadap aplikasi OVO.

### Hasil
| Model  | Akurasi Latih | Akurasi Uji  |
|--------|---------------|--------------|
| LSTM   | 92.96%        | 92.17%       |
| CNN    | 91.69%        | 92.13%       |
| GRU    | 90.88%        | 92.00%       |

**Kesimpulan**: Model LSTM menunjukkan performa terbaik dengan akurasi yang lebih tinggi pada data latih dan data uji dibandingkan dengan model CNN dan GRU. Namun, semua model menunjukkan akurasi yang sangat baik dalam menganalisis sentimen ulasan aplikasi OVO.

### Instalasi
1. **Clone Repository**:
   ```bash
   git clone https://github.com/username/repository_name.git
   cd repository_name

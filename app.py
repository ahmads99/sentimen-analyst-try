import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Memuat model yang sudah disimpan
model = load_model('model/model_LSTM.h5')

# Memuat tokenizer dan labelencoder yang sudah disimpan
tokenizer = pickle.load(open('model/tokenizer2.pkl', 'rb'))
labelencoder = pickle.load(open('model/labelencoder2.pkl', 'rb'))

# Layout untuk aplikasi Streamlit
st.set_page_config(page_title="Sentiment Analysis OVO App", layout="wide")

# Judul Aplikasi
st.title("Sentiment Analysis Aplikasi OVO")

# Subtitle
st.subheader("Prediksi Sentimen Review Aplikasi OVO (Positive, Negative, Neutral)")

# Penjelasan
st.markdown("""
    Aplikasi ini menggunakan model **Deep Learning** untuk menganalisis sentimen dari teks **review aplikasi OVO**.
    Silakan masukkan teks dari review aplikasi OVO yang ingin dianalisis dan aplikasi ini akan memberikan prediksi
    apakah teks tersebut **positif**, **negatif**, atau **netral**.
""", unsafe_allow_html=True)

# Input teks dari pengguna
user_input = st.text_area("Masukkan Teks Review Aplikasi OVO untuk Analisis Sentimen", "", height=200)

# Fungsi untuk preprocess teks
def preprocess_input(text):
    # Tokenisasi dan padding
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=2500)  # Sesuaikan dengan panjang input yang kamu tentukan
    return padded

# Fungsi untuk memprediksi sentimen
def predict_sentiment(text):
    # Preprocessing input teks
    processed_input = preprocess_input(text)
    # Prediksi dengan model
    pred = model.predict(processed_input)
    # Menentukan label sentimen
    sentiment = np.argmax(pred)
    labels = ['Negative', 'Neutral', 'Positive']  # Sesuaikan dengan urutan label yang kamu gunakan
    return labels[sentiment], pred

# Tombol pertama untuk analisis dengan key unik
if st.button('Analisis Sentimen', key='analyze_button_1'):
    if user_input:
        sentiment, prediction = predict_sentiment(user_input)
        st.subheader(f"Sentimen: **{sentiment}**", anchor="sentiment")
        st.write(f"Probabilitas Prediksi:")
        st.write(f"Negatif: {prediction[0][0]:.2f}")
        st.write(f"Netral: {prediction[0][1]:.2f}")
        st.write(f"Positif: {prediction[0][2]:.2f}")
    else:
        st.warning("Silakan masukkan teks review aplikasi OVO terlebih dahulu!")

# # Tombol kedua untuk analisis dengan key unik (berbeda dengan yang pertama)
# if st.button('Analisis Sentimen dengan Desain Modern', key='analyze_button_2'):
#     if user_input:
#         sentiment, prediction = predict_sentiment(user_input)
#         st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
#         st.markdown(f"<p class='sentiment-result'>Sentimen: **{sentiment}**</p>", unsafe_allow_html=True)
#         st.markdown(f"<p>Negatif: {prediction[0][0]:.2f}</p>", unsafe_allow_html=True)
#         st.markdown(f"<p>Netral: {prediction[0][1]:.2f}</p>", unsafe_allow_html=True)
#         st.markdown(f"<p>Positif: {prediction[0][2]:.2f}</p>", unsafe_allow_html=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#     else:
#         st.warning("Silakan masukkan teks review aplikasi OVO terlebih dahulu!")

# Styling UI
st.markdown("""
    <style>
        /* Background and General Layout */
        body {
            background-color: #F3F4F6;
            color: #333;
        }
        .stTextArea {
            font-size: 18px;
            border: 2px solid #3B82F6;
            border-radius: 12px;
            padding: 12px;
            font-family: 'Roboto', sans-serif;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stButton {
            background-color: #10B981; /* Warna dasar tombol tetap hijau */
            color: #FFFFFF; /* Warna teks tetap putih untuk kontras */
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 20px; /* Mengatur padding kiri dan kanan agar tombol tidak terlalu panjang */
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); /* Shadow yang cukup untuk memberi efek kedalaman */
            transition: all 0.3s ease;
            border: 2px solid #0F9C6E; /* Border yang lebih gelap untuk memberikan kedalaman */
            cursor: pointer; /* Kursor pointer untuk menunjukkan tombol bisa diklik */
            width: auto; /* Membiarkan lebar tombol otomatis berdasarkan konten */
        }

        .stButton:hover {
            background-color: #12C793; /* Sedikit cerahkan warna background saat hover */
            box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.3); /* Efek shadow lebih tajam saat hover */
        }




        /* Title and Subheading */
        h1 {
            font-family: 'Roboto', sans-serif;
            font-weight: 700;
            color: #1D4ED8;
        }
        h2 {
            font-family: 'Roboto', sans-serif;
            font-weight: 600;
            color: #374151;
        }

        /* Text style */
        .stMarkdown {
            font-family: 'Roboto', sans-serif;
            font-size: 16px;
            line-height: 1.5;
            color: #4B5563;
        }

        /* Sentiment result style */
        .sentiment-result {
            font-size: 24px;
            font-weight: bold;
            color: #1D4ED8;
        }

        /* Card layout for result */
        .result-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .result-card p {
            font-size: 18px;
            font-family: 'Roboto', sans-serif;
            margin-bottom: 12px;
        }
    </style>
""", unsafe_allow_html=True)

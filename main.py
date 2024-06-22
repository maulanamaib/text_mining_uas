import pandas as pd
import streamlit as st
import nltk
import re
import torch
import numpy as np

from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

label_encoder = LabelEncoder()

tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = BertModel.from_pretrained("indobenchmark/indobert-base-p1")

def bersihkan_teks(teks):
    teks_bersih = re.sub(r'http\S+', '', teks)
    teks_bersih = re.sub(r'https\S+', '', teks_bersih)
    teks_bersih = re.sub(r'Simak breaking news berita pilihan langsung ponselmu. Pilih saluran andalanmu akses berita Kompas\.com WhatsApp Channel Pastikan install aplikasi WhatsApp ya\.', '', teks_bersih)
    return teks_bersih
    
def summarize_text(text):
    # Tokenisasi kalimat
    sentences = nltk.sent_tokenize(text)

    # Tokenisasi dan encoding setiap kalimat
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)

    # Mendapatkan vektor fitur dari lapisan terakhir
    sentence_embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    # Menghitung kesamaan kosinus antar kalimat
    similarity_matrix = cosine_similarity(sentence_embeddings)

    # Menggunakan metode pagerank untuk menentukan peringkat kalimat
    import networkx as nx

    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    # Memilih kalimat dengan skor tertinggi
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = " ".join([ranked_sentences[i][1] for i in range(2)])  # Mengambil 2 kalimat teratas

    return summary

data = pd.read_excel("KompasCrawlData.xlsx")
df = pd.DataFrame(data)
st.write(df)
# Buat input text field untuk kata kunci pencarian
# keyword = st.text_input('Masukkan kata kunci pencarian:')

# # Filter data berdasarkan kata kunci
# filtered_data = df[df['Konten'].str.contains(keyword, na=False)]
# cari = st.button("Cari")
# if cari:
#     st.dataframe(filtered_data)

idx = st.selectbox('Select rows:', df.index)
# selected_rows = filtered_data.loc[selected_indices]
pilih = st.button("Pilih")
if pilih:
    # st.write(selected_rows)

    df1 = df[df['Konten'].str.contains('KOMPAS.com', na=False)].iloc[[idx]]
    df1['Konten'] = df1['Konten'].str.replace(r'[^A-Za-z0-9\s.,!?]', '', regex=True)
    df1['Konten'] = df1['Konten'].str.split('KOMPAS.com').str[1].str.strip()
    #Menghitung Jumlah Kalimat dalam Setiap Entri
    df1['jumlah_kalimat'] = df1['Konten'].apply(lambda x: len(nltk.sent_tokenize(x)))
    # Menjatuhkan baris dengan jumlah kalimat kurang dari 3
    df1 = df1[df1['jumlah_kalimat'] >= 3]
    # Menghapus kolom 'jumlah_kalimat' karena sudah tidak diperlukan
    df1 = df1.drop(columns=['jumlah_kalimat'])
    # Membersihkan kolom 'Konten' menggunakan fungsi bersihkan_teks
    df1['Konten'] = df1['Konten'].apply(bersihkan_teks)
    # Menghapus stopword
    stop_words = set(stopwords.words('indonesian'))  # Gantilah 'indonesian' sesuai dengan bahasa yang digunakan
    df1['Konten'] = df1['Konten'].apply(lambda x: ' '.join(word for word in x.split() if word.lower() not in stop_words))
    label_encoder = LabelEncoder()
    df1['Kelas'] = label_encoder.fit_transform(df1['Kelas'])
    df1['Ringkasan'] = df1['Konten'].apply(summarize_text)

    st.write(df1["Ringkasan"])


    # selected_rows = selected_rows['Konten'].str.replace(r'[^A-Za-z0-9\s.,!?]', '', regex=True)
    # selected_rows = selected_rows['Konten'].str.split('KOMPAS.com').str[1].str.strip()
    # selected_rows['jumlah_kalimat'] = selected_rows['Konten'].apply(lambda x: len(nltk.sent_tokenize(x)))
    # selected_rows = selected_rows[selected_rows['jumlah_kalimat'] >= 3]
    # selected_rows = selected_rows.drop(columns=['jumlah_kalimat'])
    # selected_rows['Konten'] = selected_rows['Konten'].apply(bersihkan_teks)
    # stop_words = set(stopwords.words('indonesian'))  # Gantilah 'indonesian' sesuai dengan bahasa yang digunakan
    # selected_rows['Konten'] = selected_rows['Konten'].apply(lambda x: ' '.join(word for word in x.split() if word.lower() not in stop_words))
    # selected_rows['Kelas'] = label_encoder.fit_transform(selected_rows['Kelas'])

    # # Indobert
    # selected_rows['Ringkasan'] = selected_rows['Konten'].apply(summarize_text)

    # st.write(selected_rows["Ringkasan"])
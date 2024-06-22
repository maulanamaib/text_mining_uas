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
    teks_bersih = re.sub(r'[^A-Za-z0-9\s.,!?]', '', teks_bersih)
    return teks_bersih

def summarize_text(text):
    sentences = nltk.sent_tokenize(text)
    
    # Tokenisasi dan encoding setiap kalimat dalam batch
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
    summary = " ".join([ranked_sentences[i][1] for i in range(min(2, len(ranked_sentences)))])  # Mengambil 2 kalimat teratas

    return summary

data = pd.read_excel("KompasCrawlData.xlsx")
df = pd.DataFrame(data)
st.write(df)

idx = st.selectbox('Select rows:', df.index)
pilih = st.button("Pilih")
if pilih:
    df1 = df[df['Konten'].str.contains('KOMPAS.com', na=False)].iloc[[idx]]
    df1['Konten'] = df1['Konten'].apply(bersihkan_teks)

    #Menghitung Jumlah Kalimat dalam Setiap Entri
    df1['jumlah_kalimat'] = df1['Konten'].apply(lambda x: len(nltk.sent_tokenize(x)))
    df1 = df1[df1['jumlah_kalimat'] >= 3]
    df1 = df1.drop(columns=['jumlah_kalimat'])

    # Membersihkan stopwords
    stop_words = set(stopwords.words('indonesian'))
    df1['Konten'] = df1['Konten'].apply(lambda x: ' '.join(word for word in x.split() if word.lower() not in stop_words))
    
    label_encoder = LabelEncoder()
    df1['Kelas'] = label_encoder.fit_transform(df1['Kelas'])
    df1['Ringkasan'] = df1['Konten'].apply(summarize_text)

    st.write(df1["Ringkasan"])

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

data = pd.read_excel("bersih.xlsx")
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

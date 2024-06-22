import pandas as pd
import streamlit as st
import nltk
import re
import torch
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

st.markdown("<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Clasiffication text kompas.com</h1>", unsafe_allow_html=True)
data = pd.read_excel("bersih.xlsx")
df = pd.DataFrame(data)
data


# Pisahkan fitur dan label

Class='''
    #Class
    X = data['Ringkasan']
    y = data['Kelas']
    X
    y
    '''
Class1='''
    #Class
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)']
    X_tfidf
    '''
Class2='''
    #Class
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X_tfidf, y, df3.index, test_size=0.2, random_state=42)
    '''
Class3='''
    #Class
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Prediksi pada data uji
    y_pred = model.predict(X_test)
    y_pred
    '''
Class4='''
    #Class
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=[
    'Bola', 'Edu', 'Food', 'Health', 'Lestari', 'Otomotif', 'Travel']))
    '''
Class5='''
    #Class
    result_df = pd.DataFrame({
    'Ringkasan': df3.loc[idx_test, 'Ringkasan'].values,
    'Kelas Asli': y_test,
    'Prediksi Kelas': y_pred
    })
    
    result_df
    '''
st.code(Class, language='python')
X = data['Ringkasan']
y = data['Kelas']
X
y

st.write(X)

st.write('Tokenisasi dan Feature Weighting menggunakan TF-IDF')
st.code(Class1, language='python')
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)
X_tfidf
st.code(Class2, language='python')
st.write('Bagi data menjadi data latih dan uji')
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X_tfidf, y, data.index, test_size=0.2, random_state=42)

st.code(Class3, language='python')
# Inisialisasi dan latih model Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)
# Prediksi pada data uji
y_pred = model.predict(X_test)
y_pred
# Evaluasi model
st.code(Class4, language='python')
"Classification Report:\n", classification_report(y_test, y_pred, target_names=[
    'Bola', 'Edu', 'Food', 'Health', 'Lestari', 'Otomotif', 'Travel'
])

st.code(Class5, language='python')
# Buat tabel baru berisi kelas asli dan prediksi kelas
result_df = pd.DataFrame({
    'Ringkasan': data.loc[idx_test, 'Ringkasan'].values,
    'Kelas Asli': y_test,
    'Prediksi Kelas': y_pred
})

result_df


# idx = st.selectbox('Select rows:', df.index)
# pilih = st.button("Pilih")
# if pilih:
#     df1 = df[df['Konten'].str.contains('KOMPAS.com', na=False)].iloc[[idx]]
#     df1['Konten'] = df1['Konten'].apply(bersihkan_teks)

#     #Menghitung Jumlah Kalimat dalam Setiap Entri
#     df1['jumlah_kalimat'] = df1['Konten'].apply(lambda x: len(nltk.sent_tokenize(x)))
#     df1 = df1[df1['jumlah_kalimat'] >= 3]
#     df1 = df1.drop(columns=['jumlah_kalimat'])

#     # Membersihkan stopwords
#     stop_words = set(stopwords.words('indonesian'))
#     df1['Konten'] = df1['Konten'].apply(lambda x: ' '.join(word for word in x.split() if word.lower() not in stop_words))
    
#     label_encoder = LabelEncoder()
#     df1['Kelas'] = label_encoder.fit_transform(df1['Kelas'])
#     df1['Ringkasan'] = df1['Konten'].apply(summarize_text)

#     st.write(df1["Ringkasan"])

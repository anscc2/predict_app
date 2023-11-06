import streamlit as st
import predict

st.title("Predicting Bully Tweet")

st.header('App ini dibuat untuk memprediksi apakah sebuah tweet terindikasi sebagai bully atau tidak')

text = st.text_area("Masukkan text")
output = ""
if st.button("Predict"):
    output = predict.predict_tweet(text)
st.success(f"Hasil Prediksi {output}")
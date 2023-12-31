import re
import pandas as pd
import torch
import torch.nn as nn
import requests
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from keras.preprocessing.sequence import pad_sequences
from sru import SRU
import streamlit as st
import predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
url_word_index = 'https://raw.githubusercontent.com/anscc2/predict_app/main/data/word_index.json'
response = requests.get(url_word_index)
word_index = response.json()

def cleaning(text):
  text = text.lower()
  text = re.sub(r'(?:@[\w_]+)', ' ', text)
  text = re.sub('#[A-Za-z0-9_]+', " ", text)
  text = re.sub('RT', ' ', text)
  text = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', ' ', text)
  text = re.sub(r'[^a-zA-Z\s]', ' ', text)
  return ' '.join(text.split())

def normalize_slang(text):
  slang_dict = pd.read_csv('https://raw.githubusercontent.com/anscc2/predict_app/main/data/new_kamusalay.csv', encoding='latin-1', header=None)
  slang_dict = slang_dict.rename(columns={0: 'original', 1: 'replacement'})
  slang_dict_map = dict(zip(slang_dict['original'], slang_dict['replacement']))
  return ' '.join([slang_dict_map[word] if word in slang_dict_map else word for word in text.split(' ')])

def remove_stopwords(text):
  url_stop = 'https://raw.githubusercontent.com/anscc2/predict_app/main/data/stopword-list.json'
  response = requests.get(url_stop)
  stopwords_indo = response.json()
  return ' '.join([word for word in text.split(' ') if word not in stopwords_indo])

def stemming(text):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  return stemmer.stem(text)

def preprocess(text):
  text = cleaning(text)
  text = normalize_slang(text)
  text = remove_stopwords(text)
  text = stemming(text)
  return text

class SRUModel(nn.Module):
  def __init__(self, pretrained_embedding):
    super(SRUModel, self).__init__()
    self.vocab_size, self.embed_dim = pretrained_embedding.shape
    self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=True)

    self.dropout = nn.Dropout(p=0.3)

    self.sru = SRU(input_size=self.embed_dim, hidden_size=128, num_layers=1)
    self.avg_pooling = nn.AdaptiveAvgPool1d(1)
    self.max_pooling = nn.AdaptiveMaxPool1d(1)
    self.fc = nn.Linear(256, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    embedded = self.embedding(x).float()
    embedded = self.dropout(embedded)
    out, _ = self.sru(embedded)
    out = self.dropout(out)
    avg_pooled = self.avg_pooling(out.permute(0, 2, 1)).view(out.size(0), -1)
    max_pooled = self.max_pooling(out.permute(0, 2, 1)).view(out.size(0), -1)
    pooled = torch.cat([avg_pooled, max_pooled], dim=1)
    output = self.fc(pooled)
    return self.sigmoid(output)


def predict_tweet(text):
  preprocessing_text = preprocess(text)
  token = preprocessing_text.split()
  encoded = [word_index.get(word, word_index['<OOV>']) for word in token]
  padded_text = pad_sequences([encoded], maxlen=52, padding='post')
  input_tensor = torch.tensor(padded_text).to(device)
  embedding_matrix = torch.load('embeddings_matrix.pth', map_location=device)

  model = SRUModel(pretrained_embedding=embedding_matrix).to(device)

  model.load_state_dict(torch.load('modelsru-fold-2.pth', map_location=device))
  model.eval()
  with torch.no_grad():
    predictions = model(input_tensor)

  predicted = (predictions > 0.5).float()
  result = 'Bully' if predicted.item() == 1 else 'Non Bully'
  return result, predictions.item()


st.title("Predicting Bully Tweet")
st.header('App ini dibuat untuk memprediksi apakah sebuah tweet terindikasi sebagai bully atau tidak')

form = st.form(key='predicttext-form')
user_input = form.text_area("Enter your text")
submit = form.form_submit_button('Predict')
if submit:
  result, score = predict_tweet(user_input)
  if result == 'Bully':
    st.error(f"This tweet is categorized as {result} (score: {score})")
  else:
    st.success(f"This tweet is categorized as {result} (score: {score})")

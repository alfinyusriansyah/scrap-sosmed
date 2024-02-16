import streamlit as st
import pandas as pd
from joblib import load
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ntscraper import Nitter
import nltk
import requests
import os
from PIL import Image
import numpy as np
import plotly.express as px
from scrap_tweet import count_common_words, take_dataframe, preprocess_text
from scrap_tweet import get_predict, scrap_tweet, convert, label_meanings, plot_wordcloud

# nltk.download('stopwords')
# nltk.download('punkt')
# stop_words = set(stopwords.words('indonesian'))
# factory = StemmerFactory()
# stemmer = factory.create_stemmer()

def main():
    #------------------------------------ get tweet --------------------------------------------------
    if 'data_tweet' not in st.session_state:
        st.session_state.data_tweet = None

    if st.session_state.data_tweet is None:
        data_tweet = scrap_tweet(username)
        prepro_text = take_dataframe(data_tweet)
        prepro_text['text'] = prepro_text['text'].apply(preprocess_text)
        st.write("Result scrap :")
        st.dataframe(prepro_text)
        
        #---------------------------------------- get predict ----------------------------------------------
        predict_tweet = get_predict(prepro_text)
        st.write("hasil predicted :")
        st.dataframe(predict_tweet)
        st.session_state.data_tweet = predict_tweet
    else:
        predict_tweet = st.session_state.data_tweet

    analyze(predict_tweet)

def analyze(predict_tweet):
        #--------------------------------------- visualisasi label -----------------------------------------------
    value_counts_result = predict_tweet['meaning'].value_counts().sort_index()
    df_value_counts = pd.DataFrame({'meaning': value_counts_result.index, 'count': value_counts_result.values})

    # plot untuk count all meaning in tweet
    fig = px.bar(df_value_counts, x='meaning', y='count', title='Frekuensi Kategori', 
             labels={'meaning': 'Kategori', 'count': 'Frekuensi'}, color='meaning')
    st.plotly_chart(fig)

    # plot untuk common word in tweet
    top_words_df = count_common_words(predict_tweet['text'])
    fig = px.bar(top_words_df, x='count', y='Common_words', title='Most Common Words', orientation='h', 
                width=700, height=700, color='Common_words')
    st.plotly_chart(fig)
    
    # plot common word by meaning tweet
    selected_meaning = st.selectbox("Pilih Kategori Meaning", list(label_meanings.values()))
    filtered_data_tweet = predict_tweet[predict_tweet['meaning'] == selected_meaning]
    st.dataframe(filtered_data_tweet)
    top_words_df2 = count_common_words(filtered_data_tweet['text'])

    # Ubah struktur data menjadi sesuai dengan treemap
    top_words_df2['path'] = 'Top Words' 
    top_words_df2['parent'] = ''
    top_words_df2.rename(columns={'Common_words': 'label', 'count': 'value'}, inplace=True)

    # Buat treemap
    fig = px.treemap(top_words_df2, path=['path', 'label'], values='value', title=f'Tree Of Unique {selected_meaning} Words')
    st.plotly_chart(fig)

    # word cloud
    pos_mask = np.array(Image.open('img/twitter_mask.png'))
    plot_wordcloud(filtered_data_tweet.text,mask=pos_mask,color='white',max_font_size=100,title_size=30,title=f"WordCloud of {selected_meaning} Tweets")


st.header("SapuJagad 99")
st.subheader("scraping and predicted tweet from ITE Law")
elements = st.container()
username = convert(elements.text_input("Username twitter :") )

if "clicked" not in st.session_state:
    st.session_state.clicked = False
if st.button("scrap") or st.session_state["clicked"]:
    st.session_state["clicked"] = True
    main()

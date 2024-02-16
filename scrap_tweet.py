from apify_client import ApifyClient
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from joblib import load
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
import pandas as pd
import requests
import re
import io
import string
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


stopwords = set(STOPWORDS)


# Load environment variables from .env file
load_dotenv()

# Access environment variables
api_key = os.environ['API_KEY']

# Initialize the ApifyClient with your API token
client = ApifyClient(api_key)

def convert(string):
    li = list(string.split(" "))
    return li

label_meanings = {
    0: 'Neutral',
    1: 'Positive',
    2: 'Negative',
    3: 'Insulting the Government',
    4: 'Insulting or Defaming Others',
    5: 'Threatening Others',
    6: 'Alluding to the Tribe, Religion, Race, and Intergroup'
}

def scrap_tweet(username):
    # Prepare the Actor input
    run_input = {
        "customMapFunction": "(object) => { return {...object} }",
        "maxItems": 20,
        "onlyImage": False,
        "onlyQuote": False,
        "onlyTwitterBlue": False,
        "onlyVerifiedUsers": False,
        "onlyVideo": False,
        "sort": "Top",
        "dateFrom" : 2024-2-5,
        "dateTo" : 2024-2-10,
        "twitterHandles": username
    }

    # Run the Actor and wait for it to finish
    run = client.actor("apidojo/tweet-scraper")\
            .call(run_input=run_input, timeout_secs=6000, memory_mbytes=256, build="latest")

    items_list = []

    # Fetch and print Actor results from the run's dataset (if there are any)
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        items_list.append(item)
        
    print("Total Data Scraping: "+ str(len(items_list)))
    return items_list

def take_dataframe(data):
    df_tweet = []
    for i in data:
        data = [i['text'],
                i['createdAt']]
        df_tweet.append(data)

    df_tweet=pd.DataFrame(df_tweet, columns = ['text','createdAt'])
    return df_tweet

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    text = text.strip()
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in custom_stopwords]
    processed_text = ' '.join(words)
    return processed_text

def get_predict(filter_data):
    with open('model/best_rf_model.pkl', 'rb') as model_file:
        best_rf_model = pickle.load(model_file)

    with open('model/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # TF-IDF Vectorization pada data yang ingin diprediksi
    data_tfidf = vectorizer.transform(filter_data['text'])
    predictions = best_rf_model.predict(data_tfidf)
    netral_class_probabilities = best_rf_model.predict_proba(data_tfidf)[:, 0]

    # Add the predicted probabilities to the DataFrame
    filter_data['predicted_prob_netral'] = netral_class_probabilities

    # Tambahkan kolom hasil prediksi ke dalam DataFrame
    filter_data['predicted_sentiment'] = predictions
    filter_data['meaning'] = filter_data['predicted_sentiment'].map(label_meanings)
    return filter_data

def load_custom_stopwords(csv_file):
    custom_stopwords = set()
    stopword_df = pd.read_csv(csv_file)
    for index, row in stopword_df.iterrows():
        custom_stopwords.update(row)
    return custom_stopwords

def count_common_words(text, n=20):
    all_text = ' '.join(text)
    words = word_tokenize(all_text)
    words = [word for word in words if word.lower() not in custom_stopwords]
    word_counts = Counter(words)
    top_words = word_counts.most_common(n)
    top_words_df = pd.DataFrame(top_words, columns=['Common_words', 'count'])
    return top_words_df

# Load custom stopwords from CSV
custom_stopwords = load_custom_stopwords("stopwordbahasaV2.csv")

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), color='white',
                   title=None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'u', "im"}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color=color,
                          stopwords=stopwords,
                          max_words=max_words,
                          max_font_size=max_font_size,
                          random_state=42,
                          width=400,
                          height=200,
                          mask=mask)
    wordcloud.generate(str(text))

    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask)
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
        plt.title(title, fontdict={'size': title_size, 'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud)
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 'verticalalignment': 'bottom'})
    plt.axis('off')
    plt.tight_layout()

    # Simpan gambar ke buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Tampilkan gambar di Streamlit
    st.image(buf, caption=title)
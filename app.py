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
import pandas as pd
import requests
import os

# nltk.download('stopwords')
# nltk.download('punkt')
# stop_words = set(stopwords.words('indonesian'))
# factory = StemmerFactory()
# stemmer = factory.create_stemmer()

if 'final_tweets' not in st.session_state:
    st.session_state.final_tweets = []
    st.session_state.final_tweet2 = []

# final_tweets = []
data_profile = []


# Create a dictionary mapping predicted labels to their meanings
label_meanings = {
    0: 'Neutral',
    1: 'Positive',
    2: 'Negative',
    3: 'Insulting the Government',
    4: 'Insulting or Defaming Others',
    5: 'Threatening Others',
    6: 'Alluding to the Tribe, Religion, Race, and Intergroup'
}

def preprocess_text(text):
    # Mengubah kalimat menjadi huruf kecil
    text = text.lower()

    # Menghapus tanda baca dari kalimat
    text = re.sub(r'[^\w\s]', '', text)

    # Menghapus spasi di awal dan akhir kalimat
    text = text.strip()
    return text

def get_profile(username):
    get_profile = scraper.get_profile_info(username)
    # Check if get_profile is not None before accessing its keys
    if get_profile is not None:
        data_profile.append([
            get_profile.get('image', ''),
            get_profile.get('name', ''),
            get_profile.get('username', ''),
            get_profile.get('location', ''),
            get_profile['stats'].get('following', ''),
            get_profile['stats'].get('followers', '')
        ])
 
        # Download the image
        image_url = get_profile.get('image', '')
 
        if image_url:
            response = requests.get(image_url, stream=True)
            response.raise_for_status() 
 
            # Replace this with the actual path where you want to save the images
            output_directory = 'D:/ngoding/scrap-detail-sosmed/img'
 
            # Ensure the output directory exists
            os.makedirs(output_directory, exist_ok=True)
 
            # Create a suitable filename based on the username
            filename = f"{username}_profile_image.jpg"
 
            # Save the image to the output directory
            with open(os.path.join(output_directory, filename), 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
 
            st.write(f"Downloaded {filename}")
        else:
            st.write(f"No image found for {username}")
    else:
        # Handle the case where get_profile is None
        st.write(f"Unable to fetch profile information for {username}")
 
    data_profile_ini = pd.DataFrame(data_profile, columns=['image', 'name', 'username', 'location', 'following', 'followers'])
    return data_profile_ini

def get_tweet(username):
    tweet = scraper.get_tweets(username, mode= "user", number=10)
    for tweet in tweet['tweets']:
        data = [tweet['text']]
        st.session_state.final_tweets.append(data)

    # df=pd.read_csv('anies_tweet2.csv')
    data_tweet = pd.DataFrame(st.session_state.final_tweets, columns = ['text'])
    data_tweet=data_tweet.dropna()
    print(data_tweet.info())
    return data_tweet

@st.cache(allow_output_mutation=True)
def get_predict(data_tweet):
    with open('best_rf_model.pkl', 'rb') as model_file:
        best_rf_model = pickle.load(model_file)

    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # TF-IDF Vectorization pada data yang ingin diprediksi
    data_tfidf = vectorizer.transform(data_tweet['text'])
    predictions = best_rf_model.predict(data_tfidf)
    netral_class_probabilities = best_rf_model.predict_proba(data_tfidf)[:, 0]

    # Add the predicted probabilities to the DataFrame
    data_tweet['predicted_prob_netral'] = netral_class_probabilities

    # Tambahkan kolom hasil prediksi ke dalam DataFrame
    data_tweet['predicted_sentiment'] = predictions
    data_tweet['meaning'] = data_tweet['predicted_sentiment'].map(label_meanings)
    return data_tweet

st.header("SapuJagad 99")
st.subheader("scraping tweet and predicted from ITE Law")
elements = st.container()
username = elements.text_input("Username twitter :") 
if st.button("scrap") :
    st.write(username)
    
    scraper = Nitter()
    #-------------------------------------------- get profile -------------------------------------------------
    # df_profile = pd.read_csv('profile.csv')
    st.subheader("Profile :")
    data_profile_ini = get_profile(username)
    name_values = data_profile_ini['name'].astype(str).tolist()
    name_loc = data_profile_ini['location'].tolist()
    name_following = data_profile_ini['following'].astype(str).tolist()
    name_followers = data_profile_ini['followers'].astype(str).tolist()

    st.image(f"img/{username}_profile_image.jpg")
    st.text("Name : " + ", ".join(name_values))
    st.text("Location : " + ", ".join(name_loc))
    st.text("Following : " + ", ".join(name_following))
    st.text("Followers : " + ", ".join(name_followers))

    #------------------------------------ get tweet --------------------------------------------------
    data_tweet = get_tweet(username)
    data_tweet['text'] = data_tweet['text'].apply(preprocess_text)
    st.write("Result scrap :")
    st.dataframe(data_tweet)

    #---------------------------------------- get predict ----------------------------------------------
    data_tweet = get_predict(data_tweet)
    st.write("hasil predicted :")
    st.dataframe(data_tweet)

    #--------------------------------------- visualisasi -----------------------------------------------
    # Hitung frekuensi untuk setiap kategori "meaning"
    value_counts_result = data_tweet['meaning'].value_counts().sort_index()

    # Membuat visualisasi dengan Seaborn
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='meaning', data=data_tweet, palette='viridis', order=value_counts_result.index, ax=ax)

    # Menambahkan nilai frekuensi di atas setiap batang histogram
    for i, value in enumerate(value_counts_result):
        ax.text(i, value + 0.1, str(value), ha='center', va='bottom')

    ax.set_title('Frekuensi Kategori')
    ax.set_xlabel('Kategori')
    ax.set_ylabel('Frekuensi')

    # Menampilkan grafik menggunakan st.pyplot(fig)
    st.pyplot(fig)

    # Dropdown untuk memilih kategori "meaning"
    selected_meaning = st.selectbox("Pilih Kategori Meaning", list(label_meanings.values()))
    
    st.session_state.final_tweet2 = data_tweet
    # Filter DataFrame berdasarkan kategori "meaning"
    filtered_data_tweet = st.session_state.final_tweet2[st.session_state.final_tweet2['meaning'] == selected_meaning]

    # Menampilkan DataFrame hasil filter
    st.write(f"Tweet untuk kategori {selected_meaning}:")
    st.dataframe(filtered_data_tweet)



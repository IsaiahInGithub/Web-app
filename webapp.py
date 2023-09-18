import streamlit as st
import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import base64
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Page config
st.set_page_config(page_title='Text Analysis App')

# Title
st.title('Text Analysis App')

# About section
st.sidebar.subheader('About')
st.sidebar.markdown("""
    This is a Text Analysis Web App that performs various text analysis tasks on your data.
    You can upload a CSV file, select a column, and analyze the text within it.
    The following tasks are available:

    - **Sentiment Analysis**: Determine the sentiment of the text.
    - **Word Cloud**: Visualize the most frequent words in the text.
    - **Part-of-Speech Tagging**: Identify adjectives, verbs, proper nouns, and common nouns.
    - **Clustering**: Perform K-Means clustering on the text data to find common themes.

    You can also download the clustered data as a CSV file.
    """)

# File upload
uploaded_file = st.file_uploader('Choose a CSV file', type=['csv'])

if uploaded_file:
    # Load dataframe
    df = pd.read_csv(uploaded_file)

    # Column selection
    column = st.selectbox('Select column', df.columns)

    # Initialize analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Initialize Spacy for part-of-speech tagging
    nlp = spacy.load("en_core_web_sm")

    # Text preprocessing
    def preprocess_text(text):
        try:
            # Tokenize and remove stopwords
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
            return ' '.join(tokens)
        except Exception as e:
            return text

    df[column] = df[column].apply(preprocess_text)

    # Sentiment analysis and display
    st.subheader('Sentiment Analysis')
    sentiment_scores = df[column].apply(lambda x: analyzer.polarity_scores(x))
    df['Sentiment'] = sentiment_scores.apply(lambda x: 'Highly Positive' if x['compound'] >= 0.5
                                            else 'Slightly Positive' if 0.5 > x['compound'] > 0
                                            else 'Neutral' if x['compound'] == 0
                                            else 'Slightly Negative' if 0 > x['compound'] >= -0.5
                                            else 'Highly Negative')

    st.write(df[[column, 'Sentiment']])

    # Word Cloud
    st.subheader('Word Cloud')
    texts = ' '.join(df[column].astype(str))
    wc = WordCloud(width=600, height=400, stopwords=stopwords.words('english')).generate(texts)
    st.image(wc.to_array(), use_container_width=True)

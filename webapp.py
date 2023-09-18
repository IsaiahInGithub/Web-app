import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Download NLTK stopwords
nltk.download('stopwords')

# Page config
st.set_page_config(page_title='Sentiment Analysis & Clustering App')

# Title
st.title('Sentiment Analysis & Clustering App')

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to classify sentiment
def classify_sentiment(compound):
    if compound >= 0.5:
        return "Highly Positive"
    elif compound >= 0:
        return "Positive"
    elif compound == 0:
        return "Neutral"
    elif compound > -0.5:
        return "Negative"
    else:
        return "Highly Negative"

# Function to generate a WordCloud
def generate_wordcloud(text):
    stopwords_set = set(stopwords.words('english'))
    wc = WordCloud(width=600, height=400, stopwords=stopwords_set).generate(text)
    return wc

# Function to process text and perform analysis
def process_text(text):
    if isinstance(text, str):  # Ensure text is a string
        scores = analyzer.polarity_scores(text)
        sentiment = classify_sentiment(scores['compound'])
        doc = nlp(text)
        adjectives = []
        verbs = []
        proper_nouns = []
        common_nouns = []

        for token in doc:
            if token.pos_ == 'ADJ':
                adjectives.append(token.text)
            elif token.pos_ == 'VERB':
                verbs.append(token.text)
            elif token.pos_ == 'PROPN':
                proper_nouns.append(token.text)
            elif token.pos_ == 'NOUN':
                common_nouns.append(token.text)

        return sentiment, ', '.join(adjectives), ', '.join(verbs), ', '.join(proper_nouns), ', '.join(common_nouns)
    else:
        return None, None, None, None, None

# File upload
uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

if uploaded_file:

    # Load dataframe
    df = pd.read_csv(uploaded_file)

    # Column selection
    column = st.selectbox('Select column', df.columns)

    # Allow user to add custom stopwords
    custom_stopwords = st.text_area("Add Custom Stopwords (comma-separated)", "")
    custom_stopwords = [word.strip() for word in custom_stopwords.split(',') if word.strip()]

    # Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Initialize variables for clustering
    num_clusters = st.slider("Number of Clusters", 2, 10)
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.9,
        max_features=5000,
        stop_words=stopwords.words('english') + custom_stopwords,
        lowercase=False  # Prevent text from being converted to lowercase
    )

    # Custom tokenizer to better handle tokenization
    def custom_tokenizer(text):
        return text.split()

    tfidf_vectorizer.set_params(tokenizer=custom_tokenizer)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(tfidf_matrix)
    df['Cluster'] = kmeans.labels_

    # Initialize variables for sentiment analysis
    sentiment_results = []

    # Initialize dictionary to store cluster keywords
    cluster_keywords = {}

    # Iterate through data
    for index, row in df.iterrows():
        text = row[column]
        sentiment, adjectives, verbs, proper_nouns, common_nouns = process_text(text)
        sentiment_results.append({
            'Response': text,
            'Sentiment': sentiment,
            'Adjectives': adjectives,
            'Verbs': verbs,
            'Proper Nouns': proper_nouns,
            'Common Nouns': common_nouns,
            'Compound': analyzer.polarity_scores(text)['compound'],
            'Cluster': row['Cluster']
        })

        # Update cluster keywords
        cluster_id = row['Cluster']
        if cluster_id not in cluster_keywords:
            cluster_keywords[cluster_id] = []
        cluster_keywords[cluster_id].extend(text.split())

    # Create a DataFrame from the results
    sentiment_df = pd.DataFrame(sentiment_results)

    # Display clustering results with cluster keywords
    st.write("### Clustering Results")

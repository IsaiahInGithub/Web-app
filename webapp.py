import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (if not already downloaded)
nltk.download("punkt")
nltk.download("stopwords")

# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and non-alphabetic tokens
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

    return " ".join(tokens)

# Title and About text
st.title("Text Analysis and Clustering App")
st.write(
    """
    This app performs basic text analysis and clustering on uploaded text data. 
    It includes sentiment analysis, word cloud generation, and text clustering.
    """
)

# Upload file
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column:", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Specify the column containing text data
    column = "text"

    # Preprocess the text data
    stop_words = set(stopwords.words("english"))
    df[column] = df[column].apply(preprocess_text)

    # Sentiment Analysis
    st.subheader("Sentiment Analysis")
    analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(text):
        sentiment_scores = analyzer.polarity_scores(text)
        return sentiment_scores

    df["Sentiment Scores"] = df[column].apply(analyze_sentiment)

    # Display sentiment analysis results
    st.write(df[["text", "Sentiment Scores"]])

    # Word Cloud
    st.subheader("Word Cloud")
    text_combined = " ".join(df[column])
    if text_combined:
        wc = WordCloud(width=600, height=400, stopwords=STOPWORDS).generate(text_combined)
        st.image(wc.to_array(), use_container_width=True)
    else:
        st.write("No text data available for word cloud analysis.")

    # Text Clustering
    st.subheader("Text Clustering")
    num_clusters = st.slider("Select the number of clusters:", 2, 10, 3)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(tfidf_matrix)

    df["Cluster"] = kmeans.labels_

    # Display clustering results
    st.write(df[["text", "Cluster"]])

    # Download CSV with clustering results
    csv_filename = f"{uploaded_file.name}_sentiment_and_clusters.csv"
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(
        f'<a href="data:file/csv;base64,{b64}" download="{csv_filename}">Download CSV File</a>',
        unsafe_allow_html=True,
    )

    # Plot cluster distribution
    cluster_distribution = df["Cluster"].value_counts()
    st.subheader("Cluster Distribution")
    plt.bar(cluster_distribution.index, cluster_distribution.values)
    st.pyplot()

    st.subheader("Cluster Keywords")
    cluster_keywords = {}
    for cluster_id in range(num_clusters):
        cluster_keywords[cluster_id] = " ".join(
            tfidf_vectorizer.get_feature_names_out()[kmeans.cluster_centers_.argsort()[cluster_id, ::-1][:10]]
        )

    for cluster_id, keywords in cluster_keywords.items():
        st.write(f"Cluster {cluster_id + 1} Keywords: {keywords}")

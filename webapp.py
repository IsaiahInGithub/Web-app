import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import plotly.express as px

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text data
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Function to get sentiment analysis
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.5:
        return "Highly Positive"
    elif 0.5 > sentiment_scores['compound'] > 0:
        return "Slightly Positive"
    elif sentiment_scores['compound'] == 0:
        return "Neutral"
    elif 0 > sentiment_scores['compound'] >= -0.5:
        return "Slightly Negative"
    else:
        return "Highly Negative"

# Main Streamlit app
def main():
    st.title("Text Analysis Dashboard")
    
    # Upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        # Sidebar with options
        st.sidebar.subheader("Text Analysis Options")
        
        # Dropdown for column selection
        column = st.sidebar.selectbox("Select a column for analysis", df.columns)
        
        # Check for empty dataframe or column
        if df.empty or df[column].dropna().empty:
            st.warning("Selected column is empty. Please choose another column.")
            return
        
        # Check if column contains text data
        if not np.issubdtype(df[column].dtype, np.number):
            df[column] = df[column].astype(str)
            df[column] = df[column].apply(preprocess_text)

            st.subheader("Sentiment Analysis")
            df['Sentiment'] = df[column].apply(get_sentiment)
            sentiment_counts = df['Sentiment'].value_counts()
            st.write(sentiment_counts)
            
            # Plot sentiment analysis results
            st.subheader("Sentiment Analysis Visualization")
            fig_sentiment = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index)
            st.plotly_chart(fig_sentiment, use_container_width=True)

            # Word Cloud
            st.subheader('Word Cloud')
            texts = ' '.join(df[column])
            if len(texts) > 0 and texts.strip():
                wc = WordCloud(width=600, height=400, stopwords=stopwords.words('english')).generate(texts)
                st.image(wc.to_array(), use_container_width=True)
            else:
                st.info("No text available for generating the Word Cloud.")
            
            # K-means Clustering
            st.subheader("K-means Clustering")
            num_clusters = st.sidebar.slider("Select the number of clusters", 2, 10)
            
            tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=1000)
            tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(tfidf_matrix)
            df['Cluster'] = kmeans.labels_
            
            st.write(df.groupby('Cluster').size())
            
            # Plot clusters
            st.subheader("K-means Clustering Visualization")
            scatter_plot = px.scatter(df, x="Cluster", color="Cluster", title="K-means Clustering")
            st.plotly_chart(scatter_plot, use_container_width=True)

            # Keywords in each cluster
            st.subheader("Keywords in Each Cluster")
            cluster_keywords = {}
            for cluster_id in range(num_clusters):
                cluster_keywords[cluster_id] = ', '.join(df[df['Cluster'] == cluster_id].head(10)[column])
                st.write(f"Cluster {cluster_id}: {cluster_keywords[cluster_id]}")

            # Download CSV
            st.subheader("Download Results")
            st.write("Download the analyzed data as a CSV file:")
            download_link = st.download_button(label="Download CSV", data=df.to_csv(), key='text_analysis')
            st.markdown(download_link, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import tempfile
import base64
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK stopwords and punkt
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Page config
st.set_page_config(page_title='Sentiment Analysis App')

# Title
st.title('Sentiment Analysis App')

def classify_sentiment(compound):
    if compound >= 0.5:
        return "Highly Positive"
    elif 0 < compound < 0.5:
        return "Slightly Positive"
    elif compound == 0:
        return "Neutral"
    elif -0.5 < compound < 0:
        return "Slightly Negative"
    else:
        return "Highly Negative"

def generate_wordcloud(text):
    stopwords_set = set(stopwords.words('english'))
    wc = WordCloud(width=600, height=400, stopwords=stopwords_set).generate(text)
    return wc

# File upload
uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

if uploaded_file:

    # Load dataframe
    df = pd.read_csv(uploaded_file)

    # Column selection
    column = st.selectbox('Select column', df.columns)

    # Allow user to add custom stopwords
    custom_stopwords = st.text_area("Custom Stopwords (comma-separated)", "")
    custom_stopwords = [word.strip() for word in custom_stopwords.split(',') if word.strip()]

    # Initialize analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Iterate through column
    results = []
    for text in df[column]:
        text = str(text) # Convert to string
        scores = analyzer.polarity_scores(text)
        sentiment = classify_sentiment(scores['compound'])
        
        # POS tagging
        tagged = nltk.pos_tag(word_tokenize(text))
        adjectives = [word for word, tag in tagged if tag.startswith('JJ')]
        verbs = [word for word, tag in tagged if tag.startswith('VB')]
        proper_nouns = [word for word, tag in tagged if tag == 'NNP' or tag == 'NNPS']
        common_nouns = [word for word, tag in tagged if tag == 'NN' or tag == 'NNS']
        
        data = {
            'Response': text,
            'Sentiment': sentiment,
            'Adjectives': ', '.join(adjectives),
            'Verbs': ', '.join(verbs),
            'Proper Nouns': ', '.join(proper_nouns),
            'Common Nouns': ', '.join(common_nouns),
            'Compound': scores['compound']
        }
        results.append(data)

    # Create a DataFrame from the results
    result_df = pd.DataFrame(results)

    # Perform K-Means clustering
    num_clusters = st.slider("Number of Clusters", 2, 10)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=5000, stop_words=stopwords.words('english') + custom_stopwords)
    tfidf_matrix = tfidf_vectorizer.fit_transform(result_df['Response'])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(tfidf_matrix)
    result_df['Cluster'] = kmeans.labels_

    # Display clustering results and keywords
    cluster_counts = result_df.groupby('Cluster').size()
    st.write("### Clustering Results")
    st.write(cluster_counts)

    cluster_keywords = {}
    for cluster_id in range(num_clusters):
        cluster_text = ' '.join(result_df[result_df['Cluster'] == cluster_id]['Response'])
        cluster_wc = generate_wordcloud(cluster_text)
        
        # Extract keywords from the WordCloud
        keywords = ', '.join([word for word, _ in cluster_wc.words_.items()][:10])
        cluster_keywords[cluster_id] = keywords

    st.write("### Keywords in Each Cluster")
    for cluster_id, keywords in cluster_keywords.items():
        st.write(f"Cluster {cluster_id}: {keywords}")
        
        # Add keywords to the table
        result_df.loc[result_df['Cluster'] == cluster_id, f'Cluster {cluster_id} Keywords'] = keywords

    # Display the table with keywords
    st.dataframe(result_df)

else:
    st.info('Upload a CSV file')

# About section
st.sidebar.header('About')
st.sidebar.markdown(
    """
    This is a Sentiment Analysis web app created with Streamlit. It allows you to upload a CSV file, 
    perform sentiment analysis on the text data, generate a word cloud, and identify different parts of speech in the text.

    **Sentiment Analysis:** The app classifies sentiment using the provided scale - Highly Positive, Slightly Positive, Neutral, Slightly Negative, and Highly Negative.

    **Word Cloud:** A word cloud is generated to visualize the most frequent words in the text.

    **Part of Speech:** Adjectives, Verbs, Proper Nouns, and Common Nouns are identified and displayed for each response in the table.

    **Custom Stopwords:** You can specify custom stopwords to filter out words during analysis.

    **Clustering:** K-Means clustering is performed to identify main themes in the data, and the percentage of data points in each cluster is displayed along with keywords in each cluster.

    **Download CSV:** You can download the analysis results as a CSV file with the same name as the uploaded file, followed by "_Sentiment Analysis".

    Enjoy exploring your text data with this interactive app!
    """
)

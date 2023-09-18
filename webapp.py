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
from nltk.tokenize import word_tokenize

# Page config
st.set_page_config(page_title='Text Analysis App')

# Title
st.title('Text Analysis App')

# About section
st.sidebar.subheader('About')
st.sidebar.write(
    """
    This is a Text Analysis Web App that performs various text analysis tasks on your data.
    You can upload a CSV file, select a column, and analyze the text within it.
    The following tasks are available:

    - **Sentiment Analysis**: Determine the sentiment of the text.
    - **Word Cloud**: Visualize the most frequent words in the text.
    - **Part-of-Speech Tagging**: Identify adjectives, verbs, proper nouns, and common nouns.
    - **Clustering**: Perform K-Means clustering on the text data to find common themes.

    You can also download the clustered data as a CSV file.
    """
)

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
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
        return ' '.join(tokens)

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

    # Part-of-Speech Tagging
    st.subheader('Part-of-Speech Tagging')
    pos_option = st.selectbox('Select part-of-speech', ['Adjectives', 'Verbs', 'Proper Nouns', 'Common Nouns'])
    doc = ' '.join(df[column].astype(str))
    doc = nlp(doc)
    
    pos_map = {
        'Adjectives': 'ADJ',
        'Verbs': 'VERB',
        'Proper Nouns': 'PROPN',
        'Common Nouns': 'NOUN'
    }

    if pos_option in pos_map:
        pos_tag = pos_map[pos_option]
        tagged_words = [token.text for token in doc if token.pos_ == pos_tag]
        st.write(f'{pos_option}:', ', '.join(tagged_words))

    # TF-IDF Vectorization and K-Means Clustering
    st.subheader('Clustering')
    num_clusters = st.number_input('Number of Clusters', min_value=2, max_value=10, value=2)
    cluster_button = st.button('Cluster Text Data')

    if cluster_button:
        # TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=0.05)
        tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])

        # K-Means Clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(tfidf_matrix)

        # Display clusters and keywords
        cluster_keywords = {}
        for cluster_id in range(num_clusters):
            cluster_data = df[df['Cluster'] == cluster_id]
            texts = ' '.join(cluster_data[column].astype(str))
            wc = WordCloud(width=200, height=200, stopwords=stopwords.words('english')).generate(texts)
            cluster_keywords[cluster_id] = ', '.join(wc.words_.keys()[:10])

            st.subheader(f'Cluster {cluster_id}')
            st.image(wc.to_array(), use_container_width=True)
            st.write('Keywords:', cluster_keywords[cluster_id])

        # Download clusters as CSV
        cluster_df = df[[column, 'Cluster']]
        csv_filename = f'clustered_data_{column}_sentiment.csv'
        csv_data = cluster_df.to_csv(index=False, encoding='utf-8')
        b64 = base64.b64encode(csv_data.encode()).decode()
        st.markdown("### Download Clustered Data")
        st.markdown(f"You can download the clustered data as a CSV file: [Download CSV](data:application/octet-stream;base64,{b64})")

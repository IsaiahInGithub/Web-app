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

  # Display results
  st.dataframe(result_df)

  # Generate wordcloud
  texts = ' '.join(df[column].astype(str))
  wc = generate_wordcloud(texts)

  # Save WordCloud image to a temporary file with a known extension (e.g., PNG)
  with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
      wc.to_file(temp_file.name)

  # Display the saved image
  st.image(temp_file.name)
  
  # Download the results as a CSV file
  csv_file = result_df.to_csv(index=False)
  b64 = base64.b64encode(csv_file.encode()).decode()  # Convert to base64
  filename = f"{uploaded_file.name.split('.')[0]}_Sentiment_Analysis.csv"
  href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
  st.markdown(href, unsafe_allow_html=True)

  # Perform K-Means clustering
  num_clusters = st.slider("Number of Clusters", 2, 10)
  tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=5000)
  tfidf_matrix = tfidf_vectorizer.fit_transform(result_df['Response'])
  kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(tfidf_matrix)
  result_df['Cluster'] = kmeans.labels_

  # Display clustering results
  st.write("### Clustering Results")
  st.write(result_df.groupby('Cluster').size())
  st.bar_chart(result_df['Cluster'].value_counts() / len(result_df) * 100)

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

    **Clustering:** K-Means clustering is performed to identify main themes in the data, and the percentage of data points in each cluster is displayed.

    **Download CSV:** You can download the analysis results as a CSV file with the same name as the uploaded file, followed by "_Sentiment Analysis".

    Enjoy exploring your text data with this interactive app!
    """
)

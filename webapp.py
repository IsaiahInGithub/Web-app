import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import tempfile

# Download NLTK stopwords and punkt
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Page config
st.set_page_config(page_title='Sentiment Analysis App')

# Title
st.title('Sentiment Analysis App')

def get_sentiment(compound):
  if compound >= 0.5:
    return 'Positive'
  elif compound <= -0.5:
    return 'Negative'
  else:
    return 'Neutral'

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
    sentiment = get_sentiment(scores['compound'])
    
    data = {
      'Response': text,
      'Sentiment': sentiment,
      'Compound': scores['compound']
    }
    results.append(data)

  # Display results
  st.dataframe(pd.DataFrame(results))

  # Generate wordcloud
  texts = ' '.join(df[column].astype(str))
  wc = generate_wordcloud(texts)

  # Save WordCloud image to a temporary file with a known extension (e.g., PNG)
  with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
      wc.to_file(temp_file.name)

  # Display the saved image
  st.image(temp_file.name)

  # POS tagging
  pos_selection = st.selectbox("Select Part of Speech", ["Proper Nouns", "Common Nouns", "Verbs", "Adjectives"])
  
  tagged = nltk.pos_tag(word_tokenize(texts))
  
  if pos_selection == "Proper Nouns":
      pos_words = [word for word, tag in tagged if tag == 'NNP' or tag == 'NNPS']
  elif pos_selection == "Common Nouns":
      pos_words = [word for word, tag in tagged if tag == 'NN' or tag == 'NNS']
  elif pos_selection == "Verbs":
      pos_words = [word for word, tag in tagged if tag.startswith('VB')]
  elif pos_selection == "Adjectives":
      pos_words = [word for word, tag in tagged if tag.startswith('JJ')]
  else:
      pos_words = []

  st.write(f'Selected {pos_selection}:', ', '.join(pos_words))
  
else:
  st.info('Upload a CSV file')

# About section
st.sidebar.header('About')
st.sidebar.markdown("""
This is a Sentiment Analysis web app created with Streamlit. It allows you to upload a CSV file, perform sentiment analysis on the text data, generate a word cloud, and identify different parts of speech in the text.

**Sentiment Analysis:** The app uses VADER Sentiment Analysis to classify the sentiment of each response as Positive, Negative, or Neutral.

**Word Cloud:** A word cloud is generated to visualize the most frequent words in the text.

**Part of Speech:** You can select from Proper Nouns, Common Nouns, Verbs, and Adjectives to identify specific parts of speech in the text.

Enjoy exploring your text data with this interactive app!
""")

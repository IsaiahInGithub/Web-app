import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import tempfile

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')

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

  # Save WordCloud image to a temporary file
  with tempfile.NamedTemporaryFile(delete=False) as temp_file:
      wc.to_file(temp_file.name)

  # Display the saved image
  st.image(temp_file.name, use_container_width=True)

  # POS tagging
  tagged = nltk.pos_tag(word_tokenize(str(texts)))
  adjectives = get_adjectives(tagged)
  nouns = get_nouns(tagged)

  st.write('Adjectives:', ', '.join(adjectives))
  st.write('Nouns:', ', '.join(nouns))
  
else:
  st.info('Upload a CSV file')

def get_adjectives(tagged):
  return [word for word, tag in tagged if tag.startswith('JJ')]

def get_nouns(tagged):
  return [word for word, tag in tagged if tag.startswith('NN')]
  
# About section
st.sidebar.header('About')
st.sidebar.info('Sentiment analysis web app using Streamlit')

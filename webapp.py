# Import libraries
import streamlit as st
import pandas as pd 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title='Sentiment Analysis App', page_icon=':smiley:', layout='wide') 

# Download NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Header 
st.title('Sentiment Analysis App :sunglasses:')

# Input text 
user_input = st.text_area("Enter text", height=200, value='Sample input text')

# VADER initialization
analyzer = SentimentIntensityAnalyzer()

# Sentiment analysis function
def analyze_sentiment(text):
  scores = analyzer.polarity_scores(text)
  compound = scores['compound']
  
  if compound >= 0.5:
    return 'Positive'
  elif compound <= -0.5:
    return 'Negative'
  else:
    return 'Neutral'

# Wordcloud function
def generate_wordcloud(text):
    stopwords_list = stopwords.words('english')
    wordcloud = WordCloud(width=600, height=400, stopwords=stopwords_list).generate(text)

    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(wordcloud)
    ax.axis('off')

    return fig

# Parts of speech function
def pos_tag_text(text):
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)  
    adjectives = [word for word, tag in tagged if tag.startswith('JJ')]
    nouns = [word for word, tag in tagged if tag.startswith('NN')]
    
    return adjectives, nouns

if st.button('Analyze Sentiment'):
    if user_input:
        sentiment = analyze_sentiment(user_input)
        st.success(f'Sentiment: {sentiment}')
    else:
        st.warning('Please enter text for analysis')
        
if st.button('Generate Word Cloud'):
    if user_input:
        fig = generate_wordcloud(user_input)
        st.pyplot(fig)
    else:
        st.warning('Please enter text for word cloud')

adjectives, nouns = pos_tag_text(user_input)
st.write('Extracted Adjectives:', ', '.join(adjectives))
st.write('Extracted Nouns:', ', '.join(nouns))

# Sidebar
st.sidebar.header('About')
st.sidebar.info('This app performs sentiment analysis using VADER and also generates wordclouds and extracts parts of speech from input text.')

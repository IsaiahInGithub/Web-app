import streamlit as st
import pandas as pd 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud 
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title='Sentiment Analysis App')

# Title 
st.title('Sentiment Analysis App')

# File Upload
uploaded_file = st.file_uploader('Choose a CSV file', type=['csv']) 

if uploaded_file:
   # Load dataframe
   df = pd.read_csv(uploaded_file)  
   
   # Allow column selection
   column = st.selectbox('Select column', df.columns)
   texts = df[column]

   # Initialize VADER   
   analyzer = SentimentIntensityAnalyzer()
   
   # Sentiment analysis
   results = [] 
   for text in texts:
      scores = analyzer.polarity_scores(text)
      sentiment = get_sentiment(scores['compound'])
      data = {
         'Response': text,
         'Sentiment': sentiment,
         'Compound': scores['compound']
      }
      results.append(data)
      
   # Display sentiment analysis results   
   st.dataframe(results) 
   
   # Wordcloud
   wc = generate_wordcloud(texts)
   st.pyplot(wc)
   
   # Parts of speech
   adjectives, nouns = get_pos(texts)  
   st.write('Adjectives:', ', '.join(adjectives))
   st.write('Nouns:', ', '.join(nouns))
   
else:
   st.info('Upload CSV file')  

# Helper functions    
def get_sentiment(compound):
   if compound >= 0.5:
      return 'Positive'
   elif compound <= -0.5:
      return 'Negative'
   else:
      return 'Neutral'

def generate_wordcloud(texts):
   text = ' '.join(texts)
   stopwords = nltk.corpus.stopwords.words('english')
   wc = WordCloud(stopwords=stopwords).generate(text)
   return wc
   
def get_pos(texts):
   tagged = nltk.pos_tag(word_tokenize(texts))  
   adjectives = [word for word, tag in tagged if tag.startswith('JJ')]
   nouns = [word for word, tag in tagged if tag.startswith('NN')]
   return adjectives, nouns

# About section 
st.sidebar.header('About')
st.sidebar.info('This app performs sentiment analysis, generates wordcloud and extracts parts of speech from text columns in a CSV file.')

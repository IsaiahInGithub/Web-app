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

# Download NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Title
st.title('Sentiment Analysis App') 

# File upload
uploaded_file = st.file_uploader('Choose a CSV file', type=['csv'])

if uploaded_file:
  df = pd.read_csv(uploaded_file)

  # Show columns as checkboxes to select  
  cols = st.columns(len(df.columns))
  selected_column = cols[0].checkbox(df.columns[0], key='col0')

  for i, col in enumerate(df.columns[1:]):
    selected = cols[i+1].checkbox(col, key=f'col{i+1}')
    if selected:
      selected_column = col  
      
  if selected_column:
    user_input = df[selected_column]
  else:
    st.error('Please select a column')
    
else:
  # Text input
  user_input = st.text_area('Enter text') 

# VADER initialization
analyzer = SentimentIntensityAnalyzer()

# Sentiment analysis function  
def analyze_sentiment(text):
  score = analyzer.polarity_scores(text)['compound']
  if score >= 0.05:
    return 'Positive'
  elif score <= -0.05:
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

if st.button('Analyze Sentiment'):
  if user_input is not None:
    sentiment = analyze_sentiment(str(user_input))
    st.success(f'Sentiment: {sentiment}')
  else:
    st.warning('Please upload file or enter text')
    
if st.button('Generate Word Cloud'):
  if user_input is not None:
    fig = generate_wordcloud(str(user_input))
    st.pyplot(fig)
  else:
    st.warning('Please upload file or enter text')
    
st.sidebar.header('About')
st.sidebar.info('Sentiment analysis app using Streamlit')

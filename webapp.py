# Import necessary libraries
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Function to generate wordcloud
def generate_wordcloud(text):
  stopwords_list = stopwords.words('english')
  wordcloud = WordCloud(width=600, height=400, stopwords=stopwords_list).generate(text)
  
  fig, ax = plt.subplots(figsize=(10,8))
  ax.imshow(wordcloud)
  ax.axis('off')
  
  st.pyplot(fig)

# Function for parts of speech tagging  
def pos_tag_text(text):
  tokens = word_tokenize(text)
  tagged = nltk.pos_tag(tokens)
  adjectives = [word for word, tag in tagged if tag.startswith('JJ')]
  nouns = [word for word, tag in tagged if tag.startswith('NN')]
  
  return adjectives, nouns

# Rest of app code 

...

if st.button("Generate Wordcloud"):
  generate_wordcloud(user_input)

adjectives, nouns = pos_tag_text(user_input)
st.write("Adjectives:", ", ".join(adjectives))  
st.write("Nouns:", ", ".join(nouns))

...

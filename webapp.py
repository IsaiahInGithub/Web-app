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
nltk.download('punkt')  # Download data for sentence tokenization
nltk.download('averaged_perceptron_tagger')  # Download data for the tagger

# Header 
st.title('Sentiment Analysis App :sunglasses:')

# Option to upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# VADER initialization
analyzer = SentimentIntensityAnalyzer()

# Sentiment analysis function
def analyze_sentiment(text):
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = float(sentiment_scores['compound'])
  
    if compound_score >= 0.5:
        return "Highly Positive"
    elif compound_score < 0.5 and compound_score > 0:
        return "Slightly Positive"
    elif compound_score == 0:
        return "Neutral"
    elif compound_score < 0 and compound_score >= -0.5:
        return "Slightly Negative"
    elif compound_score < -0.5:
        return "Highly Negative"
    else:
        return "Unknown"

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
    # Ensure that NLTK's data for sentence tokenization is available
    nltk.data.find('tokenizers/punkt')
    
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)  
    adjectives = [word for word, tag in tagged if tag.startswith('JJ')]
    nouns = [word for word, tag in tagged if tag.startswith('NN')]
    
    return adjectives, nouns

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded CSV Data:")
    st.write(df)
    
    # Allow users to choose a column for analysis
    selected_column = st.selectbox("Select a column for analysis:", df.columns)

    if selected_column:
        # Perform sentiment analysis on the selected column
        st.subheader(f'Sentiment Analysis for "{selected_column}":')
        df['Sentiment'] = df[selected_column].apply(analyze_sentiment)
        st.write(df[[selected_column, 'Sentiment']])

        # Generate word cloud from the selected column
        st.subheader(f'Word Cloud for "{selected_column}":')
        text = ' '.join(df[selected_column].astype(str))
        fig = generate_wordcloud(text)
        st.pyplot(fig)

# Option to enter text
user_input = st.text_area("Enter text", height=200, value='Sample input text')

if st.button('Analyze Sentiment'):
    if user_input:
        st.subheader('Sentiment Analysis Result:')
        sentiment = analyze_sentiment(user_input)
        st.write(f'Sentiment: {sentiment}')
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
st.sidebar.info('This app performs sentiment analysis using VADER, generates word clouds, and extracts parts of speech from input text or uploaded CSV files.')

# Import necessary libraries
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Sentiment Analysis Tool with VADER",
    page_icon=":speech_balloon:",
)

# Download the VADER lexicon (only needed once)
nltk.download('vader_lexicon')

# Create a Streamlit web app
st.title("Sentiment Analysis Tool with VADER")

# Create a textarea for user input
st.subheader("Enter text for sentiment analysis:")
user_input = st.text_area("")

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Define a function to analyze sentiment
def analyze_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)
    
    if float(sentiment_scores['compound']) >= 0.5:
        st.write("Sentiment: Highly Positive")
    if float(sentiment_scores['compound']) < 0.5 and float(sentiment_scores['compound']) > 0:
        st.write("Sentiment: Slightly Positive")
    if float(sentiment_scores['compound']) == 0:
        st.write("Sentiment: Neutral")
    if float(sentiment_scores['compound']) < 0 and float(sentiment_scores['compound']) >= -0.5:
        st.write("Sentiment: Slightly Negative")
    if float(sentiment_scores['compound']) < -0.5:
        st.write("Sentiment: Highly Negative")

# Create a button to analyze sentiment
if st.button("Analyze"):
    if user_input:
        sentiment = analyze_sentiment(user_input)
    else:
        st.warning("Please enter some text for analysis.")

# Add an about section (optional)
st.sidebar.subheader("About")
st.sidebar.info("This is a sentiment analysis tool using VADER and Streamlit.")

# Run the app
if __name__ == "__main__":
    st.set_option('deprecation.showfileUploaderEncoding', False)  # To hide a warning
    st.write("Enter some text and click 'Analyze' to perform sentiment analysis.")

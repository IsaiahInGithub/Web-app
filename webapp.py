# Import necessary libraries
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Download the VADER lexicon (only needed once)
nltk.download('vader_lexicon')

# Create a Streamlit web app
st.title("Sentiment Analysis Tool with VADER")

# Create a sidebar for file upload
st.sidebar.subheader("Upload a File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Text file", type=["csv", "txt"])

# Create a textarea for user input
st.subheader("Or Enter text for sentiment analysis:")
user_input = st.text_area("")

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Define a function to analyze sentiment
def analyze_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores

# Define a function to analyze sentiment for a DataFrame column
def analyze_sentiment_df(df, column_name):
    df['Sentiment'] = df[column_name].apply(analyze_sentiment)
    return df

# Create a button to analyze sentiment
if st.button("Analyze"):
    sentiment_table = []
    if user_input:
        sentiment_scores = analyze_sentiment(user_input)
        sentiment_table.append(sentiment_scores)
    else:
        st.warning("Please enter some text for analysis.")

# Perform sentiment analysis on uploaded file (if provided)
if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "csv":
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Sentiment Analysis for Uploaded CSV File")
            st.write(df)
            # Replace 'YourColumnName' with the actual column name containing text data
            df = analyze_sentiment_df(df, 'YourColumnName')
            sentiment_table = df['Sentiment'].tolist()
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
    elif file_extension == "txt":
        file_content = uploaded_file.read()
        st.subheader("Sentiment Analysis for Uploaded Text File")
        st.write(file_content)
        sentiment_scores = analyze_sentiment(file_content)
        sentiment_table.append(sentiment_scores)

# Display sentiment scores in a table
if sentiment_table:
    st.subheader("Sentiment Scores")
    sentiment_df = pd.DataFrame(sentiment_table)
    st.write(sentiment_df)

# Add an about section (optional)
st.sidebar.subheader("About")
st.sidebar.info("This is a sentiment analysis tool using VADER and Streamlit.")

# Run the app
if __name__ == "__main__":
    st.set_option('deprecation.showfileUploaderEncoding', False)  # To hide a warning
    st.set_page_config(
        page_title="Sentiment Analysis Tool with VADER",
        page_icon=":speech_balloon:",
    )
    st.write("Enter some text, click 'Analyze,' or upload a file for sentiment analysis.")

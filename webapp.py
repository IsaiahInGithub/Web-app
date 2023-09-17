# Import necessary libraries
import streamlit as st
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Sentiment Analysis Tool with VADER",
    page_icon=":speech_balloon:",
)

uploaded_file = st.sidebar.file_uploader("Choose a CSV or Text file", type=["csv", "txt"])

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
        st.write("Highly Positive")
    if float(sentiment_scores['compound']) < 0.5 and float(sentiment_scores['compound']) > 0:
        st.write("Slightly Positive")
    if float(sentiment_scores['compound']) == 0:
        st.write("Neutral")
    if float(sentiment_scores['compound']) < 0 and float(sentiment_scores['compound']) >= -0.5:
        st.write("Slightly Negative")
    if float(sentiment_scores['compound']) < -0.5:
        st.write("Highly Negative")

def analyze_sentiment_df(df, column_name):
    df['Sentiment'] = df[column_name].apply(analyze_sentiment)
    return df

    # Create a button to analyze sentiment
    if st.button("Analyze"):
        if user_input:
            sentiment = analyze_sentiment(user_input)
        else:
            st.warning("Please enter some text for analysis.")
    
if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "csv":
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Sentiment Analysis for Uploaded CSV File")
            st.write(df)
            df = analyze_sentiment_df(df, 'Comments')  # Assuming 'Text' is the column with text data
            st.write(sentiment + ": " + df)
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
    elif file_extension == "txt":
        file_content = uploaded_file.read()
        st.subheader("Sentiment Analysis for Uploaded Text File")
        st.write(file_content)
        sentiment = analyze_sentiment(file_content)
        st.write(f"Sentiment: {sentiment}")

# Add an about section (optional)
st.sidebar.subheader("About")
st.sidebar.info("This is a sentiment analysis tool using VADER and Streamlit.")

# Run the app
if __name__ == "__main__":
    st.set_option('deprecation.showfileUploaderEncoding', False)  # To hide a warning
    st.write("Enter some text and click 'Analyze' to perform sentiment analysis.")

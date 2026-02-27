import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Sentiment Intelligence", layout="wide")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

st.title("AI Sentiment Intelligence Platform")
st.write("Transformer-Based Deep Learning Sentiment Analyzer")

# -------- Single Text Analysis --------
st.header("Single Text Analysis")

text = st.text_area("Enter text")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter text.")
    else:
        result = classifier(text)
        label = result[0]["label"]
        score = result[0]["score"]

        st.subheader("Prediction")
        st.write("Sentiment:", label)
        st.write("Confidence:", round(score * 100, 2), "%")

        # Probability bar
        st.progress(float(score))

# -------- Batch Analysis --------
st.header("Batch Analysis (Multiple Sentences)")

uploaded_file = st.file_uploader("Upload CSV file with a column named 'text'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("CSV must contain a column named 'text'")
    else:
        predictions = classifier(list(df["text"]))
        df["sentiment"] = [p["label"] for p in predictions]
        df["confidence"] = [round(p["score"] * 100, 2) for p in predictions]

        st.subheader("Results")
        st.dataframe(df)

        # Sentiment Distribution Chart
        st.subheader("Sentiment Distribution")
        sentiment_counts = df["sentiment"].value_counts()

        fig, ax = plt.subplots()
        sentiment_counts.plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

        # Download option
        st.download_button(
            "Download Results CSV",
            df.to_csv(index=False),
            file_name="sentiment_results.csv",
            mime="text/csv"
        )
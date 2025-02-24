import streamlit as st
import pandas as pd
import joblib
import ast

# Load ML Model & TF-IDF Vectorizer
@st.cache_resource
def load_model():
    return joblib.load("tagging_model.pkl"), joblib.load("tfidf_vectorizer.pkl")

model, vectorizer = load_model()

# Title
st.title("🚀 Automated Tagging for Stack Overflow")
st.write("Enter a question or description to get predicted tags.")

# Text Input
text_input = st.text_area("Enter text:", "")

# Prediction Function
def predict_tags(text):
    transformed_text = vectorizer.transform([text])
    predicted_tags = model.predict(transformed_text)
    return predicted_tags

# Predict Button
if st.button("Predict Tags"):
    if text_input:
        tags = predict_tags(text_input)
        st.success(f"Predicted Tags: {tags}")
    else:
        st.warning("Enter some text first!")

# File Upload for Bulk Prediction
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Check if 'text' column exists
    if "merchandise" in df.columns:
        df["predicted_tags"] = df["merchandise"].apply(lambda x: ", ".join(predict_tags(x)))
        st.write(df.head())  # Display sample results
        
        # Download Results
        csv_output = df.to_csv(index=False)
        st.download_button("Download Predictions", csv_output, "predicted_tags.csv", "text/csv")
    else:
        st.error("CSV must have a 'description' column.")

st.write("Built with ❤️ using Streamlit")

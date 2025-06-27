import streamlit as st
import pandas as pd
import joblib

# Load ML Model & TF-IDF Vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("tagging_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Title
st.title("🚀 Stack Overflow Tag Predictor")
st.write("Enter a programming question or upload a CSV file to predict tags.")

# Text Input
text_input = st.text_area("Enter a Stack Overflow question or post text:")

# Predict Single Entry
def predict_tags(text):
    transformed = vectorizer.transform([text])
    return model.predict(transformed)[0]  # Assuming model returns a list or array of tags

# Predict Button
if st.button("🔍 Predict Tags"):
    if text_input.strip():
        tags = predict_tags(text_input)
        st.success(f"🏷 Predicted Tags: {tags}")
    else:
        st.warning("⚠️ Please enter some text first.")

# File Upload for Bulk Predictions
st.subheader("📁 Bulk CSV Upload")
uploaded_file = st.file_uploader("Upload a CSV with a 'description' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "description" in df.columns:
        df["predicted_tags"] = df["description"].apply(lambda x: ", ".join(predict_tags(x)))
        st.write(df.head())

        # Downloadable CSV
        csv_output = df.to_csv(index=False)
        st.download_button("📥 Download CSV with Tags", csv_output, file_name="predicted_tags.csv", mime="text/csv")
    else:
        st.error("❌ The uploaded CSV must contain a 'description' column.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")

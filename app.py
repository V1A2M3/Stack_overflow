import streamlit as st
import pandas as pd
import joblib

# Load ML Model, Vectorizer & MultiLabelBinarizer
@st.cache_resource
def load_model():
    model = joblib.load("/mnt/data/tagging_model.pkl")
    vectorizer = joblib.load("/mnt/data/tfidf_vectorizer.pkl")
    mlb = joblib.load("/mnt/data/mlb.pkl")
    return model, vectorizer, mlb

model, vectorizer, mlb = load_model()

# App Title
st.title("🚀 Stack Overflow Tag Predictor")
st.write("Enter a programming question or upload a CSV file to predict tags.")

# Predict tags for a single text entry
def predict_tags(text):
    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)
    tags = mlb.inverse_transform(prediction)
    return tags[0] if tags else []

# --- Single Text Input ---
text_input = st.text_area("Enter a Stack Overflow question or post text:")

if st.button("🔍 Predict Tags"):
    if text_input.strip():
        tags = predict_tags(text_input)
        st.success(f"🏷 Predicted Tags: {', '.join(tags)}")
    else:
        st.warning("⚠️ Please enter some text first.")

# --- Bulk CSV Upload ---
st.subheader("📁 Bulk CSV Upload")
uploaded_file = st.file_uploader("Upload a CSV with a 'description' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "description" in df.columns:
        df["predicted_tags"] = df["description"].apply(lambda x: ", ".join(predict_tags(x)))
        st.write(df.head())

        csv_output = df.to_csv(index=False)
        st.download_button("📥 Download CSV with Tags", csv_output, file_name="predicted_tags.csv", mime="text/csv")
    else:
        st.error("❌ The uploaded CSV must contain a 'description' column.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")

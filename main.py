import streamlit as st
from inference import predict

st.title("📰 News Topic Classifier (BERT)")
st.write("Enter a news headline and get its category.")

user_input = st.text_area("News Headline:")

if st.button("Classify"):
    label, probs = predict(user_input)
    st.success(f"Predicted Category: {label}")
    st.write("Probabilities:")
    for i, cat in enumerate(["World", "Sports", "Business", "Sci/Tech"]):
        st.write(f"{cat}: {probs[0][i]:.4f}")

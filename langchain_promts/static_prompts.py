import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = GoogleGenerativeAI(model="gemini-3.5-flash")

st.title("Research paper summarizer")

input_text = st.text_input("Enter the research paper text:", placeholder="Paste your research paper text here...")
if st.button("Summarize"):
    if input_text:
        response = model.invoke(f"Summarize the following research paper text:\n{input_text}")
        st.subheader("Summary:")
        st.write(response)
    else:
        st.warning("Please enter the research paper text to summarize.")
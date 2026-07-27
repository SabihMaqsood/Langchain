from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

model = GoogleGenerativeAI(model="gemini-3.5-flash")

st.title("AI Cooking Assistant")

Human_msg = st.text_input("Ask your cooking question")

template = PromptTemplate(
    template="""
You are a professional cooking assistant.

Answer the user's cooking question in a simple and friendly way.

User Question:
{Human_msg}

Instructions:
- Give clear steps.
- Suggest useful cooking tips.
- Keep the answer easy to understand.
""",
    input_variables=["Human_msg"]
)

# Fill Prompt
prompt = template.invoke({"Human_msg": Human_msg})

# Button
if st.button("Get Answer"):

    if Human_msg:
        response = model.invoke(prompt)
        st.write(response)
    else:
        st.warning("Please enter your cooking question.")
        
        
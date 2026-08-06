import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Page config
st.set_page_config(page_title="Simple Chatbot", page_icon="💬")
st.title("💬 Simple AI Chatbot")

# Initialize model
@st.cache_resource
def get_model():
    return GoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

model = get_model()

# Create prompt template
prompt_template = PromptTemplate(
    template="""You are a helpful assistant. Answer the following question:
    
Question: {question}

Answer:""",
    input_variables=["question"]
)

# Create chain
chain = prompt_template | model | StrOutputParser()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if user_input := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({"question": user_input})
            st.write(response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
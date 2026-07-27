import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page setup
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")
st.title("🤖 Gemini AI Chatbot")
st.markdown("### Ask me anything!")

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")

# Check if API key exists
if not api_key:
    st.error("⚠️ GOOGLE_API_KEY not found in .env file")
    st.stop()

# Initialize the Gemini model
@st.cache_resource
def get_model():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.7,
        google_api_key=api_key
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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Sidebar with controls
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Temperature control
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("Powered by Google Gemini")

# Update model with new temperature if changed
if temperature != model.temperature:
    model.temperature = temperature

# Get user input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chain.invoke({"question": user_input})
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")
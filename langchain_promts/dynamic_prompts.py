from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = GoogleGenerativeAI(model="gemini-3.5-flash")

st.title("Dynamic Prompt Research Paper Summarizer")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Create template
template = PromptTemplate(
    template="""
You are a research paper summarizer. Your task is to summarize the research paper based on the user's selections.

Paper Name: {paper_input}
Explanation Style: {style_input}
Explanation Length: {length_input}

Please provide a summary that includes:
1. Mathematical details: Include relevant mathematical equations and derivations.
2. Analogies and Examples: Provide analogies and examples to clarify complex concepts.
3. Case Studies: If the paper includes case studies or practical applications, summarize them as well.
""",
    input_variables=["paper_input", "style_input", "length_input"],
)

if st.button("Summarize"):
    # FIX: Use format() which returns a string directly
    prompt_string = template.format(
        paper_input=paper_input,
        style_input=style_input,
        length_input=length_input
    )
    
    if prompt_string.strip():
        summary = model.invoke(prompt_string)
        st.write(summary)
    else:
        st.warning("Please fill in all fields")
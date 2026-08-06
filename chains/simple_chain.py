from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate   
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.4-flash")

parser = StrOutputParser()

# create tow diff prompts template

template1 = PromptTemplate(
    template="What is the capital of {country}?",
    input_variables=["country"]
)


template2 = PromptTemplate(
    template="How much its total area {text}?",
    input_variables=["text"]
)

chain = template1 | llm | parser | template2 | llm | parser

result = chain.invoke({"country": "Pakistan"})

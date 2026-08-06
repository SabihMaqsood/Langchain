from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = GoogleGenerativeAI(model="gemini-3.5-flash")

response = model.invoke("Write a poem about the ocean.")

print(response)
